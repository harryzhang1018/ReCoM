"""Stage ED2: pretrain the contact impulse decoder on teacher-forced recorded states with the (frozen) geometry
encoder in the loop.  Targets are the mass-normalized state-derived net wrench (recom.data.impulse_targets)."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..config import dump_json
from ..data.dataset import CAT_CONTACT, CAT_FIRST_IMPACT, CAT_NAMES, CAT_NEAR, CAT_REBOUND, CAT_REST, ImpulseFrameDataset, collate_dict, compute_wrench_normalization
from ..data.impulse_targets import LABEL_VERSION, PHYS_KEYS
from ..eval.metrics import aggregate_impulse_metrics, impulse_frame_metrics
from ..geometry.transforms import quat_mul, quat_to_rotmat
from ..models.impulse_decoder import ContactImpulseDecoder, build_decoder
from .common import MetricsLogger, cosine_lr, load_caches, save_checkpoint, seed_all, to_device
from .train_transition import batch_contacts, load_encoder

TRAIN_CATEGORIES = [CAT_NEAR, CAT_FIRST_IMPACT, CAT_CONTACT, CAT_REBOUND, CAT_REST]   # far_free never has an active slot -> masked from the loss
DECODER_CONFIG_KEYS = ("slot_embed_dim", "ctx_dim", "width", "n_blocks", "n_heads", "head_mode", "use_slot_embedding", "pooled_only", "out_scale", "K")


def batch_phys(batch: dict) -> dict[str, torch.Tensor]:
    return {k: batch[k] for k in PHYS_KEYS}


def yaw_augment(batch: dict, generator: torch.Generator | None = None) -> dict:
    """Rotate every sample of a (B,T,...) batch by a random yaw about world z (the box-plane contact problem is exactly
    invariant to it): quaternion (left-multiplied), world velocities, world contact quantities and the wrench labels.
    Box-local quantities (p_box_local, half_extents, phys) are unchanged; the geometry encoder is re-run on the rotated pose."""
    B = batch["states"].shape[0]
    st = batch["states"]
    yaw = torch.rand(B, generator=generator, device="cpu").to(st.device) * (2 * torch.pi)
    half = 0.5 * yaw
    qz = torch.stack([torch.cos(half), torch.zeros_like(half), torch.zeros_like(half), torch.sin(half)], -1)      # (B,4)
    R = quat_to_rotmat(qz)                                                                                          # (B,3,3)
    rot = lambda x: torch.einsum("bij,b...j->b...i", R, x)  # noqa: E731
    out = dict(batch)
    pos, quat, v, w = st[..., 0:3], st[..., 3:7], st[..., 7:10], st[..., 10:13]
    q2 = quat_mul(qz[:, None, :].expand_as(quat), quat)
    q2 = q2 * torch.sign(q2[..., :1] + 1e-12)
    out["states"] = torch.cat([rot(pos), q2, rot(v), rot(w)], -1)
    if "next_states" in batch:
        ns = batch["next_states"]
        q3 = quat_mul(qz[:, None, :].expand_as(ns[..., 3:7]), ns[..., 3:7])
        out["next_states"] = torch.cat([rot(ns[..., 0:3]), q3 * torch.sign(q3[..., :1] + 1e-12), rot(ns[..., 7:10]), rot(ns[..., 10:13])], -1)
        out["targets"] = torch.cat([rot(batch["targets"][..., :3]), rot(batch["targets"][..., 3:])], -1)
    for k in ("target_dv_contact", "target_dL_contact", "target_dw_contact", "target_j_slot", "c_n", "c_p_box_world", "c_p_ground_world"):
        if k in batch:
            out[k] = rot(batch[k])
    return out


def load_decoder(path: str, device) -> tuple[ContactImpulseDecoder, dict]:
    ck = torch.load(path, map_location=device, weights_only=False)
    dec = build_decoder(ck["config"]).to(device)
    dec.load_state_dict(ck["model_state_dict"])
    dec.eval()
    return dec, ck["config"]


def impulse_loss(out: dict, batch: dict, active: torch.Tensor, wrench_std: torch.Tensor, dL_std: torch.Tensor, delta: float = 1.0, label_thr: float = 1e-4,
                 tan_weight: float = 1.0, rest_eq_weight: float = 0.0, solver_anchor_weight: float = 0.0,
                 anchor_v0: float = 0.15, anchor_width: float = 0.05) -> tuple[torch.Tensor, dict]:
    """Huber loss on the normalized net wrench over frames where the geometry encoder activated at least one slot.
    Frames with a nonzero label but no active slot are encoder misses: excluded from the loss, counted as missed.

    tan_weight   : extra weight on the creep/spin channels of the main loss (tangential dv_xy and yaw dL_z; the plane
                   normal is world z, so the normal/tangential split is exact by component).
    rest_eq_weight: equilibrium regularizer on GT-resting frames: pull the *prediction* to the exact static balance
                   (dv = g dt ẑ, dL = 0) instead of Chrono's micro-oscillating resting labels (§9.12 creep).
    solver_anchor_weight: pull the net wrench toward the closed-form single-contact solver wrench on low-speed frames,
                   weight σ((v0 − s)/width) with s = |v| + |ω|·|he| (leaves impacts unconstrained)."""
    m = (active.amax(-1) > 0).float()                                                   # (...,)
    w_dv = torch.tensor([tan_weight, tan_weight, 1.0], device=m.device)
    w_dL = torch.tensor([1.0, 1.0, tan_weight], device=m.device)
    e_dv = (torch.nn.functional.huber_loss((out["dv_c"] - batch["target_dv_contact"]) / wrench_std[:3], torch.zeros_like(out["dv_c"]), reduction="none", delta=delta) * w_dv).sum(-1)
    e_dL = (torch.nn.functional.huber_loss((out["dL"] - batch["target_dL_contact"]) / dL_std, torch.zeros_like(out["dL"]), reduction="none", delta=delta) * w_dL).sum(-1)
    denom = m.sum().clamp_min(1.0)
    l_dv, l_dL = (e_dv * m).sum() / denom, (e_dL * m).sum() / denom
    label = batch["target_dv_contact"].norm(dim=-1) > label_thr
    missed = (label & (m == 0)).float().sum() / label.float().sum().clamp_min(1.0)
    loss = l_dv + l_dL
    parts = {"dv": l_dv.item(), "dL": l_dL.item(), "missed": missed.item(), "frac_eval": (m.sum() / m.numel()).item()}
    lead1 = [1] * (m.dim() - 1)
    if rest_eq_weight > 0:
        rest = (batch["category"] == CAT_REST).float() * m
        g_dt = (batch["gravity"] * batch["dt"]).view(-1, *lead1)
        e_eq = ((out["dv_c"][..., :2] / wrench_std[:2]).abs().sum(-1)
                + (out["dv_c"][..., 2] - g_dt).abs() / wrench_std[2]
                + (out["dL"] / dL_std).abs().sum(-1))
        l_eq = (e_eq * rest).sum() / rest.sum().clamp_min(1.0)
        loss = loss + rest_eq_weight * l_eq
        parts["eq"] = l_eq.item()
    if solver_anchor_weight > 0 and "wrench_solver" in out:
        st = batch["states"]
        s = st[..., 7:10].norm(dim=-1) + st[..., 10:13].norm(dim=-1) * batch["half_extents"].norm(dim=-1).view(-1, *lead1)
        wlow = torch.sigmoid((anchor_v0 - s) / anchor_width) * m
        e_anc = ((out["wrench"] - out["wrench_solver"]) / wrench_std).abs().sum(-1)
        l_anc = (e_anc * wlow).sum() / wlow.sum().clamp_min(1.0)
        loss = loss + solver_anchor_weight * l_anc
        parts["anc"] = l_anc.item()
    return loss, parts


@torch.no_grad()
def evaluate_decoder(decoder, encoder, cache, device, geometry_source: str = "learned", batch_size: int = 1024, max_frames: int | None = None) -> dict:
    decoder.eval()
    ds = ImpulseFrameDataset(cache)
    idx = np.arange(len(ds))
    if max_frames and len(idx) > max_frames:
        idx = np.sort(np.random.default_rng(0).choice(idx, max_frames, replace=False))
    acc: dict[str, list] = {}
    cats = []
    for s in range(0, len(idx), batch_size):
        batch = to_device(collate_dict([ds[i] for i in idx[s: s + batch_size]]), device)
        c = batch_contacts(batch, geometry_source, encoder, 0, device)
        out = decoder(c, batch["states"], batch["half_extents"], batch_phys(batch))
        m = impulse_frame_metrics(out, batch, c["active"], c["n"], batch["mu"][:, None].expand_as(c["active"][..., 0]))
        for k, v in m.items():
            acc.setdefault(k, []).append(v)
        cats.append(batch["category"].reshape(-1).cpu().numpy())
    decoder.train()
    return aggregate_impulse_metrics(acc, np.concatenate(cats))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/pilot1b")
    ap.add_argument("--out", default="runs/impulse_decoder")
    ap.add_argument("--encoder-ckpt", required=True)
    ap.add_argument("--geometry-source", default="learned", choices=["learned", "gt", "analytic"], help="contact geometry fed to the decoder during training")
    ap.add_argument("--head-mode", default="cone", choices=["cone", "free"])
    ap.add_argument("--pooled-only", action="store_true", help="POOL-DEC ablation: decode from the pooled context only")
    ap.add_argument("--no-slot-embedding", action="store_true", help="ablation: explicit geometry features only")
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--n-blocks", type=int, default=2)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--huber-delta", type=float, default=0.1, help="in normalized units; small = near-L1 so resting (0.2 sigma) and impact (50 sigma) frames get equal weight")
    ap.add_argument("--tan-weight", type=float, default=1.0, help="extra main-loss weight on the creep/spin channels (dv_xy, dL_z)")
    ap.add_argument("--rest-eq-weight", type=float, default=0.0, help="equilibrium regularizer on resting frames: prediction -> (g dt z_hat, 0)")
    ap.add_argument("--solver-anchor-weight", type=float, default=0.0, help="pull the wrench to the solver prior on low-speed frames")
    ap.add_argument("--anchor-v0", type=float, default=0.15, help="solver-anchor speed threshold (m/s; Chrono min_bounce_speed)")
    ap.add_argument("--anchor-width", type=float, default=0.05, help="solver-anchor sigmoid width (m/s)")
    ap.add_argument("--no-timing-feats", action="store_true", help="v1 decoder inputs (regressed d only, no exact contact-point height / velocity deficit)")
    ap.add_argument("--no-scaled-head", action="store_true", help="v1 head: fixed out_scale instead of the per-slot physical scale")
    ap.add_argument("--head-scale", default="vn", choices=["vn", "delassus", "solver"], help="per-slot scale: approach speed, the frictionless single-contact normal impulse, or the closed-form frictional single-contact solver (also adds its impulse as features)")
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--yaw-aug", action="store_true", help="random yaw rotation of every training sample (exact symmetry of the box-plane problem)")
    ap.add_argument("--no-chrono-gap", action="store_true", help="timing features from the plain point height instead of the envelope-corrected Chrono gap")
    ap.add_argument("--uniform-sampling", action="store_true", help="sample contact frames uniformly instead of balancing the categories (less repetition of the ~6k first-impact frames)")
    ap.add_argument("--eval-every", type=int, default=2000)
    ap.add_argument("--max-train-episodes", type=int, default=None)
    ap.add_argument("--max-eval-episodes", type=int, default=120)
    ap.add_argument("--max-eval-frames", type=int, default=200_000)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args(argv)
    seed_all(a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dump_json(vars(a), out / "args.json")
    logger = MetricsLogger(out)
    device = torch.device(a.device)

    caches = load_caches(a.data, ["train", "val", "test", "test_geometry"])
    if a.max_train_episodes:
        caches["train"].episodes = caches["train"].episodes[: a.max_train_episodes]
    for k in ("val", "test", "test_geometry"):
        if caches[k] is not None and a.max_eval_episodes:
            caches[k].episodes = caches[k].episodes[: a.max_eval_episodes]
    encoder, _ = load_encoder(a.encoder_ckpt, device)
    encoder.requires_grad_(False)
    wnorm = compute_wrench_normalization(caches["train"])
    wrench_std = torch.as_tensor(wnorm["wrench_std"], dtype=torch.float32, device=device)
    dL_std = torch.as_tensor(wnorm["dL_std"], dtype=torch.float32, device=device)
    slot_dim = int(encoder.decoder.queries.shape[-1]) if hasattr(encoder, "decoder") else 128
    decoder = ContactImpulseDecoder(slot_embed_dim=slot_dim, width=a.width, n_blocks=a.n_blocks, n_heads=a.n_heads, head_mode=a.head_mode,
                                    use_slot_embedding=not a.no_slot_embedding, pooled_only=a.pooled_only, out_scale=float(wrench_std[:3].mean()),
                                    timing_feats=not a.no_timing_feats, scaled_head=not a.no_scaled_head, head_scale=a.head_scale, dropout=a.dropout, chrono_gap_feats=not a.no_chrono_gap).to(device)
    cfg = {**decoder.config(), "encoder_ckpt": a.encoder_ckpt, "geometry_source": a.geometry_source, "normalization": {k: v.tolist() for k, v in wnorm.items()}, "label_version": LABEL_VERSION}
    print(f"impulse decoder params={sum(p.numel() for p in decoder.parameters())} wrench_std={wnorm['wrench_std'].round(4).tolist()} dL_std={wnorm['dL_std'].round(5).tolist()}", flush=True)

    ds = ImpulseFrameDataset(caches["train"], TRAIN_CATEGORIES)
    print("train category counts:", ds.category_counts(), flush=True)
    if a.uniform_sampling:
        from torch.utils.data import RandomSampler
        sampler = RandomSampler(ds, replacement=True, num_samples=a.steps * a.batch, generator=torch.Generator().manual_seed(a.seed))
    else:
        sampler = ds.balanced_sampler(a.steps * a.batch, seed=a.seed)
    dl = DataLoader(ds, batch_size=a.batch, sampler=sampler, collate_fn=collate_dict, num_workers=a.workers, drop_last=True, persistent_workers=a.workers > 0)
    opt = torch.optim.AdamW(decoder.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    aug_gen = torch.Generator().manual_seed(a.seed + 1)
    t0 = time.time()
    step = 0
    for batch in dl:
        for g in opt.param_groups:
            g["lr"] = cosine_lr(step, a.steps, a.lr)
        batch = to_device(batch, device)
        if a.yaw_aug:
            batch = yaw_augment(batch, aug_gen)
        c = batch_contacts(batch, a.geometry_source, encoder, 0, device)
        out_ = decoder(c, batch["states"], batch["half_extents"], batch_phys(batch))
        loss, parts = impulse_loss(out_, batch, c["active"], wrench_std, dL_std, a.huber_delta,
                                   tan_weight=a.tan_weight, rest_eq_weight=a.rest_eq_weight,
                                   solver_anchor_weight=a.solver_anchor_weight, anchor_v0=a.anchor_v0, anchor_width=a.anchor_width)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
        opt.step()
        step += 1
        if step % 100 == 0:
            logger.log({"step": step, "loss": loss.item(), **{"l_" + k: v for k, v in parts.items()}, "lr": opt.param_groups[0]["lr"], "t": time.time() - t0})
            print(f"step {step} loss {loss.item():.4f} " + " ".join(f"{k}={v:.3f}" for k, v in parts.items()) + f" ({time.time() - t0:.0f}s)", flush=True)
        if step % a.eval_every == 0 or step == a.steps:
            ev = evaluate_decoder(decoder, encoder, caches["val"], device, a.geometry_source, max_frames=a.max_eval_frames)
            logger.log({"step": step, "val": ev})
            print("  [val] " + " ".join(f"{k}={v}" for k, v in ev.items() if k != "by_category"), flush=True)
            save_checkpoint(out / "last.pt", decoder, cfg)
        if step >= a.steps:
            break
    final = {"impulse": {}}
    for name in ("val", "test", "test_geometry"):
        if caches[name] is None:
            continue
        final["impulse"][name] = evaluate_decoder(decoder, encoder, caches[name], device, a.geometry_source)
        if a.geometry_source == "learned":   # ablation 7: Chrono geometry instead of predicted geometry
            final["impulse"][name + "/gt_geometry"] = evaluate_decoder(decoder, encoder, caches[name], device, "gt")
        print(f"[{name}] " + " ".join(f"{k}={v}" for k, v in final["impulse"][name].items() if k != "by_category"), flush=True)
        fi = final["impulse"][name].get("by_category", {}).get(CAT_NAMES[CAT_FIRST_IMPACT])
        if fi:
            print(f"  first_impact: dv_mae={fi['dv_mae']} (zero baseline {fi['zero_baseline_dv_mae']}) dL_mae={fi['dL_mae']} (zero {fi['zero_baseline_dL_mae']}) missed={fi['missed_impulse_rate']}", flush=True)
    dump_json(final, out / "final_metrics.json")
    save_checkpoint(out / "final.pt", decoder, cfg)


if __name__ == "__main__":
    main()
