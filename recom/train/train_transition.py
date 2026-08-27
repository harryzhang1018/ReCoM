"""Experiments A/B/D/E: train the box transition model (state-only or contact-conditioned) and evaluate
teacher-forced one-step error plus closed-loop rollouts with selectable contact sources."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from ..config import dump_json
from ..data.dataset import CAT_FIRST_IMPACT, CAT_REBOUND, TransitionWindowDataset, collate_dict, compute_state_normalization
from ..models.transition import BoxTransitionModel, gt_contacts_from_batch
from .common import MetricsLogger, cosine_lr, load_caches, save_checkpoint, seed_all, to_device
from .rollout import ContactSource, contacts_from_encoder_output, evaluate_rollouts


def load_encoder(path: str, device):
    from ..models.contact_encoder import build_encoder
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = ck["config"]
    enc = build_encoder(cfg["encoder"], d_model=cfg["d_model"], latent_dim=cfg["latent_dim"]).to(device)
    enc.load_state_dict(ck["model_state_dict"])
    enc.eval()
    return enc, cfg["latent_dim"]


def batch_contacts(batch: dict, source: str, encoder, latent_dim: int, device, grad: bool = False, return_raw: bool = False):
    """Contacts for a (B,T) window from the requested source.  grad=True runs the learned encoder with autograd
    (joint fine-tuning); return_raw also returns the flat (B*T) encoder output for the retained contact loss."""
    if source == "gt":
        c = gt_contacts_from_batch(batch, latent_dim)
        return (c, None) if return_raw else c
    B, T, _ = batch["states"].shape
    st = batch["states"].reshape(B * T, 13)
    he = batch["half_extents"][:, None, :].expand(B, T, 3).reshape(B * T, 3)
    if source == "analytic":
        from ..models.analytic_baseline import AnalyticBoxPlaneEncoder
        enc = AnalyticBoxPlaneEncoder(latent_dim=max(latent_dim, 1)).to(device)
        with torch.no_grad():
            out = enc(he, st[:, 0:3], st[:, 3:7])
    else:
        with torch.set_grad_enabled(grad):
            if grad:
                out = encoder(he, st[:, 0:3], st[:, 3:7])
            else:  # chunk to bound memory (the point encoder's kNN block is ~2.5 MB per frame)
                outs = [encoder(he[i: i + 1024], st[i: i + 1024, 0:3], st[i: i + 1024, 3:7]) for i in range(0, B * T, 1024)]
                out = {k: torch.cat([o[k] for o in outs]) for k in outs[0] if k != "tokens"}
    c = contacts_from_encoder_output(out, st[:, 0:3], hard=not grad)
    c = {k: v.reshape(B, T, *v.shape[1:]) for k, v in c.items()}
    if latent_dim == 0:
        c.pop("latent")
    return (c, out) if return_raw else c


def window_contact_loss(out: dict, batch: dict) -> torch.Tensor:
    """Explicit contact-set loss of the encoder on a (B,T) window vs the recorded Chrono contacts (plan Sec. 13:
    retained during joint fine-tuning so the latent cannot silently stop representing contact geometry)."""
    from ..models.losses import contact_set_loss
    B, T = batch["c_active"].shape[:2]
    pos = batch["states"][..., 0:3].reshape(B * T, 3)
    origin = torch.cat([pos[:, :2], torch.zeros_like(pos[:, 2:3])], -1)[:, None, :]
    gt = {"active": batch["c_active"].reshape(B * T, -1), "d": batch["c_d"].reshape(B * T, -1),
          "p_box_local": batch["c_p_box_local"].reshape(B * T, -1, 3), "p_ground_rel": batch["c_p_ground_world"].reshape(B * T, -1, 3) - origin,
          "n": batch["c_n"].reshape(B * T, -1, 3), "n_contacts": batch["c_n_contacts"].reshape(B * T)}
    scale = batch["half_extents"].min(-1).values[:, None].expand(B, T).reshape(B * T)
    loss, _ = contact_set_loss(out, gt, scale)
    return loss


@torch.no_grad()
def one_step_eval(model, cache, device, T: int, source: str, encoder, latent_dim: int, max_windows: int = 20000) -> dict:
    model.eval()
    ds = TransitionWindowDataset(cache, T, stride=T)
    idx = np.arange(len(ds))
    if len(idx) > max_windows:
        idx = np.sort(np.random.default_rng(0).choice(idx, max_windows, replace=False))
    err_v, err_w, cats = [], [], []
    for s in range(0, len(idx), 64):
        batch = to_device(collate_dict([ds[i] for i in idx[s: s + 64]]), device)
        c = batch_contacts(batch, source, encoder, latent_dim, device) if model.contact_mode != "none" else None
        d = model.predict_delta(batch["states"], batch["half_extents"], c)
        e = (d - batch["targets"]).reshape(-1, 6)
        err_v.append(e[:, :3].norm(dim=-1).cpu().numpy())
        err_w.append(e[:, 3:].norm(dim=-1).cpu().numpy())
        cats.append(batch["category"].reshape(-1).cpu().numpy())
    err_v, err_w, cats = map(np.concatenate, (err_v, err_w, cats))
    from ..data.dataset import CAT_NAMES
    res = {"dv_err_mean": float(err_v.mean()), "dw_err_mean": float(err_w.mean())}
    for c_, name in enumerate(CAT_NAMES):
        m = cats == c_
        if m.any():
            res[f"dv_err_{name}"] = float(err_v[m].mean())
            res[f"dw_err_{name}"] = float(err_w[m].mean())
    model.train()
    return res


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/fixed1a")
    ap.add_argument("--out", default="runs/transition_state_only")
    ap.add_argument("--contact-mode", default="none", choices=["none", "explicit", "latent", "explicit+latent"])
    ap.add_argument("--train-contact-source", default="gt", choices=["gt", "analytic", "learned"])
    ap.add_argument("--eval-contact-sources", default="gt,analytic", help="comma list of gt,analytic,learned")
    ap.add_argument("--encoder-ckpt", default=None)
    ap.add_argument("--block-size", type=int, default=32)
    ap.add_argument("--n-layer", type=int, default=3)
    ap.add_argument("--n-embd", type=int, default=128)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--event-weight", type=float, default=4.0, help="oversampling weight for windows containing impacts")
    ap.add_argument("--rollout-horizon", type=int, default=0, help="Experiment E: extra k-step unrolled loss")
    ap.add_argument("--rollout-gamma", type=float, default=0.9)
    ap.add_argument("--no-gravity-prior", action="store_true", help="disable the known free-flight residual prior")
    ap.add_argument("--loss", default="mse", choices=["mse", "huber"])
    ap.add_argument("--no-contact-gate", action="store_true", help="disable gating the residual by contact activation")
    ap.add_argument("--eval-only", default=None, help="path to a transition checkpoint: skip training, only evaluate")
    ap.add_argument("--finetune-encoder", action="store_true", help="Experiment E: let gradients reach the encoder")
    ap.add_argument("--contact-loss-weight", type=float, default=1.0, help="Experiment E: weight of the retained explicit contact loss")
    ap.add_argument("--eval-every", type=int, default=2000)
    ap.add_argument("--max-train-episodes", type=int, default=None)
    ap.add_argument("--max-eval-episodes", type=int, default=64)
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
    T = a.block_size

    caches = load_caches(a.data, ["train", "val", "test", "test_geometry"])
    if a.max_train_episodes:
        caches["train"].episodes = caches["train"].episodes[: a.max_train_episodes]
    for k in ("val", "test", "test_geometry"):
        if caches[k] is not None:
            caches[k].episodes = caches[k].episodes[: a.max_eval_episodes]
    encoder, latent_dim = (load_encoder(a.encoder_ckpt, device) if a.encoder_ckpt else (None, 0))
    if a.contact_mode in ("latent", "explicit+latent"):
        assert encoder is not None, "latent contact modes need --encoder-ckpt"
    if a.contact_mode == "explicit":
        latent_dim = 0
    if encoder is not None and not a.finetune_encoder:
        encoder.requires_grad_(False)
    meta0 = caches["train"].episodes[0].meta
    dt, g = meta0["dt"], meta0["episode"]["physics"]["gravity"]
    prior = None if a.no_gravity_prior else np.array([0.0, 0.0, -g * dt, 0.0, 0.0, 0.0])
    norm = compute_state_normalization(caches["train"], prior)
    soft_gate = a.finetune_encoder and a.train_contact_source == "learned"
    model = BoxTransitionModel(norm, contact_mode=a.contact_mode, block_size=T, n_layer=a.n_layer, n_embd=a.n_embd, latent_dim=latent_dim, dt=dt,
                               gravity_prior=not a.no_gravity_prior, gravity=g, contact_gate=not a.no_contact_gate, soft_gate=soft_gate).to(device)
    if a.eval_only:
        ck = torch.load(a.eval_only, map_location=device, weights_only=False)
        model.load_state_dict(ck["model_state_dict"])
        if a.finetune_encoder and encoder is not None and (Path(a.eval_only).parent / "encoder_finetuned.pt").exists():
            encoder.load_state_dict(torch.load(Path(a.eval_only).parent / "encoder_finetuned.pt", map_location=device, weights_only=False)["model_state_dict"])
        a.steps = 0
        print(f"eval-only: loaded {a.eval_only} (soft_gate={soft_gate})", flush=True)
    loss_fn = torch.nn.functional.mse_loss if a.loss == "mse" else torch.nn.functional.huber_loss
    print(f"transition model contact_mode={a.contact_mode} params={sum(p.numel() for p in model.parameters())}", flush=True)

    H = a.rollout_horizon
    ds = TransitionWindowDataset(caches["train"], T + H, stride=2)
    # event oversampling: windows containing first impact / repeated impact frames
    w = np.ones(len(ds))
    for i, (e_i, s) in enumerate(ds.index):
        cat = caches["train"].episodes[e_i].category[s: s + T + H]
        if np.isin(cat, (CAT_FIRST_IMPACT, CAT_REBOUND)).any():
            w[i] = a.event_weight
    sampler = WeightedRandomSampler(torch.from_numpy(w), num_samples=max(a.steps, 1) * a.batch, replacement=True, generator=torch.Generator().manual_seed(a.seed))
    dl = DataLoader(ds, batch_size=a.batch, sampler=sampler, collate_fn=collate_dict, num_workers=a.workers, drop_last=True, persistent_workers=a.workers > 0)
    params = list(model.parameters()) + (list(encoder.parameters()) if (encoder is not None and a.finetune_encoder) else [])
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=1e-4)
    src_train = ContactSource(a.train_contact_source, encoder, latent_dim=latent_dim)
    tstd = model.target_std

    t0 = time.time()
    step = 0
    for batch in (dl if a.steps > 0 else []):
        for g in opt.param_groups:
            g["lr"] = cosine_lr(step, a.steps, a.lr)
        batch = to_device(batch, device)
        full_states, full_next = batch["states"], batch["next_states"]
        win = {k: (v[:, :T] if torch.is_tensor(v) and v.dim() >= 2 and v.shape[1] == T + H else v) for k, v in batch.items()}
        joint = a.finetune_encoder and a.train_contact_source == "learned"
        c, enc_out = (batch_contacts(win, a.train_contact_source, encoder, latent_dim, device, grad=joint, return_raw=True) if model.contact_mode != "none" else (None, None))
        pred = model.norm_target(model.predict_delta(win["states"], win["half_extents"], c))
        loss_1 = loss_fn(pred, model.norm_target(win["targets"]))
        loss = loss_1
        loss_r = torch.zeros((), device=device)
        loss_c = torch.zeros((), device=device)
        if joint and enc_out is not None:
            loss_c = window_contact_loss(enc_out, win)
            loss = loss + a.contact_loss_weight * loss_c
        if H > 0:
            # Experiment E: unroll H steps from the window end with predicted states and recomputed contacts
            hist_s, hist_c = win["states"], c
            he = win["half_extents"]
            for k in range(1, H + 1):
                d = model.predict_delta(hist_s, he, hist_c)[:, -1]
                s_next = model.integrate(hist_s[:, -1], d)
                gt_next = full_next[:, T - 1 + k - 1]
                loss_r = loss_r + (a.rollout_gamma ** (k - 1)) * loss_fn((s_next[:, 7:13] - gt_next[:, 7:13]) / tstd, torch.zeros_like(tstd).expand_as(s_next[:, 7:13]))
                if hist_c is not None:
                    gt_c = {kk: batch["c_" + kk][:, T - 1 + k] for kk in ("active", "d", "n", "p_box_local")} if a.train_contact_source == "gt" else None
                    c_next = src_train(s_next, he, gt_c)
                    if latent_dim == 0:
                        c_next.pop("latent", None)
                    c_next = {kk: c_next[kk] for kk in hist_c}
                    hist_c = {kk: torch.cat([hist_c[kk][:, 1:], c_next[kk][:, None]], 1) for kk in hist_c}
                hist_s = torch.cat([hist_s[:, 1:], s_next[:, None]], 1)
            loss = loss + loss_r / H
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        step += 1
        if step % 100 == 0:
            logger.log({"step": step, "loss": loss.item(), "loss_1step": loss_1.item(), "loss_rollout": loss_r.item(), "loss_contact": loss_c.item(), "lr": opt.param_groups[0]["lr"], "t": time.time() - t0})
            print(f"step {step} loss {loss.item():.5f} (1-step {loss_1.item():.5f} rollout {loss_r.item():.5f} contact {loss_c.item():.4f}) {time.time() - t0:.0f}s", flush=True)
        if step % a.eval_every == 0 or step == a.steps:
            ev = one_step_eval(model, caches["val"], device, T, a.train_contact_source, encoder, latent_dim)
            logger.log({"step": step, "val_one_step": ev})
            print("  [val one-step] " + " ".join(f"{k}={v:.4g}" for k, v in ev.items()), flush=True)
            save_checkpoint(out / "last.pt", model, {"contact_mode": a.contact_mode, "block_size": T, "n_layer": a.n_layer, "n_embd": a.n_embd, "latent_dim": latent_dim, "normalization": {k: v.tolist() for k, v in norm.items()}, "encoder_ckpt": a.encoder_ckpt, "gravity_prior": not a.no_gravity_prior, "gravity": g, "dt": dt, "contact_gate": not a.no_contact_gate, "soft_gate": soft_gate})
        if step >= a.steps:
            break

    final = {"one_step": {}, "rollout": {}}
    for split in ("val", "test", "test_geometry"):
        if caches[split] is None:
            continue
        final["one_step"][split] = one_step_eval(model, caches[split], device, T, a.train_contact_source, encoder, latent_dim)
        for src in a.eval_contact_sources.split(","):
            if model.contact_mode == "none" and src != "gt":
                continue
            if src == "learned" and encoder is None:
                continue
            source = ContactSource(src, encoder, latent_dim=latent_dim)
            summ, rows = evaluate_rollouts(model, source, caches[split], device, T=T)
            final["rollout"][f"{split}/{src}"] = summ
            dump_json(rows, out / f"rollout_rows_{split}_{src}.json")
            keys = ["pos_err@100", "pos_err@500", "pos_err_final", "rot_err_deg_final", "v_err_mean", "w_err_mean", "impact_dv_err", "impact_dw_err", "apex_err", "max_penetration_pred", "artificial_energy_max"]
            print(f"[rollout {split}/{src}] " + " ".join(f"{k}={summ[k]['median']:.4g}" for k in keys if k in summ), flush=True)
    dump_json(final, out / "final_metrics.json")
    if encoder is not None and a.finetune_encoder:
        ck = torch.load(a.encoder_ckpt, map_location="cpu", weights_only=False)
        save_checkpoint(out / "encoder_finetuned.pt", encoder, ck["config"])
    save_checkpoint(out / "final.pt", model, {"contact_mode": a.contact_mode, "block_size": T, "n_layer": a.n_layer, "n_embd": a.n_embd, "latent_dim": latent_dim, "normalization": {k: v.tolist() for k, v in norm.items()}, "encoder_ckpt": a.encoder_ckpt, "gravity_prior": not a.no_gravity_prior, "gravity": g, "dt": dt, "contact_gate": not a.no_contact_gate, "soft_gate": soft_gate})


if __name__ == "__main__":
    main()
