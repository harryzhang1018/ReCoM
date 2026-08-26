"""Experiment C: train a contact encoder (geometry + pose -> canonical contact set) on the balanced query view."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..config import dump_json
from ..data.dataset import ContactQueryDataset, collate_dict
from ..eval.metrics import aggregate_contact_metrics, contact_frame_metrics, first_impact_timing
from ..models.contact_encoder import build_encoder
from ..models.losses import contact_set_loss
from .common import MetricsLogger, cosine_lr, load_caches, save_checkpoint, seed_all, to_device


def gt_from_batch(batch: dict) -> dict:
    pos = batch["pos"]
    origin = torch.cat([pos[:, :2], torch.zeros_like(pos[:, 2:3])], -1)[:, None, :]
    return {
        "active": batch["c_active"], "d": batch["c_d"], "p_box_local": batch["c_p_box_local"],
        "p_ground_rel": batch["c_p_ground_world"] - origin, "n": batch["c_n"], "n_contacts": batch["c_n_contacts"],
    }


@torch.no_grad()
def evaluate_encoder(model, cache, device, batch_size: int = 2048, max_frames: int | None = None) -> dict:
    """Frame-level metrics over ALL frames of the cache's episodes + first-impact timing per episode."""
    model.eval()
    ds = ContactQueryDataset(cache)
    idx = np.arange(len(ds))
    if max_frames and len(idx) > max_frames:
        idx = np.random.default_rng(0).choice(idx, max_frames, replace=False)
        idx.sort()
    acc: dict[str, list] = {}
    cats = []
    frame_pred_active = {}
    for s in range(0, len(idx), batch_size):
        batch = to_device(collate_dict([ds[i] for i in idx[s: s + batch_size]]), device)
        out = model(batch["half_extents"], batch["pos"], batch["quat"])
        gt = gt_from_batch(batch)
        m = contact_frame_metrics(out, gt, batch["half_extents"].min(-1).values)
        for k, v in m.items():
            acc.setdefault(k, []).append(v)
        cats.append(batch["category"].cpu().numpy())
        pa = (torch.sigmoid(out["logit"]) > 0.5).any(1).cpu().numpy()
        for e, st, a in zip(batch["episode_index"].cpu().numpy(), batch["step"].cpu().numpy(), pa):
            frame_pred_active.setdefault(int(e), {})[int(st)] = bool(a)
    res = aggregate_contact_metrics(acc, np.concatenate(cats))
    if max_frames is None:
        errs = []
        for e_i, ep in enumerate(cache.episodes):
            fa = np.array([frame_pred_active[e_i].get(k, False) for k in range(ep.n_steps)])
            t = first_impact_timing(fa, ep.events["first_impact_step"])
            if t is not None:
                errs.append(abs(t))
        if errs:
            res["first_impact_err_steps_median"] = float(np.median(errs))
            res["first_impact_err_steps_p99"] = float(np.percentile(errs, 99))
            res["first_impact_missed"] = int(len(cache.episodes) - len(errs))
    model.train()
    return res


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/pilot1b")
    ap.add_argument("--encoder", default="patch", choices=["patch", "point", "analytic"])
    ap.add_argument("--out", default="runs/contact_patch")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--latent-dim", type=int, default=16)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--max-train-episodes", type=int, default=None)
    ap.add_argument("--max-eval-episodes", type=int, default=120)
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

    caches = load_caches(a.data, ["train", "val", "test", "test_geometry"], None)
    if a.max_train_episodes:
        caches["train"].episodes = caches["train"].episodes[: a.max_train_episodes]
    for k in ("val", "test", "test_geometry"):
        if caches[k] is not None and a.max_eval_episodes:
            caches[k].episodes = caches[k].episodes[: a.max_eval_episodes]
    model = build_encoder(a.encoder, **({} if a.encoder == "analytic" else {"d_model": a.d_model, "latent_dim": a.latent_dim})).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"encoder={a.encoder} params={n_params}", flush=True)

    if a.encoder != "analytic":
        ds = ContactQueryDataset(caches["train"])
        print("train category counts:", ds.category_counts(), flush=True)
        sampler = ds.balanced_sampler(a.steps * a.batch, seed=a.seed)
        dl = DataLoader(ds, batch_size=a.batch, sampler=sampler, collate_fn=collate_dict, num_workers=a.workers, drop_last=True, persistent_workers=a.workers > 0)
        opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
        t0 = time.time()
        step = 0
        for batch in dl:
            for g in opt.param_groups:
                g["lr"] = cosine_lr(step, a.steps, a.lr)
            batch = to_device(batch, device)
            out_ = model(batch["half_extents"], batch["pos"], batch["quat"])
            loss, parts = contact_set_loss(out_, gt_from_batch(batch), batch["half_extents"].min(-1).values)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1
            if step % 100 == 0:
                logger.log({"step": step, "loss": loss.item(), **{"l_" + k: v for k, v in parts.items()}, "lr": opt.param_groups[0]["lr"], "t": time.time() - t0})
                print(f"step {step} loss {loss.item():.4f} " + " ".join(f"{k}={v:.3f}" for k, v in parts.items()) + f" ({time.time() - t0:.0f}s)", flush=True)
            if step % a.eval_every == 0 or step == a.steps:
                ev = evaluate_encoder(model, caches["val"], device, max_frames=200_000)
                logger.log({"step": step, "val": ev})
                print(f"  [val] " + " ".join(f"{k}={v}" for k, v in ev.items() if k != "by_category"), flush=True)
                save_checkpoint(out / "last.pt", model, {"encoder": a.encoder, "d_model": a.d_model, "latent_dim": a.latent_dim})
            if step >= a.steps:
                break
    # final full evaluation
    final = {}
    for name in ("val", "test", "test_geometry"):
        if caches[name] is None:
            continue
        final[name] = evaluate_encoder(model, caches[name], device)
        print(f"[{name}] " + " ".join(f"{k}={v}" for k, v in final[name].items() if k != "by_category"), flush=True)
    dump_json(final, out / "final_metrics.json")
    if a.encoder != "analytic":
        save_checkpoint(out / "final.pt", model, {"encoder": a.encoder, "d_model": a.d_model, "latent_dim": a.latent_dim})


if __name__ == "__main__":
    main()
