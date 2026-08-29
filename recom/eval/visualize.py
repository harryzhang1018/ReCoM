"""Failure-case visualizer (deliverable 9): predicted vs Chrono contact points/normals and rollout traces."""
from __future__ import annotations

import numpy as np
import torch

from ..geometry.box_plane_analytic import CORNER_SIGNS
from ..geometry.transforms import quat_to_rotmat_np


def _box_edges(pos, quat, he):
    R = quat_to_rotmat_np(quat)
    c = (R @ (CORNER_SIGNS * he).T).T + pos
    edges = [(i, j) for i in range(8) for j in range(i + 1, 8) if bin(i ^ j).count("1") == 1]
    return c, edges


def plot_contact_frame(ax, pos, quat, he, gt: dict, pred: dict | None = None, title: str = ""):
    """3D plot of the box, Chrono contacts (green) and predicted contacts (red) with normals."""
    c, edges = _box_edges(np.asarray(pos), np.asarray(quat), np.asarray(he))
    for i, j in edges:
        ax.plot(*zip(c[i], c[j]), color="0.6", lw=1)
    L = 2.5 * float(np.max(he))
    xs, ys = np.meshgrid([-L, L], [-L, L])
    ax.plot_surface(xs + pos[0], ys + pos[1], np.zeros_like(xs), alpha=0.1, color="k")
    act = np.asarray(gt["active"]) > 0.5
    for p, n in zip(np.asarray(gt["p_box_world"])[act], np.asarray(gt["n"])[act]):
        ax.scatter(*p, color="g", s=30)
        ax.quiver(*p, *(0.3 * L * n), color="g", lw=1)
    if pred is not None:
        pa = np.asarray(pred["active"]) > 0.5
        R = quat_to_rotmat_np(np.asarray(quat))
        for pl, n in zip(np.asarray(pred["p_box_local"])[pa], np.asarray(pred["n"])[pa]):
            p = R @ pl + np.asarray(pos)
            ax.scatter(*p, color="r", marker="x", s=40)
            ax.quiver(*p, *(0.3 * L * n), color="r", lw=1)
    ax.set_title(title)
    ax.set_xlim(pos[0] - L, pos[0] + L)
    ax.set_ylim(pos[1] - L, pos[1] + L)
    ax.set_zlim(-0.2 * L, 1.8 * L)


@torch.no_grad()
def save_worst_frames(encoder, cache, device, out_path: str, n_frames: int = 6, max_frames: int = 50000) -> list[dict]:
    """Evaluate the encoder on frames with GT contact, pick the worst matched-point errors, save a figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from ..data.dataset import ContactQueryDataset, collate_dict
    from ..models.losses import hungarian_match
    from ..train.train_contact import gt_from_batch
    ds = ContactQueryDataset(cache, categories=[2, 3, 4, 5])
    idx = np.arange(len(ds))
    if len(idx) > max_frames:
        idx = np.sort(np.random.default_rng(0).choice(idx, max_frames, replace=False))
    errs, items = [], []
    for s in range(0, len(idx), 1024):
        chunk = [ds[i] for i in idx[s: s + 1024]]
        batch = {k: v.to(device) for k, v in collate_dict(chunk).items()}
        out = encoder(batch["half_extents"], batch["pos"], batch["quat"])
        gt = gt_from_batch(batch)
        perm = hungarian_match(out, gt, batch["half_extents"].min(-1).values)
        B, K = gt["active"].shape
        g = lambda x: torch.gather(x, 1, perm.view(B, K, *([1] * (x.dim() - 2))).expand(B, K, *x.shape[2:]))  # noqa: E731
        pe = ((g(out["p_box_local"]) - gt["p_box_local"]).norm(dim=-1) * gt["active"]).max(1).values
        miss = ((torch.sigmoid(g(out["logit"])) < 0.5).float() * gt["active"]).sum(1)
        score = pe + 0.05 * miss
        for i, it in enumerate(chunk):
            errs.append(float(score[i]))
            items.append((it, {k: v[i].cpu() for k, v in out.items() if k not in ("tokens", "slot_embedding")}, {k: v[i].cpu() for k, v in gt.items()}))
    order = np.argsort(errs)[::-1][:n_frames]
    fig = plt.figure(figsize=(4 * n_frames, 4))
    rows = []
    for j, i in enumerate(order):
        it, pr, gt = items[i]
        ax = fig.add_subplot(1, n_frames, j + 1, projection="3d")
        pred = {"active": torch.sigmoid(pr["logit"]).numpy(), "p_box_local": pr["p_box_local"].numpy(), "n": pr["n"].numpy()}
        gtd = {"active": gt["active"].numpy(), "p_box_world": it["c_p_box_world"].numpy(), "n": gt["n"].numpy()}
        plot_contact_frame(ax, it["pos"].numpy(), it["quat"].numpy(), it["half_extents"].numpy(), gtd, pred, title=f"ep{int(it['episode_index'])} k{int(it['step'])} err={errs[i]:.3f}")
        rows.append({"episode_index": int(it["episode_index"]), "step": int(it["step"]), "score": errs[i]})
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return rows


def plot_rollout(pred: np.ndarray, gt: np.ndarray, dt: float, out_path: str, title: str = "") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = np.arange(gt.shape[0]) * dt
    fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    axes[0].plot(t, gt[:, 2], "k", label="Chrono z")
    axes[0].plot(t, pred[:, 2], "r--", label="predicted z")
    axes[1].plot(t, np.linalg.norm(gt[:, 7:10], axis=1), "k", label="|v| Chrono")
    axes[1].plot(t, np.linalg.norm(pred[:, 7:10], axis=1), "r--", label="|v| predicted")
    axes[2].plot(t, np.linalg.norm(gt[:, 10:13], axis=1), "k", label="|w| Chrono")
    axes[2].plot(t, np.linalg.norm(pred[:, 10:13], axis=1), "r--", label="|w| predicted")
    for ax in axes:
        ax.legend(loc="upper right")
    axes[0].set_title(title)
    axes[2].set_xlabel("t [s]")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
