#!/usr/bin/env python
"""Overlay video: Chrono ground-truth box drop vs the neural simulator's closed-loop rollout.

Renders the recorded Chrono episode (blue, solid) and the transition model's autoregressive
rollout (orange, dashed; learned point-encoder contacts + impulse-decoder wrench) as two
boxes in one 3D scene, with COM-height and position-error insets.

    python scripts/make_overlay_video.py --episode pilot1b_00561
"""
import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import FFMpegWriter
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from recom.data.dataset import EpisodeCache                    # noqa: E402
from recom.eval.visualize import _box_edges                    # noqa: E402
from recom.geometry.box_plane_analytic import CORNER_SIGNS     # noqa: E402
from recom.models.transition import BoxTransitionModel         # noqa: E402
from recom.train.rollout import ContactSource, rollout_batch   # noqa: E402
from recom.train.train_impulse import load_decoder             # noqa: E402
from recom.train.train_transition import load_encoder          # noqa: E402

# validated dataviz palette (light mode): slot 1 = Chrono, slot 2 = neural
SURFACE, INK, INK2 = "#fcfcfb", "#0b0b0b", "#52514e"
C_GT, C_NN = "#2a78d6", "#eb6834"


def load_model(run: Path, device):
    ck = torch.load(run / "final.pt", map_location=device, weights_only=False)
    cfg = ck["config"]
    norm = {k: np.asarray(v) for k, v in cfg["normalization"].items()}
    model = BoxTransitionModel(norm, contact_mode=cfg["contact_mode"], block_size=cfg["block_size"], n_layer=cfg["n_layer"],
                               n_embd=cfg["n_embd"], latent_dim=cfg["latent_dim"], dt=cfg["dt"], gravity_prior=cfg["gravity_prior"],
                               gravity=cfg["gravity"], contact_gate=cfg["contact_gate"], soft_gate=cfg["soft_gate"],
                               gyro_prior=cfg["gyro_prior"], physics_residual=cfg["physics_residual"]).to(device)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    return model, cfg


def _box_faces(corners: np.ndarray) -> list[np.ndarray]:
    """Six quads from the 8 world-space corners (CORNER_SIGNS index order)."""
    ring = [(-1, -1), (1, -1), (1, 1), (-1, 1)]
    faces = []
    for a in range(3):
        u, v = [i for i in range(3) if i != a]
        for s in (-1.0, 1.0):
            idx = [i for i in range(8) if CORNER_SIGNS[i, a] == s]
            idx.sort(key=lambda i: ring.index((int(CORNER_SIGNS[i, u]), int(CORNER_SIGNS[i, v]))))
            faces.append(corners[idx])
    return faces


def draw_box(ax, pos, quat, he, color, alpha, ls):
    c, edges = _box_edges(pos, quat, he)
    ax.add_collection3d(Poly3DCollection(_box_faces(c), facecolor=color, alpha=alpha, edgecolor="none", zsort="max", zorder=6))
    ax.add_collection3d(Line3DCollection([(c[i], c[j]) for i, j in edges], colors=color, lw=1.4, linestyles=ls, zorder=7))


def style_inset(ax, title):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#d5d4d0")
    ax.tick_params(colors=INK2, labelsize=8, width=0.8)
    ax.grid(axis="y", color="#ececea", lw=0.8)
    ax.set_axisbelow(True)
    ax.set_title(title, color=INK, fontsize=9.5, loc="left", pad=6)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episode", default="pilot1b_00561")
    p.add_argument("--data", default="data/pilot1b")
    p.add_argument("--run", default="runs/local_ed3_jl6r_8k_v6_h8")
    p.add_argument("--out", default=None)
    p.add_argument("--subsample", type=int, default=5, help="sim steps per video frame")
    p.add_argument("--fps", type=int, default=50)
    p.add_argument("--hold", type=float, default=1.5, help="seconds to hold the final frame")
    p.add_argument("--dpi", type=int, default=150)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    run = Path(args.run)
    model, cfg = load_model(run, device)
    encoder, _ = load_encoder(cfg["encoder_ckpt"], device)
    decoder, _ = load_decoder(cfg["decoder_ckpt"], device)
    source = ContactSource("learned", encoder, latent_dim=cfg["latent_dim"], decoder=decoder)

    ep = EpisodeCache(args.data, [args.episode]).episodes[0]
    n, T, dt = ep.n_steps, model.block_size, cfg["dt"]
    gt = ep.state[: n + 1].astype(np.float64)
    pred = rollout_batch(model, source, [ep], device)[0, : n + 1].astype(np.float64)
    he = ep.half_extents.astype(np.float64)
    t_axis = np.arange(n + 1) * dt
    pos_err = np.linalg.norm(pred[:, :3] - gt[:, :3], axis=1)
    print(f"{ep.episode_id}: n={n} he={he} final pos err {pos_err[-1]*100:.1f} cm")

    # scene bounds (true aspect)
    allp = np.concatenate([gt[:, :3], pred[:, :3]])
    m = 1.35 * float(np.linalg.norm(he))
    cx, cy = (allp[:, 0].min() + allp[:, 0].max()) / 2, (allp[:, 1].min() + allp[:, 1].max()) / 2
    hs = max(np.ptp(allp[:, 0]), np.ptp(allp[:, 1])) / 2 + m
    z1 = allp[:, 2].max() + m
    fig = plt.figure(figsize=(12.8, 7.2), dpi=args.dpi)
    fig.patch.set_facecolor(SURFACE)
    ax = fig.add_axes([-0.04, -0.06, 0.72, 1.06], projection="3d", computed_zorder=False)

    axz = fig.add_axes([0.72, 0.52, 0.245, 0.30])
    style_inset(axz, "COM height (m)")
    axz.axvspan(0, T * dt, color="#ececea")
    axz.plot(t_axis, gt[:, 2], color=C_GT, lw=2, solid_capstyle="round")
    axz.plot(t_axis, pred[:, 2], color=C_NN, lw=2, ls=(0, (4, 2)))
    axz.set_xlim(0, t_axis[-1])
    axz.set_ylim(0, None)
    curz = axz.axvline(0, color=INK2, lw=1)
    fi = ep.events["first_impact_step"]
    axz.axvline(fi * dt, color="#b8b7b2", lw=0.9, ls=(0, (2, 2)))
    axz.text(fi * dt + 0.012, 0.05, "first impact", transform=axz.get_xaxis_transform(), color=INK2, fontsize=7.5)

    axe = fig.add_axes([0.72, 0.10, 0.245, 0.30])
    style_inset(axe, "COM position error (cm)")
    axe.axvspan(0, T * dt, color="#ececea")
    axe.axvline(fi * dt, color="#b8b7b2", lw=0.9, ls=(0, (2, 2)))
    axe.plot(t_axis, pos_err * 100, color=INK2, lw=2)
    axe.set_xlim(0, t_axis[-1])
    axe.set_xlabel("time (s)", color=INK2, fontsize=8.5)
    cure = axe.axvline(0, color=INK2, lw=1)

    fig.text(0.035, 0.955, "Box drop — Chrono (NSC) vs neural simulator", color=INK, fontsize=15, weight="bold")
    fig.text(0.035, 0.918, f"{ep.episode_id} · JL-6-R v6 wrench bottleneck + physics residual · "
             f"autoregressive after a {T}-step ({T*dt*1e3:.0f} ms) ground-truth priming context · "
             f"{args.subsample*dt*args.fps:.2f}× real time", color=INK2, fontsize=9.5)
    handles = [Line2D([], [], color=C_GT, lw=2.5, label="Chrono (ground truth)"),
               Line2D([], [], color=C_NN, lw=2.5, ls=(0, (4, 2)), label="neural sim (rollout)")]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.033, 0.895), frameon=False,
               fontsize=10, labelcolor=INK, handlelength=2.4)

    # ground plane + grid (0.2 m)
    g = 0.2
    gx = np.arange(np.floor((cx - hs) / g) * g, cx + hs + g, g)
    gy = np.arange(np.floor((cy - hs) / g) * g, cy + hs + g, g)
    plane = [np.array([[gx[0], gy[0], 0], [gx[-1], gy[0], 0], [gx[-1], gy[-1], 0], [gx[0], gy[-1], 0]])]
    grid = [[(x, gy[0], 0), (x, gy[-1], 0)] for x in gx] + [[(gx[0], y, 0), (gx[-1], y, 0)] for y in gy]

    steps = list(range(0, n + 1, args.subsample)) + [n] * int(args.hold * args.fps)
    F = len(steps)
    sym = None
    try:
        from recom.eval.metrics import symmetry_rot_err_deg_np
        sym = symmetry_rot_err_deg_np(pred[n, 3:7], gt[n, 3:7], "d2")
    except Exception:
        pass
    final_txt = f"final: COM error {pos_err[-1]*100:.1f} cm" + (f" · orientation {sym:.1f}° (mod box symmetry)" if sym is not None else "")

    out = Path(args.out or f"results/videos/overlay_{ep.episode_id}.mp4")
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = FFMpegWriter(fps=args.fps, codec="h264", extra_args=["-pix_fmt", "yuv420p", "-crf", "18"])
    with writer.saving(fig, str(out), args.dpi):
        for k, s in enumerate(steps):
            ax.cla()
            ax.set_axis_off()
            ax.set_facecolor(SURFACE)
            ax.add_collection3d(Poly3DCollection(plane, facecolor="#f1f0ee", edgecolor="none", zorder=1))
            ax.add_collection3d(Line3DCollection(grid, colors="#dddcd8", lw=0.7, zorder=2))
            ax.plot(gt[: s + 1, 0], gt[: s + 1, 1], gt[: s + 1, 2], color=C_GT, lw=1.6, alpha=0.85, zorder=4)
            ax.plot(pred[: s + 1, 0], pred[: s + 1, 1], pred[: s + 1, 2], color=C_NN, lw=1.6, ls=(0, (4, 2)), alpha=0.9, zorder=5)
            draw_box(ax, gt[s, 0:3], gt[s, 3:7], he, C_GT, 0.50, "solid")
            draw_box(ax, pred[s, 0:3], pred[s, 3:7], he, C_NN, 0.26, (0, (4, 2)))
            ax.set_xlim(cx - hs, cx + hs)
            ax.set_ylim(cy - hs, cy + hs)
            ax.set_zlim(-0.02 * z1, z1)
            ax.set_box_aspect((1, 1, z1 * 1.02 / (2 * hs)))
            ax.view_init(elev=16, azim=-62 + 26 * k / max(F - 1, 1))
            phase = ("ground-truth priming", INK2) if s < T else ("neural rollout (autoregressive)", C_NN)
            ax.text2D(0.055, 0.800, f"t = {s*dt:.3f} s", transform=ax.transAxes, color=INK, fontsize=12, family="monospace")
            ax.text2D(0.055, 0.765, phase[0], transform=ax.transAxes, color=phase[1], fontsize=10)
            ax.text2D(0.055, 0.03, "ground grid 0.2 m", transform=ax.transAxes, color=INK2, fontsize=8)
            if s >= n and args.hold:
                ax.text2D(0.055, 0.725, final_txt, transform=ax.transAxes, color=INK, fontsize=10)
            curz.set_xdata([s * dt, s * dt])
            cure.set_xdata([s * dt, s * dt])
            writer.grab_frame(facecolor=SURFACE)
            if k % 50 == 0:
                print(f"frame {k}/{F}", flush=True)
    print(f"wrote {out} ({F} frames, {F/args.fps:.1f} s)")


if __name__ == "__main__":
    main()
