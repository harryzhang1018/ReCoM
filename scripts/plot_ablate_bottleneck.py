#!/usr/bin/env python
"""Figure for the closed-loop bottleneck ablation (scripts/ablate_bottleneck.py):
median error-vs-time per cell + where FULL's position drift accrues by GT regime.

    python scripts/plot_ablate_bottleneck.py --dir results/ablate_bottleneck/val
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#ececea"
# validated categorical palette, fixed slot order
SERIES = [("FULL", "#2a78d6"), ("FULL-HYB", "#eb6834"), ("E-ANA", "#1baf7a"), ("D-ORACLE", "#eda100"), ("N-OFF", "#e87ba4")]
REGIMES = [("far_free", "free flight"), ("near_contact", "near contact"), ("first_impact", "first impact"),
           ("contact", "sustained contact"), ("rebound_repeat", "rebound"), ("resting", "resting (GT settled)")]


def style(ax, title):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#d5d4d0")
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.set_title(title, color=INK, fontsize=11, loc="left", pad=8)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", default="results/ablate_bottleneck/val")
    p.add_argument("--dt", type=float, default=1e-3)
    args = p.parse_args()
    d = Path(args.dir)
    summary = json.loads((d / "summary.json").read_text())
    have = [(c, col) for c, col in SERIES if (d / f"curves_{c}.npz").exists()]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    fig.subplots_adjust(left=0.05, right=0.985, top=0.82, bottom=0.13, wspace=0.28)

    for ax, key, title, scale in ((axes[0], "pos_err", "median COM position error (m)", 1.0),
                                  (axes[1], "rot_err_deg", "median orientation error (°)", 1.0)):
        style(ax, title)
        ends = []
        for c, col in have:
            z = np.load(d / f"curves_{c}.npz")
            t = z["grid"] * args.dt
            y = z[key] * scale
            ax.plot(t, y, color=col, lw=2, solid_capstyle="round")
            ends.append([c, t[-1], y[-1]])
        # de-collided direct labels, kept inside the axes
        span = max(e[2] for e in ends) - min(min(e[2] for e in ends), 0)
        gap = 0.05 * span
        ends.sort(key=lambda e: e[2])
        for i in range(1, len(ends)):
            ends[i][2] = max(ends[i][2], ends[i - 1][2] + gap)
        for c, tx, ty in ends:
            ax.annotate(c, (tx, ty), xytext=(5, 0), textcoords="offset points", color=INK, fontsize=8, va="center")
        ax.set_xlabel("time (s)", color=INK2, fontsize=9)
        ax.set_xlim(0, max(e[1] for e in ends) * 1.22)

    ax = axes[2]
    style(ax, "FULL: median position drift accrued per regime (cm)")
    vals = [summary["FULL"].get(f"pos_accr_{k}", {}).get("median", 0.0) for k, _ in REGIMES]
    y = np.arange(len(REGIMES))
    ax.barh(y, vals, height=0.62, color="#2a78d6", edgecolor=SURFACE, lw=2)
    ax.set_yticks(y, [lbl for _, lbl in REGIMES], fontsize=9)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, lw=0.8)
    for yi, v in zip(y, vals):
        ax.annotate(f"{v:.2f}", (v, yi), xytext=(4, 0), textcoords="offset points", color=INK, fontsize=8.5, va="center")

    handles = [plt.Line2D([], [], color=col, lw=2.5, label=c) for c, col in have]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.045, 0.985), ncol=len(have),
               frameon=False, fontsize=9.5, labelcolor=INK, handlelength=1.8, columnspacing=1.4)
    fig.text(0.045, 0.905, "Closed-loop stage ablation — frozen JL-6-R v6+H8, 255 val episodes, learned/analytic/oracle substitutions",
             color=INK2, fontsize=9.5)
    out = d / "bottleneck.png"
    fig.savefig(out, facecolor=SURFACE)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
