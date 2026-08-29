#!/usr/bin/env python
"""Collect final_metrics.json of all runs under a directory into markdown tables."""
from __future__ import annotations

import json
import re
import statistics
import sys
from pathlib import Path


def fmt(v):
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)


def main(root: str = "runs") -> None:
    runs = sorted(p for p in Path(root).iterdir() if (p / "final_metrics.json").exists())
    contact_rows, roll_rows, one_rows, imp_rows = [], [], [], []
    ckeys = ["frame_recall", "frame_precision", "slot_recall", "d_mae", "point_err_median_pct_min_dim", "normal_deg_median", "cardinality_acc", "ece", "first_impact_err_steps_median", "first_impact_err_steps_p99", "first_impact_missed"]
    rkeys = ["pos_err@100", "pos_err@500", "pos_err_final", "rot_err_deg_final", "sym_rot_err_deg_final", "settled_face_match", "v_err_mean", "w_err_mean", "impact_dv_err", "impact_dw_err", "apex_err", "apex_time_err_steps", "max_penetration_pred", "artificial_energy_max"]
    okeys = ["dv_err_mean", "dv_err_far_free", "dv_err_near_contact", "dv_err_first_impact", "dw_err_first_impact", "dv_err_rebound_repeat", "dw_err_rebound_repeat", "dv_err_resting", "dw_err_resting"]
    ikeys = ["dv_mae", "zero_baseline_dv_mae", "dL_mae", "zero_baseline_dL_mae", "dw_mae", "dv_rel_err_median", "dir_err_deg_median", "missed_impulse_rate", "spurious_impulse_rate", "cone_violation_rate"]
    groups: dict[tuple, list] = {}
    for r in runs:
        m = json.load(open(r / "final_metrics.json"))
        if "one_step" in m:
            for split, v in m["one_step"].items():
                one_rows.append([r.name, split] + [fmt(v.get(k)) for k in okeys])
            for key, v in m["rollout"].items():
                roll_rows.append([r.name, key] + [fmt(v[k]["median"]) if k in v else "-" for k in rkeys])
                g = re.sub(r"_s\d+$", "", r.name)
                if g != r.name:
                    groups.setdefault((g, key), []).append([v[k]["median"] if k in v else None for k in rkeys])
        if "impulse" in m:
            for split, v in m["impulse"].items():
                for cat in ("first_impact", "rebound_repeat", "resting"):
                    c = v.get("by_category", {}).get(cat)
                    if c:
                        imp_rows.append([r.name, f"{split}/{cat}"] + [fmt(c.get(k)) for k in ikeys])
        if "one_step" not in m and "impulse" not in m:
            for split, v in m.items():
                contact_rows.append([r.name, split] + [fmt(v.get(k)) for k in ckeys])
    seed_rows = []
    for (g, key), vals in sorted(groups.items()):
        if len(vals) > 1:
            cols = []
            for j in range(len(rkeys)):
                x = [v[j] for v in vals if v[j] is not None]
                cols.append(f"{statistics.mean(x):.4g} ± {statistics.pstdev(x):.2g}" if x else "-")
            seed_rows.append([f"{g} (n={len(vals)})", key] + cols)

    def table(title, header, rows):
        if not rows:
            return
        print(f"\n### {title}\n")
        print("| " + " | ".join(header) + " |")
        print("|" + "---|" * len(header))
        for row in rows:
            print("| " + " | ".join(row) + " |")
    table("Contact encoders (Experiment C)", ["run", "split"] + ckeys, contact_rows)
    table("Impulse decoders (ED2; per category, mass-normalized units m/s, m^2/s, rad/s)", ["run", "split/category"] + ikeys, imp_rows)
    table("Transition one-step errors (m/s, rad/s)", ["run", "split"] + okeys, one_rows)
    table("Closed-loop rollouts (median over episodes)", ["run", "split/source"] + rkeys, roll_rows)
    table("Closed-loop rollouts, seed groups (mean ± std of per-seed medians)", ["group", "split/source"] + rkeys, seed_rows)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "runs")
