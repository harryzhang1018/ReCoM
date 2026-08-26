#!/usr/bin/env python
"""Collect final_metrics.json of all runs under a directory into markdown tables."""
from __future__ import annotations

import json
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
    contact_rows, roll_rows, one_rows = [], [], []
    ckeys = ["frame_recall", "frame_precision", "slot_recall", "d_mae", "point_err_median_pct_min_dim", "normal_deg_median", "cardinality_acc", "ece", "first_impact_err_steps_median", "first_impact_err_steps_p99", "first_impact_missed"]
    rkeys = ["pos_err@100", "pos_err@500", "pos_err_final", "rot_err_deg_final", "v_err_mean", "w_err_mean", "impact_dv_err", "impact_dw_err", "apex_err", "apex_time_err_steps", "max_penetration_pred", "artificial_energy_max"]
    okeys = ["dv_err_mean", "dv_err_far_free", "dv_err_near_contact", "dv_err_first_impact", "dw_err_first_impact", "dv_err_rebound_repeat", "dv_err_resting"]
    for r in runs:
        m = json.load(open(r / "final_metrics.json"))
        if "one_step" in m:
            for split, v in m["one_step"].items():
                one_rows.append([r.name, split] + [fmt(v.get(k)) for k in okeys])
            for key, v in m["rollout"].items():
                roll_rows.append([r.name, key] + [fmt(v[k]["median"]) if k in v else "-" for k in rkeys])
        else:
            for split, v in m.items():
                contact_rows.append([r.name, split] + [fmt(v.get(k)) for k in ckeys])
    def table(title, header, rows):
        if not rows:
            return
        print(f"\n### {title}\n")
        print("| " + " | ".join(header) + " |")
        print("|" + "---|" * len(header))
        for row in rows:
            print("| " + " | ".join(row) + " |")
    table("Contact encoders (Experiment C)", ["run", "split"] + ckeys, contact_rows)
    table("Transition one-step errors (m/s, rad/s)", ["run", "split"] + okeys, one_rows)
    table("Closed-loop rollouts (median over episodes)", ["run", "split/source"] + rkeys, roll_rows)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "runs")
