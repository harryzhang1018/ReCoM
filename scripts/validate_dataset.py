#!/usr/bin/env python
"""Run dataset validation on a generated dataset: schema/convention checks on every episode,
free-flight + first-contact timing checks, and record/replay on a subset.

    python scripts/validate_dataset.py data/smoke [--replay N]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from recom.config import dump_json  # noqa: E402
from recom.data.splits import check_no_leakage, load_splits  # noqa: E402
from recom.data.storage import episode_ids, load_episode  # noqa: E402
from recom.data.validate import first_contact_timing_check, free_flight_check, replay_check, validate_episode  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--replay", type=int, default=5)
    a = ap.parse_args()
    ids = episode_ids(a.root)
    report = {"n_episodes": len(ids), "schema_failures": {}, "free_flight": [], "timing": [], "replay": []}
    for eid in ids:
        rec = load_episode(a.root, eid)
        fails = validate_episode(rec)
        if fails:
            report["schema_failures"][eid] = fails
        report["free_flight"].append(free_flight_check(rec))
        report["timing"].append(first_contact_timing_check(rec))
    for eid in ids[: a.replay]:
        rec = load_episode(a.root, eid)
        report["replay"].append({"episode_id": eid, **replay_check(rec)})
    splits = load_splits(Path(a.root) / "splits.json")
    check_no_leakage(splits)
    ff = report["free_flight"]
    tm = report["timing"]
    summary = {
        "schema_pass": len(report["schema_failures"]) == 0,
        "free_flight_pass": all(x["pass"] for x in ff),
        "free_flight_max_pos_err": max(x["pos_err"] for x in ff),
        "free_flight_max_vel_err": max(x["vel_err"] for x in ff),
        "timing_pass": all(x["pass"] for x in tm),
        "timing_err_steps_median": float(np.median([abs(x["err_steps"]) for x in tm])),
        "timing_err_steps_max": float(np.max([abs(x["err_steps"]) for x in tm])),
        "replay_pass": all(x["pass"] for x in report["replay"]),
        "replay_max_state_diff": max(max(x[k] for k in x if k.startswith("max_abs_diff")) for x in report["replay"]) if report["replay"] else None,
        "no_split_leakage": True,
    }
    report["summary"] = summary
    dump_json(report, Path(a.root) / "validation_report.json")
    print("schema failures:", report["schema_failures"] if report["schema_failures"] else "none")
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
