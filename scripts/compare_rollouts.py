#!/usr/bin/env python
"""Paired per-episode comparison of closed-loop rollouts between runs (encoder-decoder study, plan Sec. 9.6 / 11).

    python scripts/compare_rollouts.py --base runs/ed3_base64_s0 --runs runs/ed3_jl6_s0 runs/ed3_jl6r_s0 --source learned

For every split (val/test/test_geometry) and metric, reports the median of each run, the paired per-episode difference
(run - base), the median relative change, a bootstrap 95 % CI of the median difference and the fraction of episodes
improved.  Seed groups (runs/<name>_s0,_s1,...) can be passed as a glob; episodes are paired by episode_id.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

METRICS = ["pos_err@500", "pos_err@1000", "pos_err_final", "rot_err_deg@500", "rot_err_deg@1000", "rot_err_deg_final", "sym_rot_err_deg_final", "settled_face_match", "impact_dv_err", "impact_dw_err",
           "v_err_mean", "w_err_mean", "max_penetration_pred", "artificial_energy_max", "final_height_err", "final_ang_speed_err"]
HIGHER_IS_BETTER = {"settled_face_match"}


def load_rows(run: str, split: str, source: str) -> dict[str, dict]:
    """Rows of `rollout_rows_<split>_<source>.json` (or the gzipped archive copy `.json.gz`) keyed by episode id."""
    import gzip
    p = Path(run) / f"rollout_rows_{split}_{source}.json"
    if p.exists():
        rows = json.load(open(p))
    elif p.with_suffix(".json.gz").exists():
        rows = json.load(gzip.open(p.with_suffix(".json.gz"), "rt"))
    else:
        return {}
    return {r["episode_id"]: r for r in rows}


def bootstrap_median_ci(d: np.ndarray, n: int = 2000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), (n, len(d)))
    med = np.median(d[idx], axis=1)
    return float(np.percentile(med, 2.5)), float(np.percentile(med, 97.5))


def compare(base_rows: dict, run_rows: dict, metric: str) -> dict | None:
    ids = [e for e in base_rows if e in run_rows and metric in base_rows[e] and metric in run_rows[e]]
    if len(ids) < 3:
        return None
    b = np.array([base_rows[e][metric] for e in ids], dtype=np.float64)
    r = np.array([run_rows[e][metric] for e in ids], dtype=np.float64)
    fin = np.isfinite(b) & np.isfinite(r)
    n_div = int((~fin).sum())
    b, r = b[fin], r[fin]
    if len(b) < 3:
        return None
    d = r - b
    lo, hi = bootstrap_median_ci(d)
    better = (d > 0) if metric in HIGHER_IS_BETTER else (d < 0)
    rel = (np.median(r) - np.median(b)) / abs(np.median(b)) if abs(np.median(b)) > 1e-9 else float("nan")
    return {"n": int(len(b)), "base_median": float(np.median(b)), "run_median": float(np.median(r)), "rel_change_of_median": float(rel),
            "median_diff": float(np.median(d)), "ci95_median_diff": [lo, hi], "frac_improved": float(better.mean()),
            "significant": bool(hi < 0 or lo > 0), "n_diverged_pairs": n_div}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="baseline run dir (or glob of seed dirs; rows are pooled by episode with seed suffix)")
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--source", default="learned")
    ap.add_argument("--splits", default="val,test,test_geometry")
    ap.add_argument("--metrics", default=",".join(METRICS))
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    metrics = a.metrics.split(",")

    def rows_for(pattern: str, split: str) -> dict:
        out = {}
        dirs = sorted(glob.glob(pattern)) or [pattern]
        for d in dirs:
            tag = Path(d).name
            for eid, r in load_rows(d, split, a.source).items():
                out[f"{eid}|{tag[-3:]}" if len(dirs) > 1 else eid] = r
        return out

    report = {}
    for split in a.splits.split(","):
        base_rows = rows_for(a.base, split)
        if not base_rows:
            continue
        report[split] = {}
        print(f"\n### {split} ({a.source} contacts): run vs {a.base}\n")
        print("| run | metric | n | base median | run median | rel. change | median paired diff [95% CI] | frac improved | sig |")
        print("|---|---|---|---|---|---|---|---|---|")
        for run in a.runs:
            rr = rows_for(run, split)
            for m in metrics:
                c = compare(base_rows, rr, m)
                if c is None:
                    continue
                report[split].setdefault(run, {})[m] = c
                lo, hi = c["ci95_median_diff"]
                print(f"| {Path(run).name} | {m} | {c['n']} | {c['base_median']:.4g} | {c['run_median']:.4g} | {('n/a' if np.isnan(c['rel_change_of_median']) else f'{100 * c['rel_change_of_median']:+.1f}%')} | {c['median_diff']:+.4g} [{lo:+.3g}, {hi:+.3g}] | {c['frac_improved']:.2f} | {'*' if c['significant'] else ''} |")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(report, open(a.out, "w"), indent=2)


if __name__ == "__main__":
    main()
