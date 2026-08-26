#!/usr/bin/env python
"""Generate a box-drop dataset from a YAML config (deterministic, multiprocessing).

    python scripts/generate_dataset.py configs/smoke.yaml [--n N] [--workers W] [--out DIR]
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from recom.config import DatasetGenConfig, dump_json  # noqa: E402
from recom.data.splits import make_splits, save_splits  # noqa: E402
from recom.data.storage import index_summary, write_index  # noqa: E402
from recom.sim.sampling import orientation_coverage, sample_episode  # noqa: E402


def _worker(args):
    cfg_dict, ep_index, out = args
    from recom.data.storage import save_episode
    from recom.sim.chrono_box_drop import run_episode

    cfg = DatasetGenConfig.from_dict(cfg_dict)
    ep = sample_episode(cfg, ep_index)
    rec = run_episode(ep)
    save_episode(rec, out)
    m = rec.meta
    return {
        "episode_id": ep.episode_id,
        "index": ep_index,
        "seed": ep.seed,
        "geometry_id": ep.box.geometry_id,
        "geometry_group": ep.box.geometry_group,
        "half_extents": list(ep.box.half_extents),
        "clearance": ep.clearance,
        "approach_mode": ep.approach_mode,
        "n_steps": m["n_steps"],
        "termination_reason": m["termination_reason"],
        "first_impact_step": rec.events["first_impact_step"],
        "n_impacts": rec.events["n_impacts"],
        "settle_step": rec.events["settle_step"],
        "max_penetration": rec.events["max_penetration"],
        "wall_time_s": m["wall_time_s"],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--out", type=str, default=None)
    a = ap.parse_args()
    cfg = DatasetGenConfig.from_yaml(a.config)
    if a.n is not None:
        cfg.n_episodes = a.n
    if a.workers is not None:
        cfg.n_workers = a.workers
    if a.out is not None:
        cfg.output_dir = a.out
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    dump_json(cfg.to_dict(), out / "gen_config.json")

    cfg_dict = cfg.to_dict()
    jobs = [(cfg_dict, i, str(out)) for i in range(cfg.n_episodes)]
    t0 = time.time()
    ctx = mp.get_context("spawn")  # one Chrono system per process; never re-use across episodes in the same process
    results = []
    with ctx.Pool(cfg.n_workers, maxtasksperchild=50) as pool:
        for i, r in enumerate(pool.imap(_worker, jobs, chunksize=1)):
            results.append(r)
            if (i + 1) % max(1, cfg.n_episodes // 20) == 0 or i + 1 == cfg.n_episodes:
                print(f"[{i + 1}/{cfg.n_episodes}] {time.time() - t0:.1f}s", flush=True)
    results.sort(key=lambda r: r["index"])
    write_index(out, cfg_dict, results)
    splits = make_splits({"episodes": results}, seed=cfg.seed)
    save_splits(splits, out / "splits.json")
    cov = orientation_coverage([sample_episode(cfg, i) for i in range(cfg.n_episodes)])
    summary = index_summary(out)
    summary["orientation_coverage"] = cov
    summary["split_sizes"] = {k: len(v) for k, v in splits.items()}
    summary["generation_wall_s"] = time.time() - t0
    dump_json(summary, out / "summary.json")
    print("summary:", summary)


if __name__ == "__main__":
    main()
