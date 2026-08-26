"""On-disk layout:  <root>/episodes/<episode_id>.npz  +  <episode_id>.json  +  <root>/index.json"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np

from ..config import dump_json, load_json
from .schema import EpisodeRecord


def episode_paths(root: str | Path, episode_id: str) -> tuple[Path, Path]:
    d = Path(root) / "episodes"
    return d / f"{episode_id}.npz", d / f"{episode_id}.json"


def save_episode(rec: EpisodeRecord, root: str | Path) -> Path:
    npz, js = episode_paths(root, rec.meta["episode"]["episode_id"])
    npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(npz, **rec.arrays())
    dump_json({"meta": rec.meta, "events": rec.events}, js)
    return npz


def load_episode(root: str | Path, episode_id: str) -> EpisodeRecord:
    npz, js = episode_paths(root, episode_id)
    with np.load(npz) as z:
        arrs = {k: z[k] for k in z.files}
    mj = load_json(js)
    return EpisodeRecord.from_arrays(mj["meta"], arrs, mj["events"])


def write_index(root: str | Path, gen_config: dict, episodes: list[dict]) -> None:
    dump_json({"gen_config": gen_config, "episodes": episodes}, Path(root) / "index.json")


def read_index(root: str | Path) -> dict:
    return load_json(Path(root) / "index.json")


def episode_ids(root: str | Path) -> list[str]:
    return [e["episode_id"] for e in read_index(root)["episodes"]]


def iter_episodes(root: str | Path, ids: Iterable[str] | None = None):
    for eid in (ids if ids is not None else episode_ids(root)):
        yield load_episode(root, eid)


def index_summary(root: str | Path) -> dict:
    idx = read_index(root)
    eps = idx["episodes"]
    n_steps = np.array([e["n_steps"] for e in eps])
    return {
        "n_episodes": len(eps),
        "total_steps": int(n_steps.sum()),
        "mean_steps": float(n_steps.mean()) if len(eps) else 0,
        "termination": {r: sum(1 for e in eps if e["termination_reason"] == r) for r in set(e["termination_reason"] for e in eps)},
        "approach_modes": {m: sum(1 for e in eps if e["approach_mode"] == m) for m in (1, 2, 3)},
        "geometry_groups": len(set(e["geometry_group"] for e in eps)),
    }
