"""Episode-level splits (never by frame): in-distribution, geometry holdout, height holdout, orientation holdout."""
from __future__ import annotations

import numpy as np

from ..config import dump_json, load_json


def make_splits(index: dict, seed: int = 0, frac=(0.70, 0.15, 0.15), geometry_holdout_frac: float = 0.15,
                height_holdout: tuple[float, float] | None = None) -> dict[str, list[str]]:
    """Return {split_name: [episode_id, ...]}.

    - train/val/test: random split by episode among the non-held-out geometry groups.
    - test_geometry: all episodes of held-out geometry groups (whole groups; never split).
    - test_height: episodes with release clearance outside `height_holdout` (low/high bands), if given.
    - test_orientation_{corner,edge,face}: convenience views of the test split by approach mode.
    """
    eps = index["episodes"]
    rng = np.random.default_rng(seed)
    groups = sorted(set(e["geometry_group"] for e in eps))
    n_hold = int(round(geometry_holdout_frac * len(groups))) if len(groups) > 1 else 0
    held = set(rng.permutation(groups)[:n_hold].tolist())
    splits: dict[str, list[str]] = {"train": [], "val": [], "test": [], "test_geometry": [], "test_height": []}
    in_dist = [e for e in eps if e["geometry_group"] not in held]
    for e in eps:
        if e["geometry_group"] in held:
            splits["test_geometry"].append(e["episode_id"])
    if height_holdout is not None:
        lo, hi = height_holdout
        keep = []
        for e in in_dist:
            if e["clearance"] < lo or e["clearance"] > hi:
                splits["test_height"].append(e["episode_id"])
            else:
                keep.append(e)
        in_dist = keep
    ids = [e["episode_id"] for e in in_dist]
    perm = rng.permutation(len(ids))
    n_tr = int(round(frac[0] * len(ids)))
    n_va = int(round(frac[1] * len(ids)))
    splits["train"] = [ids[i] for i in sorted(perm[:n_tr])]
    splits["val"] = [ids[i] for i in sorted(perm[n_tr:n_tr + n_va])]
    splits["test"] = [ids[i] for i in sorted(perm[n_tr + n_va:])]
    by_id = {e["episode_id"]: e for e in eps}
    for mode, name in ((1, "corner"), (2, "edge"), (3, "face")):
        splits[f"test_orientation_{name}"] = [i for i in splits["test"] if by_id[i]["approach_mode"] == mode]
    return splits


def check_no_leakage(splits: dict[str, list[str]]) -> None:
    core = ("train", "val", "test", "test_geometry", "test_height")
    seen: dict[str, str] = {}
    for name in core:
        for i in splits.get(name, []):
            assert i not in seen, f"episode {i} in both {seen[i]} and {name}"
            seen[i] = name


def save_splits(splits: dict, path) -> None:
    check_no_leakage(splits)
    dump_json(splits, path)


def load_splits(path) -> dict[str, list[str]]:
    return load_json(path)
