"""Shared training utilities."""
from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch

from ..config import dump_json, load_json
from ..data.dataset import EpisodeCache
from ..data.splits import load_splits


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_caches(root: str, splits: list[str], max_episodes: int | None = None) -> dict[str, EpisodeCache]:
    sp = load_splits(Path(root) / "splits.json")
    out = {}
    for name in splits:
        ids = sp.get(name, [])
        if max_episodes:
            ids = ids[:max_episodes]
        t0 = time.time()
        out[name] = EpisodeCache(root, ids) if ids else None
        print(f"loaded split {name}: {len(ids)} episodes in {time.time() - t0:.1f}s", flush=True)
    return out


def cosine_lr(step: int, total: int, base: float, warmup: int = 200, min_frac: float = 0.05) -> float:
    if step < warmup:
        return base * (step + 1) / warmup
    p = min(1.0, (step - warmup) / max(1, total - warmup))
    return base * (min_frac + (1 - min_frac) * 0.5 * (1 + math.cos(math.pi * p)))


class MetricsLogger:
    def __init__(self, out_dir: Path) -> None:
        self.path = Path(out_dir) / "metrics.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, rec: dict) -> None:
        with open(self.path, "a") as f:
            f.write(json.dumps(rec, default=float) + "\n")


def to_device(batch: dict, device) -> dict:
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}


def save_checkpoint(path: Path, model: torch.nn.Module, config: dict, extra: dict | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "config": config, **(extra or {})}, path)
