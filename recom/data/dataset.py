"""Torch datasets for Study 1.

1. TransitionWindowDataset  -- chronological windows (s_t, c_t) -> s_{t+1} for NeDM-style training.
2. ContactQueryDataset      -- balanced frame-level view (pose, geometry) -> canonical contact set.

Both are built from an in-memory `EpisodeCache` (float32) so that random access is cheap.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

from ..config import K_SLOTS
from .impulse_targets import PHYS_KEYS, gyro_dw_world, impulse_targets_from_record, phys_arrays
from .schema import EpisodeRecord
from .storage import load_episode

# ---- transition state layout (frozen) -------------------------------------------------
# input state token s_t = [pos(3), quat(4), lin_vel(3), ang_vel_world(3)] (13) ; geometry = half_extents(3)
STATE_DIM = 13
TARGET_DIM = 6          # [d lin_vel (3), d ang_vel_world (3)]

# ---- contact-query categories (Section 10.2) ------------------------------------------
CAT_FAR, CAT_NEAR, CAT_FIRST_IMPACT, CAT_CONTACT, CAT_REBOUND, CAT_REST = range(6)
CAT_NAMES = ["far_free", "near_contact", "first_impact", "contact", "rebound_repeat", "resting"]
FAR_GAP = 0.05
FIRST_IMPACT_WINDOW = 5


def frame_categories(rec: EpisodeRecord) -> np.ndarray:
    """Strata for the balanced contact-query view (Section 10.2).

    far_free      : no contact, analytic gap > FAR_GAP
    near_contact  : no contact, analytic gap <= FAR_GAP
    first_impact  : first FIRST_IMPACT_WINDOW frames of the first contact run
    contact       : other in-contact frames (not resting)
    rebound_repeat: windows around later impacts (approach + first frames of later contact runs)
    resting       : settled frames in contact
    """
    n = rec.n_steps
    C = rec.canon
    cat = np.full(n, CAT_FAR, dtype=np.int64)
    gap = C["analytic_min_gap"]
    nc = C["n_contacts"]
    settled = rec.states["settled"][:n]
    cat[(nc == 0) & (gap <= FAR_GAP)] = CAT_NEAR
    cat[nc > 0] = CAT_CONTACT
    runs = rec.events["contact_runs"]
    W = FIRST_IMPACT_WINDOW
    for a, b in runs[1:]:
        cat[max(0, a - W): min(n, a + W)] = CAT_REBOUND
    if runs:
        a, b = runs[0]
        cat[a: min(n, a + W)] = CAT_FIRST_IMPACT
    cat[settled & (nc > 0)] = CAT_REST
    return cat


@dataclass
class EpisodeArrays:
    episode_id: str
    half_extents: np.ndarray            # (3,)
    state: np.ndarray                   # (N+1, 13)
    contact: dict[str, np.ndarray]      # per-step canonical contact arrays (N, ...)
    category: np.ndarray                # (N,)
    events: dict
    meta: dict
    impulse: dict[str, np.ndarray] = field(default_factory=dict)   # mass-normalized wrench labels (N, ...), see impulse_targets
    phys: dict[str, np.ndarray] = field(default_factory=dict)      # PHYS_KEYS as float32 arrays (0-d or (3,))

    @property
    def n_steps(self) -> int:
        return self.state.shape[0] - 1


def episode_to_arrays(rec: EpisodeRecord) -> EpisodeArrays:
    S = rec.states
    state = np.concatenate([S["pos"], S["quat"], S["lin_vel"], S["ang_vel_world"]], axis=1).astype(np.float32)
    C = rec.canon
    contact = {
        "active": C["c_active"].astype(np.float32),
        "d": C["c_d"].astype(np.float32),
        "n": C["c_n"].astype(np.float32),
        "p_box_local": C["c_p_box_local"].astype(np.float32),
        "p_box_world": C["c_p_box_world"].astype(np.float32),
        "p_ground_world": C["c_p_ground_world"].astype(np.float32),
        "force_world": C["c_force_world"].astype(np.float32),
        "n_contacts": C["n_contacts"].astype(np.int64),
        "contact_mode": C["contact_mode"].astype(np.int64),
        "analytic_min_gap": C["analytic_min_gap"].astype(np.float32),
        "near_contact": C["near_contact"].astype(np.float32),
    }
    return EpisodeArrays(
        episode_id=rec.meta["episode"]["episode_id"],
        half_extents=np.asarray(rec.meta["half_extents"], dtype=np.float32),
        state=state,
        contact=contact,
        category=frame_categories(rec),
        events=rec.events,
        meta=rec.meta,
        impulse=impulse_targets_from_record(rec),
        phys=phys_arrays(rec.meta),
    )


class EpisodeCache:
    def __init__(self, root: str, ids: list[str]) -> None:
        self.root = root
        self.episodes = [episode_to_arrays(load_episode(root, eid)) for eid in ids]

    def __len__(self) -> int:
        return len(self.episodes)


class TransitionWindowDataset(Dataset):
    """Windows of length T (never crossing episodes): returns states (T,13), targets (T,6),
    contacts (dict of (T,K,...)), half_extents (3), and category (T,)."""

    def __init__(self, cache: EpisodeCache, T: int, stride: int = 1) -> None:
        self.cache, self.T = cache, T
        self.index: list[tuple[int, int]] = []
        for e_i, ep in enumerate(cache.episodes):
            n_valid = ep.n_steps - T + 1
            for s in range(0, max(n_valid, 0), stride):
                self.index.append((e_i, s))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        e_i, s = self.index[i]
        ep = self.cache.episodes[e_i]
        T = self.T
        st = ep.state[s: s + T + 1]
        item = {
            "states": torch.from_numpy(st[:T]),
            "next_states": torch.from_numpy(st[1:]),
            "targets": torch.from_numpy(st[1:, 7:13] - st[:T, 7:13]),
            "half_extents": torch.from_numpy(ep.half_extents),
            "category": torch.from_numpy(ep.category[s: s + T]),
        }
        for k, v in ep.contact.items():
            item["c_" + k] = torch.from_numpy(np.ascontiguousarray(v[s: s + T]))
        for k, v in ep.impulse.items():
            item[k] = torch.from_numpy(np.ascontiguousarray(v[s: s + T]))
        for k in PHYS_KEYS:
            item[k] = torch.from_numpy(ep.phys[k])
        return item


class ImpulseFrameDataset(TransitionWindowDataset):
    """Frame-level view (T = 1 windows) for impulse-decoder pretraining; same keys as TransitionWindowDataset with a
    leading time axis of length 1, so the encoder/decoder batch code is shared.  `categories` restricts the frames."""

    def __init__(self, cache: EpisodeCache, categories: list[int] | None = None) -> None:
        super().__init__(cache, T=1, stride=1)
        if categories is not None:
            self.index = [(e, k) for e, k in self.index if cache.episodes[e].category[k] in categories]
        self.categories = np.array([cache.episodes[e].category[k] for e, k in self.index], dtype=np.int64)

    def balanced_sampler(self, num_samples: int, seed: int = 0, weights: dict[int, float] | None = None) -> WeightedRandomSampler:
        """Inverse-frequency sampling over categories, optionally scaled by `weights` {category: factor}."""
        counts = np.bincount(self.categories, minlength=len(CAT_NAMES)).astype(np.float64)
        w = np.where(counts > 0, 1.0 / np.maximum(counts, 1), 0.0)
        if weights:
            for c, f in weights.items():
                w[c] *= f
        g = torch.Generator().manual_seed(seed)
        return WeightedRandomSampler(torch.from_numpy(w[self.categories]), num_samples=num_samples, replacement=True, generator=g)

    def category_counts(self) -> dict[str, int]:
        c = np.bincount(self.categories, minlength=len(CAT_NAMES))
        return {CAT_NAMES[i]: int(c[i]) for i in range(len(CAT_NAMES))}


class ContactQueryDataset(Dataset):
    """Frame-level (pose, geometry) -> canonical contact set. Use `balanced_sampler` for training."""

    def __init__(self, cache: EpisodeCache, categories: list[int] | None = None) -> None:
        self.cache = cache
        self.index: list[tuple[int, int]] = []
        for e_i, ep in enumerate(cache.episodes):
            for k in range(ep.n_steps):
                if categories is None or ep.category[k] in categories:
                    self.index.append((e_i, k))
        self.categories = np.array([cache.episodes[e].category[k] for e, k in self.index])

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        e_i, k = self.index[i]
        ep = self.cache.episodes[e_i]
        st = ep.state[k]
        item = {
            "half_extents": torch.from_numpy(ep.half_extents),
            "pos": torch.from_numpy(st[0:3]),
            "quat": torch.from_numpy(st[3:7]),
            "category": torch.tensor(int(ep.category[k])),
            "episode_index": torch.tensor(e_i),
            "step": torch.tensor(k),
        }
        for key, v in ep.contact.items():
            item["c_" + key] = torch.from_numpy(np.asarray(v[k]).copy())
        return item

    def balanced_sampler(self, num_samples: int, seed: int = 0) -> WeightedRandomSampler:
        counts = np.bincount(self.categories, minlength=len(CAT_NAMES)).astype(np.float64)
        w = np.where(counts > 0, 1.0 / np.maximum(counts, 1), 0.0)[self.categories]
        g = torch.Generator().manual_seed(seed)
        return WeightedRandomSampler(torch.from_numpy(w), num_samples=num_samples, replacement=True, generator=g)

    def category_counts(self) -> dict[str, int]:
        c = np.bincount(self.categories, minlength=len(CAT_NAMES))
        return {CAT_NAMES[i]: int(c[i]) for i in range(len(CAT_NAMES))}


def collate_dict(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}


def gyro_prior_episode(ep: EpisodeArrays) -> np.ndarray:
    """Exact torque-free delta omega_world (N,3) for every step of an episode (from the pre-step state)."""
    st = ep.state[:-1].astype(np.float64)
    return gyro_dw_world(st[:, 3:7], st[:, 10:13], ep.phys["inertia_diag_over_m"].astype(np.float64), float(ep.meta["dt"]))


def compute_state_normalization(cache: EpisodeCache, prior: np.ndarray | None = None, gyro: bool = False) -> dict[str, np.ndarray]:
    """Per-channel z-score statistics over the train split; `prior` (6,) is subtracted from the targets and, with
    `gyro`, so is the exact state-dependent gyroscopic delta omega (transition model `gyro_prior`)."""
    st = np.concatenate([ep.state[:-1] for ep in cache.episodes], axis=0).astype(np.float64)
    tg = np.concatenate([ep.state[1:, 7:13] - ep.state[:-1, 7:13] for ep in cache.episodes], axis=0).astype(np.float64)
    if prior is not None:
        tg = tg - np.asarray(prior, dtype=np.float64)
    if gyro:
        tg[:, 3:6] -= np.concatenate([gyro_prior_episode(ep) for ep in cache.episodes], axis=0)
    return {
        "state_mean": st.mean(0), "state_std": np.maximum(st.std(0), 1e-6),
        "target_mean": tg.mean(0), "target_std": np.maximum(tg.std(0), 1e-6),
    }


def compute_wrench_normalization(cache: EpisodeCache) -> dict[str, np.ndarray]:
    """Scale (std, zero mean) of the mass-normalized contact wrench over contact frames of the train split:
    wrench_std (6,) for [dv_c, dw_c] and dL_std (3,) for dL.  Zero mean keeps free flight at an exactly-zero input."""
    m = np.concatenate([ep.contact["n_contacts"] > 0 for ep in cache.episodes])
    dv = np.concatenate([ep.impulse["target_dv_contact"] for ep in cache.episodes]).astype(np.float64)[m]
    dw = np.concatenate([ep.impulse["target_dw_contact"] for ep in cache.episodes]).astype(np.float64)[m]
    dL = np.concatenate([ep.impulse["target_dL_contact"] for ep in cache.episodes]).astype(np.float64)[m]
    rms = lambda x: np.maximum(np.sqrt((x ** 2).mean(0)), 1e-6)  # noqa: E731
    return {"wrench_std": np.concatenate([rms(dv), rms(dw)]), "dL_std": rms(dL)}
