"""Raw Chrono contacts -> canonical K-slot contact view + event labels (Section 8.4).

Canonical convention (frozen):
    A = box, B = ground.
    n = unit normal pointing from ground toward box (+Z for the flat ground).
    d = signed distance along n; d > 0 separated, d < 0 penetrating.
    p_box_local is clamped to the half-extents: Chrono/Bullet reports box points on the
    envelope-inflated shape laterally (verified: +-envelope offset), exact along the normal.
    p_ground = projection of the corrected box point onto the ground plane along n.
Raw arrays are never modified.
"""
from __future__ import annotations

import numpy as np

from ..config import K_SLOTS
from ..geometry.box_plane_analytic import MODE_NONE, contact_mode_from_count, corner_gaps
from ..geometry.transforms import quat_to_rotmat_np
from .schema import EpisodeRecord


def _empty_canon(n: int) -> dict[str, np.ndarray]:
    K = K_SLOTS
    return {
        "c_active": np.zeros((n, K), dtype=bool),
        "c_d": np.zeros((n, K)),
        "c_n": np.zeros((n, K, 3)),
        "c_p_box_world": np.zeros((n, K, 3)),
        "c_p_box_local": np.zeros((n, K, 3)),
        "c_p_ground_world": np.zeros((n, K, 3)),
        "c_p_ground_local": np.zeros((n, K, 3)),
        "c_force_world": np.zeros((n, K, 3)),
        "c_raw_index": -np.ones((n, K), dtype=np.int32),
        "n_contacts": np.zeros(n, dtype=np.int32),
        "n_penetrating": np.zeros(n, dtype=np.int32),
        "contact_mode": np.zeros(n, dtype=np.int32),
        "approach_mode": np.zeros(n, dtype=np.int32),
        "analytic_min_gap": np.zeros(n),
        "analytic_corner_gaps": np.zeros((n, 8)),
        "near_contact": np.zeros(n, dtype=bool),
    }


def canonicalize_frame(raw_rows: dict[str, np.ndarray], count: int, pos: np.ndarray, quat: np.ndarray, he: np.ndarray, ground_pos=None, ground_quat=None) -> dict[str, np.ndarray]:
    """Canonicalize the raw contacts of one frame. Returns per-slot arrays (K,...)."""
    K = K_SLOTS
    out = {k: v[0].copy() for k, v in _empty_canon(1).items() if k.startswith("c_")}
    R = quat_to_rotmat_np(quat)
    m = min(count, K)
    # Order slots deterministically: by raw index (Chrono manifold order) -- slots are treated as
    # an unordered set by the losses anyway.
    for j in range(m):
        a_is_box = bool(raw_rows["raw_A_is_box"][j])
        pA, pB = raw_rows["raw_pA"][j], raw_rows["raw_pB"][j]
        plane = raw_rows["raw_plane"][j]
        n_raw = plane[:, 0]  # X axis, A -> B
        F_raw = plane @ raw_rows["raw_force"][j]  # contact-frame components -> world; force on B (Chrono)
        if a_is_box:
            p_box, p_gnd = pA, pB
            n = -n_raw          # A->B is box->ground; flip to ground->box
            F_box = -F_raw      # Chrono reports force applied to B... (sign verified in tests: F.x > 0 on resting)
        else:
            p_box, p_gnd = pB, pA
            n = n_raw
            F_box = F_raw
        n = n / max(np.linalg.norm(n), 1e-12)
        d = float(raw_rows["raw_distance"][j])
        p_box_local = R.T @ (p_box - pos)
        p_box_local = np.clip(p_box_local, -he, he)
        p_box_world_c = R @ p_box_local + pos
        # Ground point: corrected box point projected along the normal onto the ground plane through
        # the raw ground point (removes the lateral envelope inflation; raw pA/pB are kept untouched).
        p_gnd = p_box_world_c - float(np.dot(p_box_world_c - p_gnd, n)) * n
        if ground_pos is None:
            p_gnd_local = p_gnd.copy()
        else:
            Rg = quat_to_rotmat_np(ground_quat)
            p_gnd_local = Rg.T @ (p_gnd - ground_pos)
        out["c_active"][j] = True
        out["c_d"][j] = d
        out["c_n"][j] = n
        out["c_p_box_world"][j] = p_box_world_c
        out["c_p_box_local"][j] = p_box_local
        out["c_p_ground_world"][j] = p_gnd
        out["c_p_ground_local"][j] = p_gnd_local
        out["c_force_world"][j] = F_box
        out["c_raw_index"][j] = j
    return out


def canonicalize_episode(rec: EpisodeRecord) -> EpisodeRecord:
    n = rec.n_steps
    he = np.asarray(rec.meta["half_extents"])
    near_margin = rec.meta["episode"]["physics"]["near_contact_margin"]
    canon = _empty_canon(n)
    pos, quat = rec.states["pos"], rec.states["quat"]
    gaps, _ = corner_gaps(pos[:n], quat[:n], he)
    canon["analytic_corner_gaps"] = gaps
    canon["analytic_min_gap"] = gaps.min(axis=1)
    n_low = (gaps <= gaps.min(axis=1, keepdims=True) + 1e-4).sum(axis=1)
    canon["approach_mode"] = np.where(n_low >= 3, 3, np.where(n_low == 2, 2, 1)).astype(np.int32)
    for k in range(n):
        cnt = int(rec.raw["raw_count"][k])
        if cnt > 0:
            rows = {key: rec.raw[key][k] for key in rec.raw if key != "raw_count"}
            fr = canonicalize_frame(rows, cnt, pos[k], quat[k], he)
            for key, val in fr.items():
                canon[key][k] = val
        m = min(cnt, K_SLOTS)
        canon["n_contacts"][k] = m
        canon["n_penetrating"][k] = int(np.sum(canon["c_active"][k] & (canon["c_d"][k] <= 0.0)))
        canon["contact_mode"][k] = contact_mode_from_count(m)
    canon["near_contact"] = (canon["analytic_min_gap"] <= near_margin) & (canon["n_contacts"] == 0)
    rec.canon = canon
    rec.events = derive_events(rec)
    return rec


def derive_events(rec: EpisodeRecord) -> dict:
    """First impact, rebound intervals, resting interval, settle time."""
    n = rec.n_steps
    nc = rec.canon["n_contacts"]
    dt = rec.meta["dt"]
    active = nc > 0
    first = int(np.argmax(active)) if active.any() else -1
    # contact episodes: maximal runs of active frames
    runs = []
    k = 0
    while k < n:
        if active[k]:
            j = k
            while j < n and active[j]:
                j += 1
            runs.append((k, j))
            k = j
        else:
            k += 1
    # rebound intervals = free-flight gaps between contact runs (after first impact)
    rebounds = [(runs[i][1], runs[i + 1][0]) for i in range(len(runs) - 1)]
    settled = rec.states["settled"]
    settle_step = int(np.argmax(settled)) if settled.any() else -1
    # resting: last contact run if the episode settled
    rest = [runs[-1][0], n] if (runs and settle_step >= 0) else None
    return {
        "first_impact_step": first,
        "first_impact_time": first * dt if first >= 0 else None,
        "contact_runs": runs,
        "rebound_intervals": rebounds,
        "n_impacts": len(runs),
        "settle_step": settle_step,
        "settle_time": settle_step * dt if settle_step >= 0 else None,
        "resting_interval": rest,
        "max_penetration": float(-min(0.0, rec.canon["c_d"][rec.canon["c_active"]].min())) if active.any() else 0.0,
        "min_analytic_gap": float(rec.canon["analytic_min_gap"].min()),
    }
