"""Versioned per-episode record schema (study plan Section 8).

Time alignment (frozen, Section 9):
    states[k]     : box state sampled BEFORE DoStepDynamics of step k, at t_k = k * dt
    raw/canon[k]  : Chrono contacts computed by the collision pass that runs at the START of
                    DoStepDynamics of step k, i.e. from states[k].  Reported after the step via
                    ReportAllContacts (verified: reported distance == pre-step analytic gap).
    forces[k]     : reaction forces from the solve of step k (contact-frame components; raw).
    states[k+1]   : state AFTER step k.
    Learning record:  (states[k], contacts[k]) -> states[k+1].

Arrays: N = number of steps; states have N + 1 rows; per-step arrays have N rows.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..config import K_SLOTS, RAW_MAX, SCHEMA_VERSION

STATE_FIELDS = {
    "t": (1,),
    "pos": (3,),             # world
    "quat": (4,),            # (w, x, y, z), canonical sign
    "lin_vel": (3,),         # world
    "ang_vel_world": (3,),
    "ang_vel_local": (3,),
    "settled": (),           # bool
}

RAW_CONTACT_FIELDS = {
    "raw_count": (),                      # int, contacts reported by Chrono this step (may exceed RAW_MAX)
    "raw_pA": (RAW_MAX, 3),
    "raw_pB": (RAW_MAX, 3),
    "raw_plane": (RAW_MAX, 3, 3),         # contact frame; X axis = normal (A -> B)
    "raw_distance": (RAW_MAX,),
    "raw_eff_radius": (RAW_MAX,),
    "raw_force": (RAW_MAX, 3),            # contact-frame components (x = normal)
    "raw_torque": (RAW_MAX, 3),
    "raw_A_is_box": (RAW_MAX,),           # bool
    "raw_constraint_offset": (RAW_MAX,),  # int
}

CANON_CONTACT_FIELDS = {
    "c_active": (K_SLOTS,),              # bool: Chrono reported this slot
    "c_d": (K_SLOTS,),                   # signed distance along n (ground -> box); <0 penetrating
    "c_n": (K_SLOTS, 3),                 # unit normal, ground -> box (world)
    "c_p_box_world": (K_SLOTS, 3),
    "c_p_box_local": (K_SLOTS, 3),       # box frame, clamped to half-extents (envelope inflation removed)
    "c_p_ground_world": (K_SLOTS, 3),
    "c_p_ground_local": (K_SLOTS, 3),
    "c_force_world": (K_SLOTS, 3),       # force ON THE BOX in world frame (may be zero: NSC caveat)
    "c_raw_index": (K_SLOTS,),           # index into the raw arrays (-1 if none)
    "n_contacts": (),                    # int, min(raw_count, K)
    "n_penetrating": (),                 # int, contacts with d <= 0
    "contact_mode": (),                  # 0 none / 1 corner / 2 edge / 3 face (from n_contacts)
    "approach_mode": (),                 # lowest-feature classification from pose (corner/edge/face)
    "analytic_min_gap": (),
    "analytic_corner_gaps": (8,),
    "near_contact": (),                  # bool: analytic_min_gap <= near margin and no Chrono contact
}


@dataclass
class EpisodeRecord:
    meta: dict[str, Any]
    states: dict[str, np.ndarray]
    raw: dict[str, np.ndarray]
    canon: dict[str, np.ndarray] = field(default_factory=dict)
    events: dict[str, Any] = field(default_factory=dict)

    @property
    def n_steps(self) -> int:
        return int(self.raw["raw_count"].shape[0])

    def arrays(self) -> dict[str, np.ndarray]:
        out = {}
        for grp, prefix in ((self.states, "s_"), (self.raw, ""), (self.canon, "")):
            for k, v in grp.items():
                out[prefix + k] = v
        return out

    @classmethod
    def from_arrays(cls, meta: dict[str, Any], arrs: dict[str, np.ndarray], events: dict[str, Any]) -> "EpisodeRecord":
        states = {k: arrs["s_" + k] for k in STATE_FIELDS if "s_" + k in arrs}
        raw = {k: arrs[k] for k in RAW_CONTACT_FIELDS if k in arrs}
        canon = {k: arrs[k] for k in CANON_CONTACT_FIELDS if k in arrs}
        return cls(meta=meta, states=states, raw=raw, canon=canon, events=events)


def empty_raw(n: int) -> dict[str, np.ndarray]:
    out = {}
    for k, shape in RAW_CONTACT_FIELDS.items():
        if k == "raw_count":
            out[k] = np.zeros(n, dtype=np.int32)
        elif k == "raw_A_is_box":
            out[k] = np.zeros((n,) + shape, dtype=bool)
        elif k == "raw_constraint_offset":
            out[k] = -np.ones((n,) + shape, dtype=np.int32)
        else:
            out[k] = np.zeros((n,) + shape, dtype=np.float64)
    return out


def empty_states(n_plus_1: int) -> dict[str, np.ndarray]:
    out = {}
    for k, shape in STATE_FIELDS.items():
        if k == "settled":
            out[k] = np.zeros(n_plus_1, dtype=bool)
        elif k == "t":
            out[k] = np.zeros(n_plus_1, dtype=np.float64)
        else:
            out[k] = np.zeros((n_plus_1,) + shape, dtype=np.float64)
    return out


__all__ = ["SCHEMA_VERSION", "K_SLOTS", "RAW_MAX", "STATE_FIELDS", "RAW_CONTACT_FIELDS", "CANON_CONTACT_FIELDS", "EpisodeRecord", "empty_raw", "empty_states"]
