"""Frozen configuration schema for Study 1 (box drop).

Everything that affects the physics or the sampled initial conditions lives here so that an
episode can be replayed exactly from its stored metadata.
"""
from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

SCHEMA_VERSION = "recom.boxdrop.v1"
K_SLOTS = 4        # canonical contact slots per frame (Bullet reports <= 4 per pair)
RAW_MAX = 8        # raw contact records kept per frame (overflow is counted, never silently lost)


@dataclass
class PhysicsConfig:
    dt: float = 1e-3
    max_duration: float = 2.0
    gravity: float = 9.81
    contact_method: str = "NSC"                   # fixed for the primary dataset
    collision_system: str = "BULLET"
    solver: str = "APGD"
    solver_max_iterations: int = 100
    solver_tolerance: float = 1e-6
    timestepper: str = "EULER_IMPLICIT_LINEARIZED"
    collision_envelope: float = 1e-3              # proximity margin at which Chrono reports contacts
    collision_margin: float = 5e-4
    min_bounce_speed: float = 0.15
    friction: float = 0.5
    restitution: float = 0.3
    rolling_friction: float = 0.0
    spinning_friction: float = 0.0
    density: float = 1000.0
    ground_size: tuple[float, float, float] = (6.0, 6.0, 0.2)   # ground box; top face is z = 0
    num_threads: int = 1
    allow_sleeping: bool = False
    # settling / early termination (documented condition; see sim.chrono_box_drop.run_episode).
    # NOTE: a box resting on 4 NSC contacts shows a deterministic rocking limit cycle
    # (|v| ~ 3 mm/s, |w| ~ 0.08 rad/s, penetration +-7 um, zero net drift), so settling is
    # detected from pose change over a window, not from instantaneous velocity.
    settle_pos_tol: float = 1e-4                  # max corner displacement over the window [m]
    settle_window_steps: int = 200
    settle_min_contacts: int = 3
    settle_mean_speed_tol: float = 1e-2           # mean |v| over the window [m/s]
    # derived-label margins
    near_contact_margin: float = 0.01             # analytic min gap <= margin -> near_contact


@dataclass
class BoxSpec:
    half_extents: tuple[float, float, float]
    geometry_id: str = ""
    geometry_group: str = ""                      # holdout grouping key (e.g. "g017")

    @property
    def sides(self) -> np.ndarray:
        return 2.0 * np.asarray(self.half_extents)


@dataclass
class EpisodeConfig:
    episode_id: str
    seed: int
    phase: str
    box: BoxSpec
    clearance: float                              # min box-to-ground clearance at release [m]
    pos_xy: tuple[float, float]
    quat: tuple[float, float, float, float]       # (w, x, y, z), canonical sign
    lin_vel0: tuple[float, float, float] = (0.0, 0.0, 0.0)
    ang_vel0: tuple[float, float, float] = (0.0, 0.0, 0.0)  # world frame
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    approach_mode: int = -1                       # lowest feature at release: 1 corner, 2 edge, 3 face

    def to_dict(self) -> dict[str, Any]:
        return _to_jsonable(dataclasses.asdict(self))

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "EpisodeConfig":
        d = dict(d)
        d["box"] = BoxSpec(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in d["box"].items()})
        d["physics"] = PhysicsConfig(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in d["physics"].items()})
        for k in ("pos_xy", "quat", "lin_vel0", "ang_vel0"):
            d[k] = tuple(d[k])
        return cls(**d)


@dataclass
class DatasetGenConfig:
    name: str = "smoke"
    output_dir: str = "data/smoke"
    phase: str = "1A"                             # "1A" fixed box, "1B" variable box, "1C" + initial twist
    n_episodes: int = 50
    seed: int = 12345
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    # geometry
    fixed_box_sides: tuple[float, float, float] = (0.20, 0.15, 0.10)
    n_geometries: int = 1                         # distinct box instances (episodes cycle through them)
    side_range: tuple[float, float] = (0.05, 0.30)
    max_aspect: float = 4.0
    # release conditions
    clearance_range: tuple[float, float] = (0.10, 1.50)
    xy_range: float = 0.0
    orientation_strata: dict | None = None        # e.g. {"face": 0.15, "edge": 0.15}; None = uniform SO(3)
    lin_vel_range: float = 0.0                    # Phase 1C only
    ang_vel_range: float = 0.0                    # Phase 1C only
    n_workers: int = 8

    def to_dict(self) -> dict[str, Any]:
        return _to_jsonable(dataclasses.asdict(self))

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "DatasetGenConfig":
        d = dict(d)
        if "physics" in d:
            d["physics"] = PhysicsConfig(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in d["physics"].items()})
        for k in ("fixed_box_sides", "side_range", "clearance_range"):
            if k in d:
                d[k] = tuple(d[k])
        return cls(**d)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "DatasetGenConfig":
        with open(path) as f:
            return cls.from_dict(yaml.safe_load(f))


def _to_jsonable(o: Any) -> Any:
    if isinstance(o, dict):
        return {k: _to_jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_to_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return o


def dump_json(obj: Any, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(_to_jsonable(obj), f, indent=2)


def load_json(path: str | Path) -> Any:
    with open(path) as f:
        return json.load(f)
