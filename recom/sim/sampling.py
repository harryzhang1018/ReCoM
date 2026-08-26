"""Deterministic initial-condition sampling for box-drop episodes (study plan Section 6)."""
from __future__ import annotations

import numpy as np

from ..config import BoxSpec, DatasetGenConfig, EpisodeConfig
from ..geometry.box_plane_analytic import MODE_NAMES, lowest_feature_mode, min_gap
from ..geometry.transforms import quat_canonical_np, quat_from_axis_angle_np, quat_mul_np, random_quat_uniform_np

GEOM_CATEGORIES = ("cube", "elongated", "flat")


def _geom_rng(cfg: DatasetGenConfig, g: int) -> np.random.Generator:
    return np.random.default_rng([cfg.seed, 7_000_003, g])


def _episode_rng(cfg: DatasetGenConfig, ep: int) -> np.random.Generator:
    return np.random.default_rng([cfg.seed, 11_000_017, ep])


def sample_box_sides(cfg: DatasetGenConfig, g: int) -> tuple[np.ndarray, str]:
    """Stratified box sides: cube-like / elongated / flat, sides in side_range, aspect <= max_aspect."""
    rng = _geom_rng(cfg, g)
    lo, hi = cfg.side_range
    cat = GEOM_CATEGORIES[g % len(GEOM_CATEGORIES)]
    for _ in range(10_000):
        if cat == "cube":
            base = rng.uniform(lo, hi)
            s = base * rng.uniform(1 / 1.3, 1.3, size=3)
        elif cat == "elongated":
            base = rng.uniform(lo, hi / 2.5)
            s = np.array([base * rng.uniform(2.0, cfg.max_aspect), base * rng.uniform(0.8, 1.25), base * rng.uniform(0.8, 1.25)])
        else:  # flat
            base = rng.uniform(lo * 2.0, hi)
            s = np.array([base * rng.uniform(0.8, 1.25), base * rng.uniform(0.8, 1.25), base * rng.uniform(1 / cfg.max_aspect, 0.5)])
        s = s[rng.permutation(3)]
        if np.all(s >= lo) and np.all(s <= hi) and s.max() / s.min() <= cfg.max_aspect:
            return s, cat
    raise RuntimeError("could not sample box sides under constraints")


def make_box_spec(cfg: DatasetGenConfig, g: int) -> BoxSpec:
    if cfg.phase == "1A" or cfg.n_geometries <= 1:
        sides = np.asarray(cfg.fixed_box_sides, dtype=np.float64)
        return BoxSpec(half_extents=tuple((sides / 2).tolist()), geometry_id="box_fixed", geometry_group="g000")
    sides, cat = sample_box_sides(cfg, g)
    return BoxSpec(
        half_extents=tuple((sides / 2).tolist()),
        geometry_id=f"box_{cat}_{g:04d}",
        geometry_group=f"g{g:03d}",
    )


_AXES = np.eye(3)


def sample_orientation(rng: np.random.Generator, strata: dict[str, float] | None) -> np.ndarray:
    """Uniform on SO(3), or stratified: with prob strata[face|edge] produce a near-face/near-edge pose.

    face: a random face normal within 15 deg of world -Z, random yaw.
    edge: a random edge nearly horizontal and lowest (tilt about the edge axis in [20, 70] deg,
          perturbation of the edge from horizontal < 10 deg), random yaw.
    Stratification changes the marginal orientation distribution; it is recorded in gen_config.
    """
    if not strata:
        return random_quat_uniform_np(rng)
    r = rng.random()
    p_face, p_edge = strata.get("face", 0.0), strata.get("edge", 0.0)
    if r >= p_face + p_edge:
        return random_quat_uniform_np(rng)
    yaw = quat_from_axis_angle_np([0, 0, 1], rng.uniform(0, 2 * np.pi))
    # rotation taking box axis `ax` (sign s) to world -Z
    ax, s_ = rng.integers(3), rng.choice([-1.0, 1.0])
    target, src = np.array([0.0, 0.0, -1.0]), s_ * _AXES[ax]
    v = np.cross(src, target)
    if np.linalg.norm(v) < 1e-9:
        base = np.array([1.0, 0, 0, 0]) if np.dot(src, target) > 0 else quat_from_axis_angle_np([1, 0, 0], np.pi)
    else:
        base = quat_from_axis_angle_np(v, np.arccos(np.clip(np.dot(src, target), -1, 1)))
    if r < p_face:
        tilt_dir = rng.uniform(0, 2 * np.pi)
        tilt = quat_from_axis_angle_np([np.cos(tilt_dir), np.sin(tilt_dir), 0], rng.uniform(0, np.deg2rad(12)))
        return quat_canonical_np(quat_mul_np(yaw, quat_mul_np(tilt, base)))
    # edge: after `base`, tilt about a horizontal axis (edge axis) by 20-70 deg, then small perturbation
    tilt = quat_from_axis_angle_np([1, 0, 0], rng.choice([-1, 1]) * rng.uniform(np.deg2rad(20), np.deg2rad(70)))
    pert = quat_from_axis_angle_np([0, 1, 0], rng.uniform(-np.deg2rad(8), np.deg2rad(8)))
    return quat_canonical_np(quat_mul_np(yaw, quat_mul_np(pert, quat_mul_np(tilt, base))))


def sample_episode(cfg: DatasetGenConfig, ep: int) -> EpisodeConfig:
    """Sample one episode deterministically from (cfg.seed, ep)."""
    g = ep % max(cfg.n_geometries, 1)
    box = make_box_spec(cfg, g)
    he = np.asarray(box.half_extents)
    rng = _episode_rng(cfg, ep)
    for _ in range(1000):
        q = sample_orientation(rng, cfg.orientation_strata)
        clearance = rng.uniform(*cfg.clearance_range)
        xy = rng.uniform(-cfg.xy_range, cfg.xy_range, size=2) if cfg.xy_range > 0 else np.zeros(2)
        # position so that the lowest corner sits exactly `clearance` above the plane
        z = clearance - float(min_gap(np.zeros(3), q, he))
        pos = np.array([xy[0], xy[1], z])
        if min_gap(pos, q, he) >= cfg.clearance_range[0] - 1e-9:
            break
    else:  # pragma: no cover
        raise RuntimeError("rejection sampling failed")
    lin_vel0 = rng.uniform(-cfg.lin_vel_range, cfg.lin_vel_range, size=3) if cfg.lin_vel_range > 0 else np.zeros(3)
    ang_vel0 = rng.uniform(-cfg.ang_vel_range, cfg.ang_vel_range, size=3) if cfg.ang_vel_range > 0 else np.zeros(3)
    return EpisodeConfig(
        episode_id=f"{cfg.name}_{ep:05d}",
        seed=int(ep),
        phase=cfg.phase,
        box=box,
        clearance=float(clearance),
        pos_xy=(float(pos[0]), float(pos[1])),
        quat=tuple(quat_canonical_np(q).tolist()),
        lin_vel0=tuple(lin_vel0.tolist()),
        ang_vel0=tuple(ang_vel0.tolist()),
        physics=cfg.physics,
        approach_mode=lowest_feature_mode(q, he),
    )


def release_position(ep: EpisodeConfig) -> np.ndarray:
    he = np.asarray(ep.box.half_extents)
    q = np.asarray(ep.quat)
    z = ep.clearance - float(min_gap(np.zeros(3), q, he))
    return np.array([ep.pos_xy[0], ep.pos_xy[1], z])


def orientation_coverage(episodes: list[EpisodeConfig]) -> dict[str, int]:
    """Diagnostic: distribution of the initially lowest feature (corner/edge/face)."""
    counts = {MODE_NAMES[m]: 0 for m in (1, 2, 3)}
    for e in episodes:
        counts[MODE_NAMES[e.approach_mode]] += 1
    return counts
