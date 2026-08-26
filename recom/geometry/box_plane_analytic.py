"""Analytic box-plane contact (correctness baseline and label derivation).

Everything here is exact for a rigid box (half-extents h = (hx, hy, hz)) against the
ground plane z = 0 with outward normal +Z.

Canonical contact contract (Study 1, K = 4 slots):
    n        : contact normal pointing FROM ground TOWARD box (= +Z for a flat ground)
    d        : signed distance along n;  d > 0 separated, d < 0 penetrating
    p_box    : point on the box surface (world frame), and its box-local coordinates
    p_ground : point on the ground plane (world frame) = p_box projected onto z = 0
"""
from __future__ import annotations

import numpy as np

from .transforms import quat_to_rotmat_np

# Corner signs, fixed order -> corner index i = 4*(sx>0) + 2*(sy>0) + (sz>0)
CORNER_SIGNS = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], dtype=np.float64)

MODE_NONE, MODE_CORNER, MODE_EDGE, MODE_FACE = 0, 1, 2, 3
MODE_NAMES = ["none", "corner", "edge", "face"]


def box_corners_local(half_extents: np.ndarray) -> np.ndarray:
    """(8, 3) corner coordinates in the box frame."""
    return CORNER_SIGNS * np.asarray(half_extents, dtype=np.float64)


def corner_gaps(pos: np.ndarray, quat: np.ndarray, half_extents: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (gaps (..., 8), corners_world (..., 8, 3)); gap = corner height above z = 0."""
    R = quat_to_rotmat_np(quat)  # (..., 3, 3)
    c_local = box_corners_local(half_extents)  # (8, 3)
    c_world = np.einsum("...ij,kj->...ki", R, c_local) + np.asarray(pos)[..., None, :]
    return c_world[..., 2], c_world


def min_gap(pos, quat, half_extents) -> float | np.ndarray:
    """Minimum analytic box-corner-to-plane gap (negative if penetrating)."""
    g, _ = corner_gaps(pos, quat, half_extents)
    return g.min(axis=-1)


FEATURE_ANGLE_TOL_DEG = 15.0


def lowest_feature_mode(quat: np.ndarray, half_extents: np.ndarray | None = None, angle_tol_deg: float = FEATURE_ANGLE_TOL_DEG) -> int:
    """Classify the lowest box feature for an orientation (pose-only, dimension-independent).

    Let u = R^T (0,0,-1) be the world "down" direction in the box frame with sorted |components|
    a >= b >= c.  face-dominant if the down direction is within angle_tol of a face normal
    (a >= cos tol); edge-dominant if it is within angle_tol of a plane spanned by two box axes
    (c <= sin tol) and not face-dominant; otherwise corner-dominant.
    """
    R = quat_to_rotmat_np(quat)
    u = np.sort(np.abs(R.T @ np.array([0.0, 0.0, -1.0])))[::-1]
    tol = np.deg2rad(angle_tol_deg)
    if u[0] >= np.cos(tol):
        return MODE_FACE
    if u[2] <= np.sin(tol):
        return MODE_EDGE
    return MODE_CORNER


def contact_mode_from_count(n_active: int) -> int:
    if n_active <= 0:
        return MODE_NONE
    if n_active == 1:
        return MODE_CORNER
    if n_active == 2:
        return MODE_EDGE
    return MODE_FACE


def analytic_contacts(
    pos: np.ndarray,
    quat: np.ndarray,
    half_extents: np.ndarray,
    margin: float,
    K: int = 4,
    coplanar_tol: float = 1e-4,
) -> dict:
    """Exact box-plane contact set in the canonical contract (single pose).

    Contacts are the corners with gap <= margin, restricted to the K lowest corners.
    Returns dict with keys: active (K,), d (K,), p_box_world (K,3), p_box_local (K,3),
    p_ground_world (K,3), n (K,3), mode (int), n_active (int), min_gap (float).
    """
    g, cw = corner_gaps(pos, quat, half_extents)
    order = np.argsort(g, kind="stable")
    sel = [i for i in order[:K] if g[i] <= margin]
    out = {
        "active": np.zeros(K, dtype=bool),
        "d": np.zeros(K, dtype=np.float64),
        "p_box_world": np.zeros((K, 3)),
        "p_box_local": np.zeros((K, 3)),
        "p_ground_world": np.zeros((K, 3)),
        "n": np.zeros((K, 3)),
        "corner_index": -np.ones(K, dtype=np.int64),
    }
    c_local = box_corners_local(half_extents)
    for k, i in enumerate(sel):
        out["active"][k] = True
        out["d"][k] = g[i]
        out["p_box_world"][k] = cw[i]
        out["p_box_local"][k] = c_local[i]
        out["p_ground_world"][k] = [cw[i, 0], cw[i, 1], 0.0]
        out["n"][k] = [0.0, 0.0, 1.0]
        out["corner_index"][k] = i
    out["n_active"] = len(sel)
    out["min_gap"] = float(g.min())
    # Mode from geometry of the lowest corners (independent of margin so that free flight has a
    # well-defined "approach mode" too): count corners within coplanar_tol of the min gap.
    n_low = int(np.sum(g <= g.min() + coplanar_tol))
    out["approach_mode"] = MODE_FACE if n_low >= 3 else (MODE_EDGE if n_low == 2 else MODE_CORNER)
    out["mode"] = contact_mode_from_count(len(sel))
    return out


def ballistic_state(pos0: np.ndarray, v0: np.ndarray, t: np.ndarray, g: float = 9.81) -> tuple[np.ndarray, np.ndarray]:
    """Free-flight analytic position/velocity for validation tests."""
    t = np.asarray(t)[..., None]
    a = np.array([0.0, 0.0, -g])
    return pos0 + v0 * t + 0.5 * a * t**2, v0 + a * t


def first_crossing_time(pos0: np.ndarray, quat0: np.ndarray, half_extents: np.ndarray, v0z: float = 0.0, g: float = 9.81) -> float:
    """Analytic time at which the lowest corner of a free-falling (non-rotating) box reaches z = 0."""
    h = float(min_gap(pos0, quat0, half_extents))
    # h + v0z t - 0.5 g t^2 = 0
    disc = v0z**2 + 2 * g * h
    return (v0z + np.sqrt(disc)) / g
