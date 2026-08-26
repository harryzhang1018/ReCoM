"""Dataset validation (study plan Section 18 "Dataset tests") and record/replay test (Section 9)."""
from __future__ import annotations

import numpy as np

from ..config import K_SLOTS, EpisodeConfig
from ..geometry.box_plane_analytic import ballistic_state, first_crossing_time
from ..geometry.transforms import local_to_world_np, quat_to_rotmat_np
from .schema import EpisodeRecord


def validate_episode(rec: EpisodeRecord, tol: dict | None = None) -> list[str]:
    """Return a list of failure messages (empty = pass)."""
    tol = tol or {}
    fails: list[str] = []
    S, R, C = rec.states, rec.raw, rec.canon
    n = rec.n_steps
    # finiteness
    for grp in (S, R, C):
        for k, v in grp.items():
            if v.dtype.kind == "f" and not np.isfinite(v).all():
                fails.append(f"non-finite values in {k}")
    # shapes
    if S["pos"].shape[0] != n + 1:
        fails.append("states must have n_steps + 1 rows")
    # quaternions
    qn = np.linalg.norm(S["quat"], axis=1)
    if np.abs(qn - 1).max() > 1e-6:
        fails.append(f"non-unit quaternion, max |q|-1 = {np.abs(qn - 1).max():.2e}")
    w = S["quat"][:, 0]
    tie = np.isclose(w, 0.0)  # same tolerance as quat_canonical_np: w ~ 0 -> tie-break on x
    if (w[~tie] < 0).any() or (S["quat"][tie, 1] < -1e-12).any():
        fails.append("quaternion sign not canonical (w < 0)")
    # time
    dt = rec.meta["dt"]
    if not np.allclose(np.diff(S["t"]), dt, atol=1e-12):
        fails.append("timestamps not monotonic with fixed dt")
    # raw -> canonical traceability
    m = np.minimum(R["raw_count"], K_SLOTS)
    if not np.array_equal(C["n_contacts"], m):
        fails.append("n_contacts != min(raw_count, K)")
    if not np.array_equal(C["c_active"].sum(1), m):
        fails.append("active slot count != n_contacts")
    if (R["raw_count"] > rec.raw["raw_pA"].shape[1]).any():
        fails.append("raw contact overflow (RAW_MAX too small)")
    # frame consistency: box-local <-> world
    act = C["c_active"]
    if act.any():
        idx = np.argwhere(act)
        pw = np.stack([local_to_world_np(C["c_p_box_local"][k, j], S["pos"][k], S["quat"][k]) for k, j in idx])
        err = np.linalg.norm(pw - C["c_p_box_world"][idx[:, 0], idx[:, 1]], axis=1).max()
        if err > 1e-9:
            fails.append(f"box-local/world contact point mismatch {err:.2e}")
        # canonical box point must lie on the box surface (some coordinate at +-h)
        he = np.asarray(rec.meta["half_extents"])
        pl = C["c_p_box_local"][act]
        on_surf = (np.abs(np.abs(pl) - he) < 1e-6).any(axis=1)
        if not on_surf.all():
            fails.append("canonical box-local contact point not on box surface")
        if (np.abs(pl) > he + 1e-9).any():
            fails.append("box-local contact point outside box")
        # normal convention: ground -> box => +Z for the flat ground
        nz = C["c_n"][act][:, 2]
        if (nz < 0.99).any():
            fails.append(f"canonical normal not +Z (min nz={nz.min():.3f})")
        # ground point on plane
        if np.abs(C["c_p_ground_world"][act][:, 2]).max() > 1e-6:
            fails.append("ground contact point not on z = 0")
        # signed distance sign vs analytic gap of the matched corner: |d - gap| small
        d = C["c_d"][act]
        gap_at = np.array([C["analytic_corner_gaps"][k].min() for k, j in idx])
        if np.median(np.abs(d - gap_at)) > tol.get("d_vs_gap_median", 2e-3):
            fails.append(f"signed distance disagrees with analytic gap (median {np.median(np.abs(d - gap_at)):.2e})")
    # reported contacts only within proximity (envelope + margins)
    env = rec.meta["collision_envelope"] + 2 * rec.meta["collision_margin"]
    if act.any() and C["c_d"][act].max() > env + 2e-3:
        fails.append(f"contact reported at d={C['c_d'][act].max():.4f} > proximity margin")
    return fails


def free_flight_check(rec: EpisodeRecord, atol_pos: float = 1e-4, atol_vel: float = 1e-3) -> dict:
    """Compare the pre-impact trajectory with analytic ballistics (validation 1 of Section 9)."""
    fi = rec.events["first_impact_step"]
    n_free = fi if fi >= 0 else rec.n_steps
    S = rec.states
    t = S["t"][:n_free + 1]
    p_an, v_an = ballistic_state(S["pos"][0], S["lin_vel"][0], t, g=rec.meta["episode"]["physics"]["gravity"])
    pos_err = np.abs(S["pos"][:n_free + 1] - p_an).max() if n_free > 0 else 0.0
    vel_err = np.abs(S["lin_vel"][:n_free + 1] - v_an).max() if n_free > 0 else 0.0
    # semi-implicit Euler: position lags analytic by <= 0.5*g*dt*t; use a generous tolerance
    return {"n_free_steps": int(n_free), "pos_err": float(pos_err), "vel_err": float(vel_err), "pass": bool(pos_err < atol_pos + 0.5 * 9.81 * rec.meta["dt"] * t[-1] and vel_err < atol_vel)}


def first_contact_timing_check(rec: EpisodeRecord) -> dict:
    """Validation 2 of Section 9: first reported contact within one step of the analytic envelope crossing."""
    ep = EpisodeConfig.from_dict(rec.meta["episode"])
    he = np.asarray(ep.box.half_extents)
    dt = rec.meta["dt"]
    fi = rec.events["first_impact_step"]
    # crossing of the proximity margin (contact is reported once gap <= envelope + margins)
    prox = rec.meta["collision_envelope"] + 2 * rec.meta["collision_margin"]
    pos0, q0 = rec.states["pos"][0], rec.states["quat"][0]
    from ..geometry.box_plane_analytic import min_gap
    h = float(min_gap(pos0, q0, he)) - prox
    g = ep.physics.gravity
    t_cross = (ep.lin_vel0[2] + np.sqrt(ep.lin_vel0[2] ** 2 + 2 * g * max(h, 0))) / g
    k_cross = t_cross / dt
    return {"first_impact_step": int(fi), "analytic_cross_step": float(k_cross), "err_steps": float(fi - k_cross), "pass": bool(fi >= 0 and abs(fi - k_cross) <= 1.5)}


def replay_check(rec: EpisodeRecord, atol_state: float = 1e-9) -> dict:
    """Validation 4 of Section 9: replay the stored config and compare states/contacts."""
    from ..sim.chrono_box_drop import run_episode
    ep = EpisodeConfig.from_dict(rec.meta["episode"])
    rec2 = run_episode(ep)
    n = min(rec.n_steps, rec2.n_steps)
    out = {"same_length": rec.n_steps == rec2.n_steps}
    for k in ("pos", "quat", "lin_vel", "ang_vel_world"):
        out[f"max_abs_diff_{k}"] = float(np.abs(rec.states[k][: n + 1] - rec2.states[k][: n + 1]).max())
    out["raw_count_equal"] = bool(np.array_equal(rec.raw["raw_count"][:n], rec2.raw["raw_count"][:n]))
    out["max_abs_diff_raw_distance"] = float(np.abs(rec.raw["raw_distance"][:n] - rec2.raw["raw_distance"][:n]).max())
    out["pass"] = bool(out["same_length"] and out["raw_count_equal"] and all(out[f"max_abs_diff_{k}"] <= atol_state for k in ("pos", "quat", "lin_vel", "ang_vel_world")))
    return out
