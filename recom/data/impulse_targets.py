"""Contact-impulse (wrench) labels derived from recorded Chrono states (encoder-decoder plan, stage ED0).

LABEL_VERSION is stored in every run config that consumes these labels.

Chrono NSC + Euler-implicit-linearized discrete update, verified on pilot1b (see docs/PROGRESS.md):
    m (v[k+1] - v[k])                                   = m g_vec dt + J[k]          (exact to ~1e-15)
    I_b (w_b[k+1] - w_b[k]) + dt * (w_b[k] x I_b w_b[k]) = L_b[k]                   (exact to ~1e-13)
with J the net contact impulse (world) and L_b the net angular contact impulse about the COM in the body
frame.  The world-frame version H[k+1]-H[k] is NOT exact (0.5-4 % error) and is kept only for the audit.
Per-contact impulses: j_i = c_force_world_i * dt; sum_i j_i == J exactly; sum_i r_i x j_i == R L_b exactly
when r_i uses Chrono's raw contact point (canonical clamped points give ~1 % median lever-arm error).

All learning labels are mass-normalized (velocity units):
    dv_c = J / m,   dL = R L_b / m,   dw_c = R I_b^-1 L_b   (contact-induced delta v / delta omega_world)
Gyroscopic free-flight prior (no contact):  dw_gyro_world = -dt R I_b^-1 (w_b x I_b w_b)  (exact).
"""
from __future__ import annotations

import numpy as np

from ..config import K_SLOTS
from ..geometry.transforms import quat_to_rotmat_np

LABEL_VERSION = "impulse_labels.v1"
PHYS_KEYS = ("mass", "log_mass", "inertia_diag", "inertia_diag_over_m", "mu", "restitution", "dt", "gravity", "envelope")


def gravity_vec(g: float) -> np.ndarray:
    return np.array([0.0, 0.0, -float(g)])


def linear_contact_dv(lin_vel: np.ndarray, g: float, dt: float) -> np.ndarray:
    """(N+1,3) world velocities -> (N,3) contact-induced delta v = J/m (gravity impulse removed)."""
    v = np.asarray(lin_vel, dtype=np.float64)
    return v[1:] - v[:-1] - gravity_vec(g) * dt


def gyro_torque_body(w_b: np.ndarray, inertia_diag: np.ndarray) -> np.ndarray:
    """w x (I w) in the body frame, (...,3)."""
    I = np.asarray(inertia_diag, dtype=np.float64)
    w = np.asarray(w_b, dtype=np.float64)
    return np.cross(w, I * w)


def angular_contact_impulse_body(ang_vel_local: np.ndarray, inertia_diag: np.ndarray, dt: float) -> np.ndarray:
    """(N+1,3) body-frame angular velocities -> (N,3) net angular contact impulse L_b (exact Chrono form)."""
    w = np.asarray(ang_vel_local, dtype=np.float64)
    I = np.asarray(inertia_diag, dtype=np.float64)
    return I * (w[1:] - w[:-1]) + dt * gyro_torque_body(w[:-1], I)


def gyro_dw_body(w_b: np.ndarray, inertia_diag: np.ndarray, dt: float) -> np.ndarray:
    """Torque-free (gyroscopic) body-frame delta omega over one step: -dt I^-1 (w x I w).  Invariant to I -> c I."""
    I = np.asarray(inertia_diag, dtype=np.float64)
    return -dt * gyro_torque_body(w_b, I) / I


def body_to_world(vec_b: np.ndarray, quat: np.ndarray) -> np.ndarray:
    R = quat_to_rotmat_np(quat)
    return np.einsum("...ij,...j->...i", R, np.asarray(vec_b, dtype=np.float64))


def world_to_body(vec_w: np.ndarray, quat: np.ndarray) -> np.ndarray:
    R = quat_to_rotmat_np(quat)
    return np.einsum("...ji,...j->...i", R, np.asarray(vec_w, dtype=np.float64))


def gyro_dw_world(quat: np.ndarray, w_world: np.ndarray, inertia_diag_over_m: np.ndarray, dt: float) -> np.ndarray:
    """Exact torque-free delta omega_world for one Chrono step from (q[k], w_world[k]).  (...,4),(...,3) -> (...,3)."""
    w_b = world_to_body(w_world, quat)
    return body_to_world(gyro_dw_body(w_b, inertia_diag_over_m, dt), quat)


def contact_dw_world(L_b: np.ndarray, quat: np.ndarray, inertia_diag: np.ndarray) -> np.ndarray:
    """Contact-induced delta omega_world = R I_b^-1 L_b."""
    return body_to_world(np.asarray(L_b, dtype=np.float64) / np.asarray(inertia_diag, dtype=np.float64), quat)


def naive_dH_world(ang_vel_local: np.ndarray, quat: np.ndarray, inertia_diag: np.ndarray) -> np.ndarray:
    """Audit only: world angular-momentum difference H[k+1]-H[k] (NOT Chrono's discrete update)."""
    w = np.asarray(ang_vel_local, dtype=np.float64)
    H = body_to_world(np.asarray(inertia_diag, dtype=np.float64) * w, quat)
    return H[1:] - H[:-1]


def box_inertia_diag_over_m(half_extents: np.ndarray) -> np.ndarray:
    """I_b / m of a homogeneous box, (...,3) -> (...,3).  Matches recom.sim.chrono_box_drop.inertia_box."""
    he = np.asarray(half_extents, dtype=np.float64)
    x2, y2, z2 = he[..., 0] ** 2, he[..., 1] ** 2, he[..., 2] ** 2
    return np.stack([y2 + z2, x2 + z2, x2 + y2], -1) / 3.0


def phys_from_meta(meta: dict) -> dict:
    """Physical parameters of an episode from its stored metadata (floats / (3,) arrays, float64)."""
    m = float(meta["mass"])
    I = np.asarray(meta["inertia_xx"], dtype=np.float64)
    return {
        "mass": m, "log_mass": float(np.log(m)), "inertia_diag": I, "inertia_diag_over_m": I / m,
        "mu": float(meta["friction"]), "restitution": float(meta["restitution"]), "dt": float(meta["dt"]),
        "gravity": float(meta["episode"]["physics"]["gravity"]), "density": float(meta.get("density", np.nan)),
        "envelope": float(meta.get("collision_envelope_actual", meta["episode"]["physics"]["collision_envelope"])),
    }


def phys_arrays(meta: dict) -> dict[str, np.ndarray]:
    """PHYS_KEYS as float32 numpy arrays (0-d or (3,)) for dataset items."""
    p = phys_from_meta(meta)
    return {k: np.asarray(p[k], dtype=np.float32) for k in PHYS_KEYS}


def _box_local_ang_vel(rec) -> np.ndarray:
    S = rec.states
    if "ang_vel_local" in S:
        return np.asarray(S["ang_vel_local"], dtype=np.float64)
    return world_to_body(S["ang_vel_world"], S["quat"])


def force_derived_labels(rec, points: str = "raw") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-step (J (N,3), L_world (N,3), j_slot (N,K,3)) integrated from the recorded per-contact forces.

    points='raw': lever arm from Chrono's raw (envelope-inflated) box contact point  -> exact torque.
    points='canonical': lever arm from the canonical clamped point c_p_box_world     -> ~1 % lever-arm error.
    """
    C, S = rec.canon, rec.states
    N = rec.n_steps
    dt = float(rec.meta["dt"])
    active = np.asarray(C["c_active"][:N], dtype=np.float64)                       # (N,K)
    j_slot = np.asarray(C["c_force_world"][:N], dtype=np.float64) * dt * active[..., None]
    if points == "raw":
        idx = np.asarray(C["c_raw_index"][:N])
        safe = np.clip(idx, 0, None)
        pA = np.take_along_axis(np.asarray(rec.raw["raw_pA"][:N], dtype=np.float64), safe[..., None], 1)
        pB = np.take_along_axis(np.asarray(rec.raw["raw_pB"][:N], dtype=np.float64), safe[..., None], 1)
        a_is_box = np.take_along_axis(np.asarray(rec.raw["raw_A_is_box"][:N]), safe, 1)
        p = np.where(a_is_box[..., None], pA, pB)
    elif points == "canonical":
        p = np.asarray(C["c_p_box_world"][:N], dtype=np.float64)
    else:
        raise ValueError(points)
    r = (p - np.asarray(S["pos"][:N], dtype=np.float64)[:, None, :]) * active[..., None]
    J = j_slot.sum(1)
    L = np.cross(r, j_slot).sum(1)
    return J, L, j_slot


def impulse_targets_from_record(rec) -> dict[str, np.ndarray]:
    """Mass-normalized learning labels (float32) from an EpisodeRecord:
    target_dv_contact (N,3), target_dL_contact (N,3) [= R L_b / m], target_dw_contact (N,3), target_j_slot (N,K,3) [= F dt / m]."""
    S = rec.states
    ph = phys_from_meta(rec.meta)
    m, I, dt, g = ph["mass"], ph["inertia_diag"], ph["dt"], ph["gravity"]
    N = rec.n_steps
    q = np.asarray(S["quat"], dtype=np.float64)[:N]
    dv_c = linear_contact_dv(S["lin_vel"][: N + 1], g, dt)
    L_b = angular_contact_impulse_body(_box_local_ang_vel(rec)[: N + 1], I, dt)
    _, _, j_slot = force_derived_labels(rec, "raw")
    return {
        "target_dv_contact": dv_c.astype(np.float32),
        "target_dL_contact": (body_to_world(L_b, q) / m).astype(np.float32),
        "target_dw_contact": contact_dw_world(L_b, q, I).astype(np.float32),
        "target_j_slot": (j_slot / m).astype(np.float32),
    }


def state_derived_wrench(rec) -> tuple[np.ndarray, np.ndarray]:
    """Un-normalized (J (N,3), L_world (N,3)) from consecutive states (audit reference)."""
    S = rec.states
    ph = phys_from_meta(rec.meta)
    N = rec.n_steps
    q = np.asarray(S["quat"], dtype=np.float64)[:N]
    J = ph["mass"] * linear_contact_dv(S["lin_vel"][: N + 1], ph["gravity"], ph["dt"])
    L_b = angular_contact_impulse_body(_box_local_ang_vel(rec)[: N + 1], ph["inertia_diag"], ph["dt"])
    return J, body_to_world(L_b, q)


__all__ = [
    "LABEL_VERSION", "PHYS_KEYS", "K_SLOTS", "gravity_vec", "linear_contact_dv", "gyro_torque_body", "angular_contact_impulse_body",
    "gyro_dw_body", "gyro_dw_world", "body_to_world", "world_to_body", "contact_dw_world", "naive_dH_world", "box_inertia_diag_over_m",
    "phys_from_meta", "phys_arrays", "force_derived_labels", "impulse_targets_from_record", "state_derived_wrench",
]
