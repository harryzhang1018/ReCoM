"""Deterministic Chrono box-drop scene, contact reporter and recorder (M0).

Scene: one dynamic rigid box (collision box shape) + one fixed ground box whose top face is z = 0.
Contact method NSC, Bullet collision, fixed solver/timestepper/envelope (PhysicsConfig).
"""
from __future__ import annotations

import time
from typing import Any

import numpy as np
import pychrono as chrono

from ..config import RAW_MAX, SCHEMA_VERSION, EpisodeConfig
from ..data.schema import EpisodeRecord, empty_raw, empty_states
from ..geometry.box_mesh import box_mesh, mesh_hash
from ..geometry.box_plane_analytic import corner_gaps
from ..geometry.transforms import quat_canonical_np
from .sampling import release_position

TAG_BOX, TAG_GROUND = 1, 2

_SOLVERS = {
    "APGD": chrono.ChSolver.Type_APGD,
    "PSOR": chrono.ChSolver.Type_PSOR,
    "BARZILAIBORWEIN": chrono.ChSolver.Type_BARZILAIBORWEIN,
    "ADMM": chrono.ChSolver.Type_ADMM,
}
_TIMESTEPPERS = {
    "EULER_IMPLICIT_LINEARIZED": chrono.ChTimestepper.Type_EULER_IMPLICIT_LINEARIZED,
    "EULER_IMPLICIT_PROJECTED": chrono.ChTimestepper.Type_EULER_IMPLICIT_PROJECTED,
    "EULER_IMPLICIT": chrono.ChTimestepper.Type_EULER_IMPLICIT,
}


def chrono_version() -> str:
    try:
        import importlib.metadata as md
        return md.version("pychrono")
    except Exception:  # pragma: no cover
        return "unknown"


class ContactReporter(chrono.ReportContactCallback):
    """Collects every raw Chrono contact record of the current frame (no canonicalization)."""

    def __init__(self) -> None:
        super().__init__()
        self.rows: list[tuple] = []

    def reset(self) -> None:
        self.rows = []

    def OnReportContact(self, pA, pB, plane_coord, distance, eff_radius, react_forces, react_torques, objA, objB, constraint_offset):  # noqa: N802
        tagA = objA.GetPhysicsItem().GetTag()
        tagB = objB.GetPhysicsItem().GetTag()
        m = plane_coord
        plane = ((m.GetAxisX().x, m.GetAxisX().y, m.GetAxisX().z),
                 (m.GetAxisY().x, m.GetAxisY().y, m.GetAxisY().z),
                 (m.GetAxisZ().x, m.GetAxisZ().y, m.GetAxisZ().z))  # columns = axes
        self.rows.append((
            (pA.x, pA.y, pA.z), (pB.x, pB.y, pB.z), plane, float(distance), float(eff_radius),
            (react_forces.x, react_forces.y, react_forces.z), (react_torques.x, react_torques.y, react_torques.z),
            tagA, tagB, int(constraint_offset),
        ))
        return True


def inertia_box(mass: float, he: np.ndarray) -> np.ndarray:
    """Diagonal inertia of a solid box with half-extents he."""
    s = 2.0 * np.asarray(he)
    return mass / 12.0 * np.array([s[1] ** 2 + s[2] ** 2, s[0] ** 2 + s[2] ** 2, s[0] ** 2 + s[1] ** 2])


class BoxDropScene:
    """Build the Chrono system for one episode configuration."""

    def __init__(self, ep: EpisodeConfig) -> None:
        self.ep = ep
        ph = ep.physics
        he = np.asarray(ep.box.half_extents, dtype=np.float64)
        self.half_extents = he

        sys = chrono.ChSystemNSC()
        sys.SetNumThreads(ph.num_threads)
        if ph.collision_system != "BULLET":
            raise ValueError("only BULLET collision is configured for Study 1")
        sys.SetCollisionSystemType(chrono.ChCollisionSystem.Type_BULLET)
        # NOTE: SetCollisionSystemType resets the static defaults to Chrono's 30 mm / 10 mm, so the
        # defaults must be set AFTER it; every model is also pinned explicitly below.
        chrono.ChCollisionModel.SetDefaultSuggestedEnvelope(ph.collision_envelope)
        chrono.ChCollisionModel.SetDefaultSuggestedMargin(ph.collision_margin)
        sys.SetGravitationalAcceleration(chrono.ChVector3d(0, 0, -ph.gravity))
        sys.SetSolverType(_SOLVERS[ph.solver])
        it = sys.GetSolver().AsIterative()
        it.SetMaxIterations(ph.solver_max_iterations)
        it.SetTolerance(ph.solver_tolerance)
        sys.SetTimestepperType(_TIMESTEPPERS[ph.timestepper])
        sys.SetMinBounceSpeed(ph.min_bounce_speed)
        sys.SetSleepingAllowed(ph.allow_sleeping)

        mat = chrono.ChContactMaterialNSC()
        mat.SetFriction(ph.friction)
        mat.SetRestitution(ph.restitution)
        mat.SetRollingFriction(ph.rolling_friction)
        mat.SetSpinningFriction(ph.spinning_friction)
        self.material = mat

        gx, gy, gz = ph.ground_size
        ground = chrono.ChBody()
        ground.SetName("ground")
        ground.SetTag(TAG_GROUND)
        ground.SetFixed(True)
        ground.SetMass(1.0)
        ground.AddCollisionShape(chrono.ChCollisionShapeBox(mat, gx, gy, gz), chrono.ChFramed(chrono.ChVector3d(0, 0, -gz / 2)))
        ground.EnableCollision(True)
        ground.GetCollisionModel().SetEnvelope(ph.collision_envelope)
        ground.GetCollisionModel().SetSafeMargin(ph.collision_margin)
        sys.Add(ground)

        volume = float(np.prod(2 * he))
        mass = ph.density * volume
        box = chrono.ChBody()
        box.SetName("box")
        box.SetTag(TAG_BOX)
        box.SetMass(mass)
        I = inertia_box(mass, he)
        box.SetInertiaXX(chrono.ChVector3d(*I))
        box.AddCollisionShape(chrono.ChCollisionShapeBox(mat, 2 * he[0], 2 * he[1], 2 * he[2]))
        box.EnableCollision(True)
        box.GetCollisionModel().SetEnvelope(ph.collision_envelope)
        box.GetCollisionModel().SetSafeMargin(ph.collision_margin)
        box.SetSleepingAllowed(ph.allow_sleeping)
        sys.Add(box)
        for body in (ground, box):
            cm = body.GetCollisionModel()
            assert abs(cm.GetEnvelope() - ph.collision_envelope) < 1e-6 and abs(cm.GetSafeMargin() - ph.collision_margin) < 1e-6, "collision envelope/margin not applied"

        self.system, self.ground, self.box = sys, ground, box
        self.mass, self.inertia = mass, I
        self.reporter = ContactReporter()
        self.reset_box()

    def reset_box(self) -> None:
        ep = self.ep
        pos = release_position(ep)
        q = np.asarray(ep.quat)
        self.box.SetPos(chrono.ChVector3d(*pos))
        self.box.SetRot(chrono.ChQuaterniond(*q))
        self.box.SetPosDt(chrono.ChVector3d(*ep.lin_vel0))
        self.box.SetAngVelParent(chrono.ChVector3d(*ep.ang_vel0))
        self.system.Setup()
        self.system.Update()

    # -- state access ------------------------------------------------------------------
    def box_state(self) -> dict[str, np.ndarray]:
        b = self.box
        p, q, v = b.GetPos(), b.GetRot(), b.GetPosDt()
        ww, wl = b.GetAngVelParent(), b.GetAngVelLocal()
        return {
            "pos": np.array([p.x, p.y, p.z]),
            "quat": quat_canonical_np(np.array([q.e0, q.e1, q.e2, q.e3])),
            "lin_vel": np.array([v.x, v.y, v.z]),
            "ang_vel_world": np.array([ww.x, ww.y, ww.z]),
            "ang_vel_local": np.array([wl.x, wl.y, wl.z]),
        }

    def step(self) -> list[tuple]:
        """Advance one physics step; return the raw contacts used by that step."""
        self.system.DoStepDynamics(self.ep.physics.dt)
        self.reporter.reset()
        self.system.GetContactContainer().ReportAllContacts(self.reporter)
        return self.reporter.rows

    def metadata(self) -> dict[str, Any]:
        ep = self.ep
        ph = ep.physics
        V, F = box_mesh(self.half_extents)
        return {
            "schema_version": SCHEMA_VERSION,
            "episode": ep.to_dict(),
            "chrono_version": chrono_version(),
            "contact_method": ph.contact_method,
            "collision_system": ph.collision_system,
            "solver": ph.solver,
            "solver_max_iterations": ph.solver_max_iterations,
            "solver_tolerance": ph.solver_tolerance,
            "timestepper": ph.timestepper,
            "dt": ph.dt,
            "collision_envelope": ph.collision_envelope,
            "collision_margin": ph.collision_margin,
            "collision_envelope_actual": float(self.box.GetCollisionModel().GetEnvelope()),
            "collision_margin_actual": float(self.box.GetCollisionModel().GetSafeMargin()),
            "min_bounce_speed": ph.min_bounce_speed,
            "friction": ph.friction,
            "restitution": ph.restitution,
            "density": ph.density,
            "mass": self.mass,
            "inertia_xx": self.inertia.tolist(),
            "half_extents": self.half_extents.tolist(),
            "geometry_id": ep.box.geometry_id,
            "geometry_group": ep.box.geometry_group,
            "collision_mesh_hash": mesh_hash(V, F),
            "collision_mesh_num_faces": int(F.shape[0]),
            "initial_pos": release_position(ep).tolist(),
            "initial_quat": list(ep.quat),
            "initial_lin_vel": list(ep.lin_vel0),
            "initial_ang_vel_world": list(ep.ang_vel0),
            "quaternion_order": "wxyz_canonical_w_nonneg",
            "world_up": "+Z",
            "ground_plane": {"origin": [0, 0, 0], "normal": [0, 0, 1]},
            "time_alignment": "states[k] pre-step; contacts[k] from collision pass at start of step k (from states[k]); states[k+1] post-step",
            "raw_normal_convention": "raw plane X axis points from contactable A to contactable B (Chrono)",
            "canonical_normal_convention": "ground -> box; d > 0 separated, d < 0 penetrating",
        }


def _fill_raw(raw: dict[str, np.ndarray], k: int, rows: list[tuple]) -> None:
    raw["raw_count"][k] = len(rows)
    for j, r in enumerate(rows[:RAW_MAX]):
        pA, pB, plane, d, er, F, T, tagA, tagB, coff = r
        raw["raw_pA"][k, j] = pA
        raw["raw_pB"][k, j] = pB
        raw["raw_plane"][k, j] = np.asarray(plane).T  # store as matrix with columns = axes
        raw["raw_distance"][k, j] = d
        raw["raw_eff_radius"][k, j] = er
        raw["raw_force"][k, j] = F
        raw["raw_torque"][k, j] = T
        raw["raw_A_is_box"][k, j] = tagA == TAG_BOX
        raw["raw_constraint_offset"][k, j] = coff


def run_episode(ep: EpisodeConfig, record_every: int = 1, canonicalize: bool = True) -> EpisodeRecord:
    """Simulate one episode and return the full record (states, raw contacts, canonical view).

    Settling condition (documented): over the last `settle_window_steps` steps, every step had
    >= settle_min_contacts Chrono contacts, every box corner moved less than settle_pos_tol
    relative to its position at the start of the window, and the mean |v| over the window is
    below settle_mean_speed_tol.  The episode terminates when the condition first holds; the
    `settled` flag marks all states inside that window.  (A box at rest on 4 NSC contacts has a
    deterministic micro-rocking limit cycle; see PhysicsConfig.)
    """
    assert record_every == 1, "Study 1 records every physics step"
    ph = ep.physics
    scene = BoxDropScene(ep)
    he = scene.half_extents
    n_max = int(round(ph.max_duration / ph.dt))
    states = empty_states(n_max + 1)
    raw = empty_raw(n_max)
    W = ph.settle_window_steps

    t0 = time.perf_counter()
    n_done = 0
    termination = "max_duration"
    contact_ok = np.zeros(n_max + 1, dtype=bool)
    corners = np.zeros((n_max + 1, 8, 3))
    for k in range(n_max):
        s = scene.box_state()
        for key, val in s.items():
            states[key][k] = val
        states["t"][k] = k * ph.dt
        corners[k] = corner_gaps(s["pos"], s["quat"], he)[1]
        rows = scene.step()
        _fill_raw(raw, k, rows)
        n_done = k + 1
        contact_ok[k] = len(rows) >= ph.settle_min_contacts
        s1 = scene.box_state()
        corners[k + 1] = corner_gaps(s1["pos"], s1["quat"], he)[1]
        if k + 1 >= W and contact_ok[k + 1 - W: k + 1].all():
            disp = np.abs(corners[k + 2 - W: k + 2] - corners[k + 1 - W]).max()
            mean_speed = np.linalg.norm(states["lin_vel"][k + 1 - W: k + 1], axis=1).mean()
            if disp < ph.settle_pos_tol and mean_speed < ph.settle_mean_speed_tol:
                termination = "settled"
                break
    s = scene.box_state()
    for key, val in s.items():
        states[key][n_done] = val
    states["t"][n_done] = n_done * ph.dt
    if termination == "settled":
        states["settled"][n_done - W + 1: n_done + 1] = True
    wall = time.perf_counter() - t0

    states = {k: v[: n_done + 1] for k, v in states.items()}
    raw = {k: v[:n_done] for k, v in raw.items()}
    meta = scene.metadata()
    meta.update({
        "n_steps": n_done,
        "planned_steps": n_max,
        "termination_reason": termination,
        "wall_time_s": wall,
        "max_raw_contacts_per_frame": int(raw["raw_count"].max()) if n_done else 0,
        "raw_overflow_frames": int(np.sum(raw["raw_count"] > RAW_MAX)),
    })
    rec = EpisodeRecord(meta=meta, states=states, raw=raw)
    if canonicalize:
        from ..data.canonicalize import canonicalize_episode
        canonicalize_episode(rec)
    return rec
