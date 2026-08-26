"""Sign / normal / time-alignment conventions of the Chrono recorder on hand-built configurations (Section 9)."""
import dataclasses

import numpy as np
import pytest

pytest.importorskip("pychrono")

from recom.config import DatasetGenConfig, PhysicsConfig
from recom.data.validate import first_contact_timing_check, free_flight_check, replay_check, validate_episode
from recom.geometry.box_plane_analytic import min_gap
from recom.geometry.transforms import quat_from_axis_angle_np
from recom.sim.chrono_box_drop import BoxDropScene
from recom.sim.sampling import sample_episode

HE = np.array([0.1, 0.075, 0.05])


def _ep(quat=(1.0, 0, 0, 0), clearance=0.05, **ph):
    cfg = DatasetGenConfig(name="t", phase="1A", physics=PhysicsConfig(max_duration=0.5, **ph))
    return dataclasses.replace(sample_episode(cfg, 0), quat=tuple(map(float, quat)), clearance=clearance)


def _contacts_at_pose(ep, pos, quat):
    """Place the box at a pose, do one step, report the contacts computed from that pose."""
    import pychrono as chrono
    sc = BoxDropScene(ep)
    sc.box.SetPos(chrono.ChVector3d(*pos))
    sc.box.SetRot(chrono.ChQuaterniond(*quat))
    sc.system.Setup()
    sc.system.Update()
    return sc.step(), sc


def test_envelope_actually_applied():
    sc = BoxDropScene(_ep())
    assert sc.box.GetCollisionModel().GetEnvelope() == pytest.approx(1e-3, abs=1e-6)
    assert sc.ground.GetCollisionModel().GetSafeMargin() == pytest.approx(5e-4, abs=1e-6)


def test_above_touching_penetrating_signs():
    ep = _ep()
    rows, _ = _contacts_at_pose(ep, [0, 0, HE[2] + 0.05], (1, 0, 0, 0))
    assert len(rows) == 0
    rows, _ = _contacts_at_pose(ep, [0, 0, HE[2]], (1, 0, 0, 0))
    assert len(rows) == 4
    for pA, pB, plane, d, *_ in rows:
        assert abs(d) < 1e-6
    rows, _ = _contacts_at_pose(ep, [0, 0, HE[2] - 0.002], (1, 0, 0, 0))
    assert len(rows) == 4
    for pA, pB, plane, d, er, F, T, tagA, tagB, coff in rows:
        assert d == pytest.approx(-0.002, abs=1e-6)
        n = np.array(plane[0])  # X axis: A -> B
        # normal points from A toward B; canonical (ground -> box) must be +Z
        n_canon = -n if tagA == 1 else n
        assert np.allclose(n_canon, [0, 0, 1], atol=1e-9)


def test_reported_distance_matches_prestep_gap_time_alignment():
    """Contacts reported after DoStepDynamics come from the collision pass on the PRE-step pose."""
    ep = _ep(clearance=0.05)
    sc = BoxDropScene(ep)
    for _ in range(200):
        s = sc.box_state()
        gap_pre = float(min_gap(s["pos"], s["quat"], HE))
        rows = sc.step()
        if rows:
            d = np.array([r[3] for r in rows])
            assert np.allclose(d, gap_pre, atol=2e-5)
            return
    pytest.fail("no contact within 200 steps")


def test_rotated_corner_and_edge_report_counts():
    ep = _ep()
    q_edge = quat_from_axis_angle_np([1, 0, 0], np.pi / 4)
    z = -min_gap(np.zeros(3), q_edge, HE)
    rows, _ = _contacts_at_pose(ep, [0, 0, z], q_edge)
    assert 1 <= len(rows) <= 2
    q_corner = quat_from_axis_angle_np([1, -1, 0], np.arctan(np.sqrt(2)))
    z = -min_gap(np.zeros(3), q_corner, HE)
    rows, _ = _contacts_at_pose(ep, [0, 0, z], q_corner)
    assert 1 <= len(rows) <= 2
    # the reported box point is near the analytic lowest corner (within envelope inflation)
    from recom.geometry.box_plane_analytic import corner_gaps
    g, cw = corner_gaps(np.array([0, 0, z]), q_corner, HE)
    for pA, pB, plane, d, er, F, T, tagA, tagB, coff in rows:
        pbox = np.array(pA if tagA == 1 else pB)
        assert np.linalg.norm(pbox - cw[np.argmin(g)]) < 3e-3


def test_translation_invariance_of_chrono_contacts():
    ep = _ep()
    q = quat_from_axis_angle_np([0.3, 0.5, 0.8], 1.1)
    z = 0.0005 - min_gap(np.zeros(3), q, HE)
    r0, _ = _contacts_at_pose(ep, [0, 0, z], q)
    r1, _ = _contacts_at_pose(ep, [0.8, -0.4, z], q)
    assert len(r0) == len(r1)
    assert np.allclose(sorted(r[3] for r in r0), sorted(r[3] for r in r1), atol=1e-6)


def test_episode_validation_free_flight_timing_replay():
    from recom.sim.chrono_box_drop import run_episode
    rec = run_episode(_ep(quat=tuple(quat_from_axis_angle_np([0.2, 0.9, 0.1], 0.8)), clearance=0.2))
    assert validate_episode(rec) == []
    assert free_flight_check(rec)["pass"]
    assert first_contact_timing_check(rec)["pass"]
    rp = replay_check(rec)
    assert rp["pass"] and rp["max_abs_diff_pos"] == 0.0
