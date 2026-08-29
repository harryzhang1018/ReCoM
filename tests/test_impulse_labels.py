"""Frozen impulse-label formulas (stage ED0): synthetic Chrono-like updates and recorded smoke data."""
from pathlib import Path

import numpy as np
import pytest

from recom.data.impulse_targets import (angular_contact_impulse_body, box_inertia_diag_over_m, contact_dw_world, force_derived_labels, gyro_dw_body, gyro_dw_world,
                                        impulse_targets_from_record, linear_contact_dv, phys_from_meta, state_derived_wrench)
from recom.geometry.transforms import quat_to_rotmat_np, random_quat_uniform_np

DT = 1e-3


def _tumble(I, w0, n):
    w = np.zeros((n + 1, 3))
    w[0] = w0
    for k in range(n):
        w[k + 1] = w[k] + gyro_dw_body(w[k], I, DT)   # Chrono's torque-free discrete update
    return w


def test_torque_free_tumbling_gives_zero_angular_impulse():
    I = np.array([1.0, 2.0, 3.0])
    w = _tumble(I, np.array([1.0, 2.0, 0.5]), 500)
    L = angular_contact_impulse_body(w, I, DT)
    assert np.abs(L).max() < 1e-12
    # the naive I*dw (no gyro term) is not zero for a tumbling non-spherical body
    assert np.abs(I * (w[1:] - w[:-1])).max() > 1e-6


def test_ballistic_free_flight_gives_zero_linear_impulse():
    g = 9.81
    v = np.zeros((100, 3))
    for k in range(99):
        v[k + 1] = v[k] + np.array([0, 0, -g]) * DT
    assert np.abs(linear_contact_dv(v, g, DT)).max() < 1e-14


def test_gyro_prior_world_frame_consistency():
    rng = np.random.default_rng(0)
    q = random_quat_uniform_np(rng, 16)
    R = quat_to_rotmat_np(q)
    I_over_m = box_inertia_diag_over_m(np.array([0.1, 0.05, 0.02]))
    w_b = rng.normal(size=(16, 3))
    w_w = np.einsum("nij,nj->ni", R, w_b)
    dw_w = gyro_dw_world(q, w_w, I_over_m, DT)
    ref = np.einsum("nij,nj->ni", R, gyro_dw_body(w_b, I_over_m, DT))
    assert np.allclose(dw_w, ref, atol=1e-14)
    # scale invariance in the inertia (only ratios matter)
    assert np.allclose(gyro_dw_world(q, w_w, 7.0 * I_over_m, DT), dw_w, atol=1e-14)
    # contact dw = R I^-1 L_b
    L_b = rng.normal(size=(16, 3))
    assert np.allclose(contact_dw_world(L_b, q, I_over_m), np.einsum("nij,nj->ni", R, L_b / I_over_m))


def test_box_inertia_matches_recorder():
    from recom.sim.chrono_box_drop import inertia_box
    he = np.array([0.1, 0.075, 0.05])
    assert np.allclose(inertia_box(2.0, he) / 2.0, box_inertia_diag_over_m(he))


@pytest.mark.skipif(not Path("data/smoke/splits.json").exists(), reason="data/smoke not generated")
def test_recorded_smoke_episode_labels_agree():
    from recom.data.splits import load_splits
    from recom.data.storage import load_episode
    ids = load_splits("data/smoke/splits.json")["train"][:2]
    for eid in ids:
        rec = load_episode("data/smoke", eid)
        ph = phys_from_meta(rec.meta)
        J_s, L_s = state_derived_wrench(rec)
        J_r, L_r, _ = force_derived_labels(rec, "raw")
        nc = rec.canon["n_contacts"][: rec.n_steps]
        assert np.abs(J_r - J_s).max() / ph["mass"] < 1e-9              # linear momentum balance is exact
        sL = np.linalg.norm(L_s, axis=-1).max()
        assert np.abs(L_r - L_s).max() < 1e-6 * max(sL, 1e-9)            # raw-point torque balance is exact
        tg = impulse_targets_from_record(rec)
        assert np.linalg.norm(tg["target_dv_contact"][nc == 0], axis=-1).max() < 1e-9
        assert np.linalg.norm(tg["target_dw_contact"][nc == 0], axis=-1).max() < 1e-7
        assert tg["target_j_slot"].shape == (rec.n_steps, 4, 3)
        # mass normalization: sum of per-slot impulses equals dv_c
        assert np.allclose(tg["target_j_slot"].sum(1), tg["target_dv_contact"], atol=1e-6)


@pytest.mark.skipif(not Path("data/smoke/splits.json").exists(), reason="data/smoke not generated")
def test_chrono_gap_formula_reproduces_reported_distance():
    """c_d == point height - (sum_a |R[2,a]| - 1) * envelope (Bullet envelope inflation), the gap the NSC solver uses."""
    import torch
    from recom.data.splits import load_splits
    from recom.data.storage import load_episode
    from recom.models.impulse_decoder import chrono_gap
    from recom.models.nedm_adapter import slot_features
    rec = load_episode("data/smoke", load_splits("data/smoke/splits.json")["train"][0])
    N = rec.n_steps
    S = rec.states
    st = torch.tensor(np.concatenate([S["pos"][:N], S["quat"][:N], S["lin_vel"][:N], S["ang_vel_world"][:N]], 1), dtype=torch.float64)
    c = {k: torch.tensor(rec.canon["c_" + k][:N], dtype=torch.float64) for k in ("active", "d", "n", "p_box_local")}
    he = torch.tensor(rec.meta["half_extents"], dtype=torch.float64).expand(N, 3)
    f = slot_features(c, st, he)
    d = chrono_gap(f, st, torch.full((N,), rec.meta["collision_envelope_actual"], dtype=torch.float64))[..., 0]
    m = c["active"] > 0
    assert (d - c["d"])[m].abs().median() < 1e-6 and (d - c["d"])[m].abs().quantile(0.9) < 5e-5
