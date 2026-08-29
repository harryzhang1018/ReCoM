"""Transition model in the wrench modes (stage ED3): gyro prior exactness, free-flight exactness, physics residual."""
import numpy as np
import torch

from recom.config import K_SLOTS
from recom.data.impulse_targets import gyro_dw_world
from recom.geometry.transforms import random_quat_uniform_np
from recom.models.transition import BoxTransitionModel, box_inertia_diag_over_m

torch.manual_seed(0)
NORM = {"state_mean": np.zeros(13), "state_std": np.ones(13), "target_mean": np.zeros(6), "target_std": np.ones(6), "wrench_std": np.full(6, 0.5), "dL_std": np.full(3, 0.1)}


def _states(B=2, T=5, seed=0):
    rng = np.random.default_rng(seed)
    q = torch.tensor(random_quat_uniform_np(rng, B * T), dtype=torch.float32).reshape(B, T, 4)
    s = torch.cat([torch.randn(B, T, 3), q, torch.randn(B, T, 3), 3 * torch.randn(B, T, 3)], -1)
    he = torch.tensor([[0.1, 0.05, 0.02], [0.15, 0.15, 0.03]])
    return s, he


def _contacts(B, T, active_value=1.0):
    return {"active": torch.full((B, T, K_SLOTS), active_value), "d": torch.zeros(B, T, K_SLOTS), "n": torch.tensor([0.0, 0, 1]).expand(B, T, K_SLOTS, 3),
            "p_box_local": torch.zeros(B, T, K_SLOTS, 3), "wrench": torch.randn(B, T, 6)}


def test_gyro_prior_matches_numpy_and_is_exact_in_free_flight():
    s, he = _states()
    m = BoxTransitionModel(NORM, contact_mode="wrench", block_size=8, dt=1e-3, gyro_prior=True).eval()
    p = m.prior_delta(s, he)
    ref = gyro_dw_world(s[..., 3:7].double().numpy(), s[..., 10:13].double().numpy(), box_inertia_diag_over_m(he.double())[:, None].expand(2, 5, 3).numpy(), 1e-3)
    assert np.allclose(p[..., 3:].double().numpy(), ref, atol=1e-6)
    assert torch.allclose(p[..., :3], torch.tensor([0.0, 0, -9.81e-3]).expand(2, 5, 3))
    # no active slot and zero wrench -> delta == prior exactly (free flight incl. gyroscopic precession)
    c = _contacts(2, 5, 0.0)
    c["wrench"] = torch.zeros(2, 5, 6)
    with torch.no_grad():
        d = m.predict_delta(s, he, c)
    assert torch.equal(d, p)
    # norm / denorm round trip with the state-dependent prior
    t = torch.randn(2, 5, 6)
    assert torch.allclose(m.denorm_target(m.norm_target(t, s, he), s, he), t, atol=1e-6)


def test_physics_residual_applies_the_wrench_exactly():
    s, he = _states()
    for mode, dim in (("wrench", 6), ("wrench_lin", 3)):
        m = BoxTransitionModel(NORM, contact_mode=mode, block_size=8, dt=1e-3, gyro_prior=True, physics_residual=True).eval()
        assert m.inp.in_features == 13 + 3 + dim
        with torch.no_grad():
            m.head[-1].weight.zero_(); m.head[-1].bias.zero_()          # residual network outputs exactly zero
            c = _contacts(2, 5)
            d = m.predict_delta(s, he, c)
            expect = m.prior_delta(s, he) + (c["wrench"] if mode == "wrench" else torch.cat([c["wrench"][..., :3], torch.zeros(2, 5, 3)], -1))
        assert torch.allclose(d, expect, atol=1e-6)
        # gate: an inactive frame ignores the wrench
        c["active"][:, 0] = 0.0
        with torch.no_grad():
            d = m.predict_delta(s, he, c)
        assert torch.allclose(d[:, 0], m.prior_delta(s, he)[:, 0], atol=1e-6)


def test_wrench_context_is_normalized_and_baseline_paths_untouched():
    s, he = _states()
    m = BoxTransitionModel(NORM, contact_mode="wrench", block_size=8, dt=1e-3).eval()
    c = _contacts(2, 5)
    ctx = m.contact_context(c, s, he)
    assert torch.allclose(ctx, c["wrench"] / 0.5)
    m2 = BoxTransitionModel({k: v for k, v in NORM.items() if k not in ("wrench_std", "dL_std")}, contact_mode="explicit", block_size=8, dt=1e-3).eval()
    assert m2.inp.in_features == 13 + 3 + 64 and not m2.gyro_prior and not m2.physics_residual
    out = m2(s, he, {k: v for k, v in c.items() if k != "wrench"})
    assert out.shape == (2, 5, 6)
    # checkpoint round trip keeps the wrench buffers
    sd = m.state_dict()
    m3 = BoxTransitionModel(NORM, contact_mode="wrench", block_size=8, dt=1e-3)
    m3.load_state_dict(sd)
    assert torch.equal(m3.wrench_std, m.wrench_std)
