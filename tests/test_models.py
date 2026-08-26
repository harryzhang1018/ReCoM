"""Model tests (Section 18): permutation invariance, translation invariance, determinism, analytic contract."""
import numpy as np
import pytest
import torch

from recom.config import K_SLOTS
from recom.geometry.transforms import random_quat_uniform_np
from recom.models.analytic_baseline import AnalyticBoxPlaneEncoder
from recom.models.contact_encoder import PatchContactEncoder, PointContactEncoder
from recom.models.losses import contact_set_loss, hungarian_match
from recom.models.transition import BoxTransitionModel

torch.manual_seed(0)
HE = torch.tensor([[0.1, 0.075, 0.05]] * 4)


def _poses(n=4, seed=0):
    rng = np.random.default_rng(seed)
    q = torch.tensor(random_quat_uniform_np(rng, n), dtype=torch.float32)
    pos = torch.tensor(rng.uniform(-0.5, 0.5, (n, 3)), dtype=torch.float32)
    pos[:, 2] = torch.tensor(rng.uniform(0.05, 0.2, n), dtype=torch.float32)
    return pos, q


@pytest.mark.parametrize("cls", [PatchContactEncoder, PointContactEncoder])
def test_translation_invariance_and_determinism(cls):
    m = cls(d_model=32).eval()
    pos, q = _poses()
    with torch.no_grad():
        a = m(HE, pos, q)
        b = m(HE, pos + torch.tensor([1.0, -2.0, 0.0]), q)
        c = m(HE, pos, q)
    for k in ("logit", "p_box_local", "p_ground_rel", "n", "d"):
        assert torch.allclose(a[k], b[k], atol=1e-5), k
        assert torch.equal(a[k], c[k]), k


def test_patch_token_permutation_invariance():
    m = PatchContactEncoder(d_model=32).eval()
    pos, q = _poses()
    tok = m.canonical_tokens(HE)
    perm = torch.randperm(12)
    inv = torch.argsort(perm)
    tok_p = {k: v[:, perm] for k, v in tok.items()}
    tok_p["adj"] = inv[tok["adj"][:, perm]]  # remap neighbour indices
    with torch.no_grad():
        a, b = m(HE, pos, q, tok), m(HE, pos, q, tok_p)
    for k in ("logit", "p_box_local", "d"):
        assert torch.allclose(a[k], b[k], atol=1e-4), k


def test_point_token_permutation_invariance():
    m = PointContactEncoder(d_model=32, k_nn=8).eval()
    pos, q = _poses()
    tok = m.canonical_tokens(HE)
    perm = torch.randperm(tok["pos"].shape[1])
    tok_p = {k: v[:, perm] for k, v in tok.items()}
    with torch.no_grad():
        a, b = m(HE, pos, q, tok), m(HE, pos, q, tok_p)
    for k in ("logit", "p_box_local", "d"):
        assert torch.allclose(a[k], b[k], atol=1e-4), k


def test_analytic_encoder_matches_numpy_contract():
    from recom.geometry.box_plane_analytic import analytic_contacts
    enc = AnalyticBoxPlaneEncoder(margin=0.002)
    rng = np.random.default_rng(3)
    q = random_quat_uniform_np(rng, 8)
    he = np.array([0.1, 0.075, 0.05])
    from recom.geometry.box_plane_analytic import min_gap
    pos = np.array([[0.2, -0.1, 0.001 - min_gap(np.zeros(3), qi, he)] for qi in q])
    out = enc(torch.tensor(np.tile(he, (8, 1)), dtype=torch.float32), torch.tensor(pos, dtype=torch.float32), torch.tensor(q, dtype=torch.float32))
    for i in range(8):
        ref = analytic_contacts(pos[i], q[i], he, margin=0.002)
        assert int((out["logit"][i] > 0).sum()) == ref["n_active"]
        assert np.allclose(np.sort(out["d"][i, : ref["n_active"]].numpy()), np.sort(ref["d"][ref["active"]]), atol=1e-5)


def test_hungarian_matching_recovers_permutation():
    B, K = 3, K_SLOTS
    gt = {"active": torch.ones(B, K), "d": torch.rand(B, K) * 0.01, "p_box_local": torch.rand(B, K, 3) * 0.1, "p_ground_rel": torch.rand(B, K, 3) * 0.1, "n": torch.tensor([0.0, 0, 1]).expand(B, K, 3), "n_contacts": torch.full((B,), K)}
    perm = torch.stack([torch.randperm(K) for _ in range(B)])
    pred = {k: (torch.gather(v, 1, perm.view(B, K, *([1] * (v.dim() - 2))).expand_as(v)) if v.dim() >= 2 and v.shape[1] == K else v) for k, v in gt.items() if k != "n_contacts"}
    pred["logit"] = torch.full((B, K), 5.0)
    m = hungarian_match(pred, gt, torch.full((B,), 0.05))
    assert torch.equal(m, torch.argsort(perm, dim=1))   # perm[b, j] = predicted slot assigned to GT slot j
    loss, parts = contact_set_loss({**pred, "cardinality": torch.zeros(B, K + 1), "log_var": torch.zeros(B, K)}, gt, torch.full((B,), 0.05))
    assert parts["p_box"] < 1e-6 and parts["d"] < 1e-9


def test_transition_integration_matches_chrono_rule():
    norm = {"state_mean": np.zeros(13), "state_std": np.ones(13), "target_mean": np.zeros(6), "target_std": np.ones(6)}
    m = BoxTransitionModel(norm, contact_mode="none", block_size=8, dt=1e-3)
    s = torch.zeros(2, 13)
    s[:, 3] = 1.0
    s[:, 2] = 1.0
    d = torch.tensor([[0.0, 0, -0.00981, 0, 0, 0], [0.1, 0, 0, 0, 0, 2.0]])
    s1 = m.integrate(s, d)
    assert torch.allclose(s1[0, 2], torch.tensor(1.0 - 0.00981 * 1e-3))
    assert torch.allclose(s1[1, 0], torch.tensor(0.1 * 1e-3))
    assert torch.allclose(s1[1, 3:7].norm(), torch.tensor(1.0))
    assert torch.allclose(s1[1, 6], torch.tensor(np.sin(0.5 * 2.0 * 1e-3)).float())
    out = m(torch.zeros(2, 8, 13), torch.tensor([[0.1, 0.075, 0.05]] * 2))
    assert out.shape == (2, 8, 6)
