"""Contact impulse decoder (stage ED2): physics helpers, masking, cone constraint, permutation equivariance."""
import numpy as np
import torch

from recom.config import K_SLOTS
from recom.data.impulse_targets import angular_contact_impulse_body, gyro_dw_world
from recom.geometry.transforms import random_quat_uniform_np
from recom.models.impulse_decoder import (ContactImpulseDecoder, aggregate_wrench, body_angular_impulse_torch, cone_violation, gyro_delta_omega_world, lever_arms,
                                          tangent_basis, wrench_to_delta)

torch.manual_seed(0)


def _inputs(B=3, T=2, seed=0):
    rng = np.random.default_rng(seed)
    q = torch.tensor(random_quat_uniform_np(rng, B * T), dtype=torch.float32).reshape(B, T, 4)
    state = torch.cat([torch.randn(B, T, 3), q, torch.randn(B, T, 3), torch.randn(B, T, 3)], -1)
    he = torch.tensor([[0.1, 0.075, 0.05]] * B)
    n = torch.nn.functional.normalize(torch.randn(B, T, K_SLOTS, 3) + torch.tensor([0.0, 0, 3]), dim=-1)
    contacts = {"active": (torch.rand(B, T, K_SLOTS) > 0.4).float(), "d": torch.randn(B, T, K_SLOTS) * 1e-3, "n": n,
                "p_box_local": (torch.rand(B, T, K_SLOTS, 3) * 2 - 1) * he[:, None, None, :], "slot_embedding": torch.randn(B, T, K_SLOTS, 16)}
    contacts["active"][0, 0] = 0.0                   # one all-inactive frame
    contacts["active"][1, 1] = 1.0
    I_over_m = torch.tensor([[0.075 ** 2 + 0.05 ** 2, 0.1 ** 2 + 0.05 ** 2, 0.1 ** 2 + 0.075 ** 2]] * B) / 3
    phys = {"mass": torch.full((B,), 3.0), "log_mass": torch.full((B,), np.log(3.0)), "inertia_diag": 3.0 * I_over_m, "inertia_diag_over_m": I_over_m,
            "mu": torch.full((B,), 0.5), "restitution": torch.full((B,), 0.3), "dt": torch.full((B,), 1e-3), "gravity": torch.full((B,), 9.81), "envelope": torch.full((B,), 1e-3)}
    return contacts, state, he, phys


def test_tangent_basis_is_orthonormal_right_handed():
    n = torch.nn.functional.normalize(torch.randn(50, 3), dim=-1)
    t1, t2 = tangent_basis(n)
    for a, b in ((n, t1), (n, t2), (t1, t2)):
        assert torch.allclose((a * b).sum(-1), torch.zeros(50), atol=1e-6)
    assert torch.allclose(t1.norm(dim=-1), torch.ones(50), atol=1e-6)
    assert torch.allclose(torch.cross(t1, t2, dim=-1), n, atol=1e-6)


def test_physics_helpers_match_numpy_labels():
    rng = np.random.default_rng(1)
    I = np.array([1.0, 2.0, 3.0])
    w = np.zeros((6, 3))
    w[0] = [1.0, 2.0, 0.5]
    dt = 1e-3
    for k in range(5):
        w[k + 1] = w[k] - dt * np.cross(w[k], I * w[k]) / I
    q = random_quat_uniform_np(rng, 5)
    from recom.geometry.transforms import quat_to_rotmat_np
    R = quat_to_rotmat_np(q)
    ww = np.einsum("nij,nj->ni", R, w[:-1])
    ww1 = np.einsum("nij,nj->ni", R, w[1:])          # w_b[k+1] expressed with R[k] (rotation about omega leaves omega fixed)
    L = body_angular_impulse_torch(torch.tensor(ww), torch.tensor(ww1), torch.tensor(q), torch.tensor(I), dt)
    assert torch.allclose(L, torch.tensor(angular_contact_impulse_body(w, I, dt)), atol=1e-12)
    assert L.abs().max() < 1e-12
    g = gyro_delta_omega_world(torch.tensor(q), torch.tensor(ww), torch.tensor(I), dt)
    assert np.allclose(g.numpy(), gyro_dw_world(q, ww, I, dt), atol=1e-14)
    # aggregation vs manual sums
    j, r, a = torch.randn(4, K_SLOTS, 3), torch.randn(4, K_SLOTS, 3), (torch.rand(4, K_SLOTS) > 0.5).float()
    dv, dL = aggregate_wrench(j, r, a)
    assert torch.allclose(dv, (j * a[..., None]).sum(1)) and torch.allclose(dL, (torch.cross(r, j, dim=-1) * a[..., None]).sum(1))
    # wrench_to_delta: dw = R I^-1 R^T dL
    qq = torch.tensor(q[:4], dtype=torch.float32)
    Rt = torch.tensor(R[:4], dtype=torch.float32)
    d6 = wrench_to_delta(dv, dL, qq, torch.tensor(I, dtype=torch.float32).expand(4, 3))
    ref = torch.einsum("nij,nj->ni", Rt, torch.einsum("nji,nj->ni", Rt, dL) / torch.tensor(I, dtype=torch.float32))
    assert torch.allclose(d6[:, 3:], ref, atol=1e-5) and torch.allclose(d6[:, :3], dv)


def test_decoder_masking_cone_and_equivariance():
    contacts, state, he, phys = _inputs()
    dec = ContactImpulseDecoder(slot_embed_dim=16, width=32, n_blocks=2, n_heads=4, head_mode="cone").eval()
    with torch.no_grad():
        out = dec(contacts, state, he, phys)
    B, T = state.shape[:2]
    assert out["j_slot"].shape == (B, T, K_SLOTS, 3) and out["wrench"].shape == (B, T, 6) and out["dL"].shape == (B, T, 3)
    inactive = contacts["active"] == 0
    assert torch.equal(out["j_slot"][inactive], torch.zeros_like(out["j_slot"][inactive]))
    assert torch.equal(out["wrench"][0, 0], torch.zeros(6))                       # all-inactive frame -> exactly zero wrench
    assert cone_violation(out["j_slot"], contacts["n"], phys["mu"][:, None].expand(B, T), contacts["active"]).sum() == 0
    # slot permutation: per-slot outputs permute, the wrench is invariant; inactive-slot content must not matter
    perm = torch.tensor([2, 0, 3, 1])
    cp = {k: v[:, :, perm] for k, v in contacts.items()}
    cp["slot_embedding"] = cp["slot_embedding"].clone()
    cp["slot_embedding"][cp["active"] == 0] = 7.0                              # garbage in inactive slots
    with torch.no_grad():
        outp = dec(cp, state, he, phys)
    assert torch.allclose(outp["j_slot"], out["j_slot"][:, :, perm], atol=1e-5)
    assert torch.allclose(outp["wrench"], out["wrench"], atol=1e-5)
    # (B, K) inputs (no time axis) and the null-embedding path (no slot_embedding key)
    c2 = {k: v[:, 0] for k, v in contacts.items() if k != "slot_embedding"}
    with torch.no_grad():
        o2 = dec(c2, state[:, 0], he, phys)
    assert o2["wrench"].shape == (B, 6)
    # lever arms: R p
    r = lever_arms(state[..., 3:7], contacts["p_box_local"])
    assert torch.allclose(r.norm(dim=-1), contacts["p_box_local"].norm(dim=-1), atol=1e-5)
    # free head and pooled-only variants run
    for kw in ({"head_mode": "free"}, {"pooled_only": True}, {"use_slot_embedding": False}, {"timing_feats": False, "scaled_head": False}):
        d = ContactImpulseDecoder(slot_embed_dim=16, width=32, **kw).eval()
        with torch.no_grad():
            o = d(contacts, state, he, phys)
        assert o["wrench"].shape == (B, T, 6) and torch.equal(o["wrench"][0, 0], torch.zeros(6))


def test_yaw_augmentation_is_an_exact_symmetry_of_the_labels():
    """Rotating the sample by a yaw and re-deriving the wrench from the rotated states gives the rotated labels."""
    from recom.train.train_impulse import yaw_augment
    from recom.models.impulse_decoder import single_contact_scale
    from recom.models.nedm_adapter import slot_features
    contacts, state, he, phys = _inputs(B=4, T=3, seed=3)
    B, T = state.shape[:2]
    batch = {"states": state, "target_dv_contact": torch.randn(B, T, 3), "target_dL_contact": torch.randn(B, T, 3), "target_j_slot": torch.randn(B, T, K_SLOTS, 3), "c_n": contacts["n"]}
    out = yaw_augment(batch, torch.Generator().manual_seed(0))
    # norms and z-components are preserved; quaternion stays unit and canonical; lever arms rotate with the frame
    for k in ("target_dv_contact", "target_dL_contact"):
        assert torch.allclose(out[k].norm(dim=-1), batch[k].norm(dim=-1), atol=1e-5) and torch.allclose(out[k][..., 2], batch[k][..., 2], atol=1e-5)
    assert torch.allclose(out["states"][..., 3:7].norm(dim=-1), torch.ones(B, T), atol=1e-5) and (out["states"][..., 3] >= 0).all()
    assert torch.allclose(out["states"][..., 2], state[..., 2]) and torch.allclose(out["states"][..., 9], state[..., 9])
    geom = {k: contacts[k] for k in ("active", "d", "n", "p_box_local")}
    heT = he[:, None, :].expand(B, T, 3)
    r0 = slot_features(geom, state, heT)[..., 8:11]
    r1 = slot_features({**geom, "n": out["c_n"]}, out["states"], heT)[..., 8:11]
    assert torch.allclose(r1.norm(dim=-1), r0.norm(dim=-1), atol=1e-5) and torch.allclose(r1[..., 2], r0[..., 2], atol=1e-5)
    # Delassus scale: single contact at the COM (r = 0) reduces to (1+e)|v_n| for a fast approach, g dt for a separating one
    f = slot_features(geom, state, heT).clone()
    f[..., 8:11] = 0.0
    f[..., 14] = -2.0
    sc = single_contact_scale(f, state[..., 3:7], phys["inertia_diag_over_m"][:, None].expand(B, T, 3), phys["restitution"][:, None].expand(B, T), torch.full((B, T, 1), 9.81e-3))
    assert torch.allclose(sc, torch.full_like(sc, 1.3 * 2.0 + 9.81e-3), atol=1e-5)
    f[..., 14] = 0.5
    sc = single_contact_scale(f, state[..., 3:7], phys["inertia_diag_over_m"][:, None].expand(B, T, 3), phys["restitution"][:, None].expand(B, T), torch.full((B, T, 1), 9.81e-3))
    assert torch.allclose(sc, torch.full_like(sc, 9.81e-3), atol=1e-6)


def test_single_contact_solver_stick_slip_and_regimes():
    """Closed-form frictional single-contact impulse: stick when inside the cone, Coulomb slip otherwise; Chrono regimes."""
    from recom.models.impulse_decoder import single_contact_solver, skew, inertia_world_inv
    from recom.models.nedm_adapter import slot_features
    B, T, K = 1, 1, K_SLOTS
    he = torch.tensor([[0.1, 0.075, 0.05]])
    I_over_m = torch.tensor([[0.075 ** 2 + 0.05 ** 2, 0.1 ** 2 + 0.05 ** 2, 0.1 ** 2 + 0.075 ** 2]]) / 3
    q = torch.tensor([[[1.0, 0, 0, 0]]])
    def run(v, w, vt_big, d):
        state = torch.cat([torch.tensor([[[0.0, 0, 0.05]]]), q, torch.tensor([[v]]), torch.tensor([[w]])], -1)
        c = {"active": torch.zeros(B, T, K), "d": torch.full((B, T, K), d), "n": torch.tensor([0.0, 0, 1]).expand(B, T, K, 3).clone(), "p_box_local": torch.zeros(B, T, K, 3)}
        c["active"][0, 0, 0] = 1.0
        c["p_box_local"][0, 0, 0] = torch.tensor([0.1, 0.075, -0.05])            # a corner
        f = slot_features(c, state, he[:, None, :].expand(B, T, 3))
        j, sf = single_contact_solver(f, state, I_over_m[:, None].expand(B, T, 3), torch.full((B, T), 0.5), torch.full((B, T), 0.3), torch.full((B, T), 9.81e-3), torch.full((B, T, K), d), torch.full((B, T), 1e-3))
        # verify the post-impact contact-point velocity against the targets using the Delassus matrix
        r = f[0, 0, 0, 8:11]
        G = torch.eye(3) - skew(r) @ inertia_world_inv(q[0, 0], I_over_m[0]) @ skew(r)
        v_f = f[0, 0, 0, 11:14] + torch.tensor([0.0, 0, 9.81e-3])
        v_after = v_f + G @ j[0, 0, 0]
        return j[0, 0, 0], sf[0, 0, 0], v_after, v_f
    # penetrating, fast approach -> restitution target on the normal velocity; stick (v'_t = 0) or, if the stick impulse
    # leaves the cone (a corner impact induces rotation and hence tangential contact velocity), Coulomb slip
    j, sf, va, vf = run([0.0, 0, -3.0], [0.0, 0, 0], False, -1e-3)
    assert torch.allclose(va[2], torch.tensor(0.3 * (-vf[2])), atol=1e-4) and j[2] > 0
    assert (sf[3] == 1.0 and va[:2].norm() < 1e-5) or (sf[3] == 0.0 and j[:2].norm() <= 0.5 * j[2] + 1e-6)
    # penetrating with strong sliding -> Coulomb slip: |j_t| = mu j_n against the sliding direction, normal target met
    j, sf, va, vf = run([4.0, 0, -3.0], [0.0, 0, 0], True, -1e-3)
    assert sf[3] == 0.0 and torch.allclose(j[:2].norm(), 0.5 * j[2], atol=1e-6) and j[0] < 0
    assert torch.allclose(va[2], torch.tensor(0.3 * (-vf[2])), atol=1e-4)
    # positive gap closing this step -> stabilization target v'_n = -d/dt
    j, sf, va, vf = run([0.0, 0, -3.0], [0.0, 0, 0], False, 1.5e-3)
    assert torch.allclose(va[2], torch.tensor(-1.5), atol=1e-5)
    # positive gap that does not close this step -> no impulse
    j, sf, va, vf = run([0.0, 0, -1.0], [0.0, 0, 0], False, 2.0e-3)
    assert j.abs().max() == 0.0
    # decoder with the solver prior runs and keeps the masking / cone properties
    contacts, state, he2, phys = _inputs()
    dec = ContactImpulseDecoder(slot_embed_dim=16, width=32, head_scale="solver", use_slot_embedding=False).eval()
    with torch.no_grad():
        out = dec(contacts, state, he2, phys)
    assert torch.equal(out["wrench"][0, 0], torch.zeros(6)) and cone_violation(out["j_slot"], contacts["n"], phys["mu"][:, None].expand(3, 2), contacts["active"]).sum() == 0
