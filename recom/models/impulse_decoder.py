"""Contact impulse decoder + deterministic wrench aggregation (encoder-decoder plan, Sections 4.4-4.7).

    geometry encoder slots (active, d, n, p_box_local, slot_embedding)
        -> per-slot dynamics features (recom.models.nedm_adapter.slot_features: lever arm, relative velocity, ...)
        -> + physical parameters + broadcast global manifold context
        -> shared MLP -> self-attention across the K slots -> per-slot impulse head
        -> j_i / m  (mass-normalized impulse, world frame; exactly zero for inactive slots)
        -> dv_c = sum_i j_i/m ,  dL = sum_i r_i x j_i/m ,  dw_c = I_world^-1 (m dL)     (deterministic)

Units: everything is mass-normalized (velocity units), see recom.data.impulse_targets.  The friction cone
is scale invariant, so the constrained head applies unchanged:  j_n >= 0,  |j_t| <= mu j_n.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import K_SLOTS
from ..geometry.transforms import quat_to_rotmat
from .contact_encoder import mlp
from .nedm_adapter import SLOT_FEAT_DIM, ContactPooling, slot_features

PHYS_FEAT_DIM = 7
EXTRA_FEAT_DIM = 3     # v2 per-slot timing features (velocity units): h_i/dt, u_i = -v_n - h_i/dt, relu(-v_n)


def chrono_gap(feats: torch.Tensor, state: torch.Tensor, envelope: torch.Tensor) -> torch.Tensor:
    """Chrono's reported (NSC) contact distance from the predicted box point: the geometric height of the point above the
    plane h_i = pos_z + r_i,z minus Bullet's envelope-inflation term (sum_a |R[2,a]| - 1) * envelope.  Verified on pilot1b to
    reproduce c_d with a median residual of 0.0000 mm (p99 0.02 mm).  The NSC stabilization constraint v_n' >= -d/dt makes the
    first-impact impulse depend on this distance at sub-mm precision (the regressed d has ~0.5 mm error).  (..., K, 1)."""
    h = state[..., 2:3].unsqueeze(-2) + feats[..., 10:11]
    R = rotmat(state[..., 3:7])
    s = R[..., 2, :].abs().sum(-1)                                             # (...)
    return h - ((s - 1.0) * envelope).unsqueeze(-1).unsqueeze(-1)


def slot_timing_features(feats: torch.Tensor, state: torch.Tensor, dt: torch.Tensor, envelope: torch.Tensor | None = None) -> torch.Tensor:
    """Exact Chrono gap d_i (chrono_gap; the plain point height if envelope is None) in velocity units d_i/dt and the NSC
    velocity deficit u_i = -v_n - d_i/dt (an impulse fires iff the contact would close its gap this step), plus relu(-v_n).
    feats: slot_features output (..., K, 19) [r at 8:11, v_n at 14]; returns (..., K, EXTRA_FEAT_DIM), zero for inactive slots."""
    if envelope is None:
        h = state[..., 2:3].unsqueeze(-2) + feats[..., 10:11]                  # (..., K, 1)
    else:
        h = chrono_gap(feats, state, envelope)
    v_n = feats[..., 14:15]
    h_dt = h / dt.unsqueeze(-1).unsqueeze(-1)
    x = torch.cat([h_dt, -v_n - h_dt, torch.relu(-v_n)], -1)
    return x * feats[..., 0:1]


# ---- deterministic physics helpers ----------------------------------------------------------------
def tangent_basis(n: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Deterministic orthonormal tangent basis (t1, t2) for unit normals n (..., 3); (t1, t2, n) is right-handed."""
    e = F.one_hot(n.abs().argmin(-1), 3).to(n.dtype)
    t1 = F.normalize(torch.cross(n, e, dim=-1), dim=-1)
    t2 = torch.cross(n, t1, dim=-1)
    return t1, t2


def rotmat(quat: torch.Tensor) -> torch.Tensor:
    lead = quat.shape[:-1]
    return quat_to_rotmat(quat.reshape(-1, 4)).reshape(*lead, 3, 3)


def lever_arms(quat: torch.Tensor, p_box_local: torch.Tensor) -> torch.Tensor:
    """World lever arms r_i = R p_i from the COM, (..., 4) x (..., K, 3) -> (..., K, 3)."""
    return torch.einsum("...ij,...kj->...ki", rotmat(quat), p_box_local)


def aggregate_wrench(j: torch.Tensor, r: torch.Tensor, active: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Masked sums: dv = sum_i j_i, dL = sum_i r_i x j_i  ((..., K, 3) -> (..., 3), (..., 3))."""
    a = active.unsqueeze(-1)
    return (j * a).sum(-2), (torch.cross(r, j, dim=-1) * a).sum(-2)


def inertia_world_inv(quat: torch.Tensor, inertia_diag: torch.Tensor) -> torch.Tensor:
    """R diag(1/I) R^T, (..., 4), (..., 3) -> (..., 3, 3)."""
    R = rotmat(quat)
    return torch.einsum("...ij,...j,...kj->...ik", R, 1.0 / inertia_diag, R)


def wrench_to_delta(dv_c: torch.Tensor, dL: torch.Tensor, quat: torch.Tensor, inertia_diag_over_m: torch.Tensor) -> torch.Tensor:
    """(dv_c, dL) -> 6-D contact-induced (delta v, delta omega_world) = [dv_c, (I_w/m)^-1 dL]."""
    dw = torch.einsum("...ij,...j->...i", inertia_world_inv(quat, inertia_diag_over_m), dL)
    return torch.cat([dv_c, dw], -1)


def body_angular_impulse_torch(w_world_k: torch.Tensor, w_world_k1: torch.Tensor, quat_k: torch.Tensor, inertia_diag: torch.Tensor, dt: float | torch.Tensor) -> torch.Tensor:
    """Frozen label formula in torch: L_b = I (w_b[k+1] - w_b[k]) + dt w_b[k] x I w_b[k], with w_b = R[k]^T w_world."""
    R = rotmat(quat_k)
    wb = torch.einsum("...ji,...j->...i", R, w_world_k)
    wb1 = torch.einsum("...ji,...j->...i", R, w_world_k1)
    if torch.is_tensor(dt) and dt.dim() > 0:
        dt = dt.unsqueeze(-1)
    return inertia_diag * (wb1 - wb) + dt * torch.cross(wb, inertia_diag * wb, dim=-1)


def gyro_delta_omega_world(quat_k: torch.Tensor, w_world_k: torch.Tensor, inertia_diag_over_m: torch.Tensor, dt: float) -> torch.Tensor:
    """Exact torque-free (gyroscopic) delta omega_world of one Chrono step: -dt R I^-1 (w_b x I w_b)."""
    R = rotmat(quat_k)
    wb = torch.einsum("...ji,...j->...i", R, w_world_k)
    dwb = -dt * torch.cross(wb, inertia_diag_over_m * wb, dim=-1) / inertia_diag_over_m
    return torch.einsum("...ij,...j->...i", R, dwb)


def cone_violation(j: torch.Tensor, n: torch.Tensor, mu: torch.Tensor, active: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """1 for active slots whose impulse leaves the friction cone (j_n < 0 or |j_t| > mu j_n), else 0.  (..., K)."""
    jn = (j * n).sum(-1)
    jt = (j - jn.unsqueeze(-1) * n).norm(dim=-1)
    return ((jn < -eps) | (jt > mu.unsqueeze(-1) * jn.clamp_min(0) + eps)).float() * active


def single_contact_scale(feats: torch.Tensor, quat: torch.Tensor, inertia_diag_over_m: torch.Tensor, restitution: torch.Tensor, g_dt: torch.Tensor,
                         min_bounce_speed: float = 0.15) -> torch.Tensor:
    """Exact single-contact frictionless normal impulse per unit mass (Delassus scaling) plus the gravity step:
        s_i = max(0, -(1 + e_i) v_n_i) / (1 + (r_i x n_i)^T (I_b/m)^-1 (r_i x n_i)) + g dt,   e_i = e if -v_n_i > v_bounce else 0
    (Chrono NSC applies restitution only above min_bounce_speed).  feats: slot_features (..., K, 19)."""
    n, r, v_n = feats[..., 2:5], feats[..., 8:11], feats[..., 14]
    rxn = torch.cross(r, n, dim=-1)                                            # world
    Iw_inv = inertia_world_inv(quat, inertia_diag_over_m)                      # (..., 3, 3) = R diag(m/I) R^T
    denom = 1.0 + torch.einsum("...ki,...ij,...kj->...k", rxn, Iw_inv, rxn)
    e = torch.where(-v_n > min_bounce_speed, restitution.unsqueeze(-1).expand_as(v_n), torch.zeros_like(v_n))
    return torch.relu(-(1.0 + e) * v_n) / denom + g_dt


def skew(v: torch.Tensor) -> torch.Tensor:
    """(..., 3) -> (..., 3, 3) cross-product matrix [v]x."""
    x, y, z = v.unbind(-1)
    o = torch.zeros_like(x)
    return torch.stack([torch.stack([o, -z, y], -1), torch.stack([z, o, -x], -1), torch.stack([-y, x, o], -1)], -2)


SOLVER_FEAT_DIM = 5   # analytic single-contact impulse in the (n, t1, t2) basis (3), stick flag, normal-velocity target


def single_contact_solver(feats: torch.Tensor, state: torch.Tensor, inertia_diag_over_m: torch.Tensor, mu: torch.Tensor, restitution: torch.Tensor,
                          g_dt: torch.Tensor, d_chrono: torch.Tensor, dt: torch.Tensor, min_bounce_speed: float = 0.15, recovery_speed: float = 0.6) -> tuple[torch.Tensor, torch.Tensor]:
    """Closed-form Chrono-NSC-style impulse of ONE frictional contact acting alone (per unit mass, world frame).

    Contact-point velocity after the free-flight gravity step: v_f = v_p + g dt e_z.  Impulse j (per unit mass) changes it by
    G j with the contact Delassus matrix G = I - [r]x (I_b/m)_w^-1 [r]x.  Normal target v'_n (verified in recorded traces):
        gap > 0            : v'_n = -d/dt   (stabilization: the gap may close, not more)      -> impulse iff v_f,n < -d/dt
        gap <= 0, bounce   : v'_n = e * (-v_f,n) when -v_f,n > min_bounce_speed (restitution)
        gap <= 0, no bounce: v'_n = min(-d/dt, recovery_speed) (penetration recovery)
    Tangential: stick (v'_t = 0) if the impulse stays inside the friction cone, else Coulomb slip against the pre-impact
    sliding direction.  Returns (j (..., K, 3) world, feats (..., K, SOLVER_FEAT_DIM) = [j in (n,t1,t2), stick, v'_n target]).
    """
    n, r, v_rel = feats[..., 2:5], feats[..., 8:11], feats[..., 11:14]
    active = feats[..., 0:1]
    lead = n.shape[:-2]
    K = n.shape[-2]
    Iw_inv = inertia_world_inv(state[..., 3:7], inertia_diag_over_m)                         # (..., 3, 3)
    S = skew(r)                                                                               # (..., K, 3, 3)
    G = torch.eye(3, device=n.device, dtype=n.dtype).expand(*lead, K, 3, 3) - S @ Iw_inv.unsqueeze(-3) @ S
    ez = torch.zeros_like(v_rel)
    ez[..., 2] = 1.0
    v_f = v_rel + g_dt[..., None, None] * ez                                                  # after this step's gravity
    v_fn = (v_f * n).sum(-1)                                                                  # (..., K)
    d_dt = d_chrono / dt.unsqueeze(-1)
    e = restitution.unsqueeze(-1).expand_as(v_fn)
    bounce = (-v_fn > min_bounce_speed)
    pen = d_chrono <= 0
    target = torch.where(pen, torch.where(bounce, e * (-v_fn), torch.clamp(-d_dt, max=recovery_speed)), -d_dt)
    need = (target - v_fn).clamp_min(0.0)                                                     # required normal velocity change (>= 0)
    # stick solve: G j = need n - v_f,t
    v_ft = v_f - v_fn.unsqueeze(-1) * n
    rhs = need.unsqueeze(-1) * n - v_ft
    j_stick = torch.linalg.solve(G, rhs.unsqueeze(-1)).squeeze(-1)
    jn_s = (j_stick * n).sum(-1)
    jt_s = j_stick - jn_s.unsqueeze(-1) * n
    mu_ = mu.unsqueeze(-1).expand_as(jn_s)
    stick_ok = (jn_s >= 0) & (jt_s.norm(dim=-1) <= mu_ * jn_s.clamp_min(0) + 1e-9)
    # slip solve: j = jn (n - mu t_hat), t_hat = sliding direction of v_f,t
    t_hat = v_ft / v_ft.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    dirn = n - mu_.unsqueeze(-1) * t_hat
    denom = torch.einsum("...i,...ij,...j->...", n, G, dirn).clamp_min(1e-6)
    jn_slip = (need / denom).clamp_min(0.0)
    j_slip = jn_slip.unsqueeze(-1) * dirn
    j = torch.where(stick_ok.unsqueeze(-1), j_stick, j_slip) * active
    j = torch.where((need > 0).unsqueeze(-1), j, torch.zeros_like(j))
    t1, t2 = tangent_basis(n)
    jf = torch.stack([(j * n).sum(-1), (j * t1).sum(-1), (j * t2).sum(-1), stick_ok.to(j.dtype), target], -1) * active
    return j, jf


def phys_features(phys: dict[str, torch.Tensor], half_extents: torch.Tensor) -> torch.Tensor:
    """(B,) / (B,3) physical parameters -> (..., PHYS_FEAT_DIM) features broadcast to half_extents' leading dims:
    [log mass, I/m / he_max^2 (3), mu, restitution, dt * 1e3]."""
    lead = half_extents.shape[:-1]
    b = lambda x: x.reshape(x.shape[0], *([1] * (len(lead) - 1)), *x.shape[1:]).expand(*lead, *x.shape[1:])  # noqa: E731
    he2 = half_extents.max(-1, keepdim=True).values ** 2
    return torch.cat([b(phys["log_mass"]).unsqueeze(-1), b(phys["inertia_diag_over_m"]) / he2, b(phys["mu"]).unsqueeze(-1),
                      b(phys["restitution"]).unsqueeze(-1), b(phys["dt"]).unsqueeze(-1) * 1e3], -1)


def broadcast_phys(x: torch.Tensor, lead: torch.Size) -> torch.Tensor:
    """(B,) or (B,d) -> (*lead,) or (*lead, d)."""
    return x.reshape(x.shape[0], *([1] * (len(lead) - 1)), *x.shape[1:]).expand(*lead, *x.shape[1:])


# ---- decoder -------------------------------------------------------------------------------------
class ContactImpulseDecoder(nn.Module):
    """Set transformer over the K contact slots -> per-slot mass-normalized impulses and the aggregated wrench.

    head_mode 'cone': j = softplus(a_n) n + friction-disk(a_t) (physically structured);  'free': unconstrained 3-D.
    use_slot_embedding=False drops the encoder's slot query (explicit-geometry-only ablation); when contacts carry
    no 'slot_embedding' (gt / analytic geometry) a learned null embedding is used.
    pooled_only=True decodes every slot from the pooled 64-D context only (POOL-DEC ablation; forces 'free').
    """

    def __init__(self, slot_embed_dim: int = 128, ctx_dim: int = 64, width: int = 128, n_blocks: int = 2, n_heads: int = 4,
                 head_mode: str = "cone", use_slot_embedding: bool = True, pooled_only: bool = False, out_scale: float = 1.0, K: int = K_SLOTS,
                 timing_feats: bool = True, scaled_head: bool = True, head_scale: str = "vn", dropout: float = 0.0, chrono_gap_feats: bool = True) -> None:
        """timing_feats (v2): add the exact contact-point height / velocity-deficit features (slot_timing_features).
        scaled_head (v2): the head outputs a multiplier of a per-slot physical scale instead of a fixed out_scale, so resting
        (g dt) and impact (|v_n|) impulses are both O(1) targets.  head_scale 'vn': s_i = relu(-v_n_i) + g dt;
        'delassus': the exact single-contact normal impulse (single_contact_scale) -> the multiplier is ~1 for a clean
        single-corner impact and the network learns only multi-contact coupling, friction and gap-closing splits."""
        super().__init__()
        assert head_mode in ("cone", "free") and head_scale in ("vn", "delassus", "solver")
        self.head_mode = "free" if pooled_only else head_mode
        self.use_slot_embedding, self.pooled_only, self.K = use_slot_embedding, pooled_only, K
        self.timing_feats, self.scaled_head, self.head_scale, self.dropout_p = timing_feats, scaled_head, head_scale, dropout
        self.chrono_gap_feats = chrono_gap_feats   # v3: envelope-corrected exact Chrono gap in the timing features
        self.slot_embed_dim, self.ctx_dim, self.width, self.n_blocks, self.n_heads = slot_embed_dim, ctx_dim, width, n_blocks, n_heads
        self.ctx_pool = ContactPooling(SLOT_FEAT_DIM, 64, ctx_dim)
        self.null_embed = nn.Parameter(torch.zeros(slot_embed_dim))
        extra = (EXTRA_FEAT_DIM if timing_feats else 0) + (SOLVER_FEAT_DIM if head_scale == "solver" else 0)
        if pooled_only:
            in_dim = ctx_dim + PHYS_FEAT_DIM + K
        else:
            in_dim = (slot_embed_dim if use_slot_embedding else 0) + SLOT_FEAT_DIM + extra + PHYS_FEAT_DIM + ctx_dim
        self.in_dim = in_dim
        self.inp = mlp(in_dim, width, width)
        self.blocks = nn.ModuleList([nn.ModuleDict({
            "attn": nn.MultiheadAttention(width, n_heads, batch_first=True), "n1": nn.LayerNorm(width),
            "ff": mlp(width, 2 * width, width), "n2": nn.LayerNorm(width)}) for _ in range(n_blocks)])
        self.head = nn.Linear(width, 3)
        self.drop = nn.Dropout(dropout)
        self.register_buffer("out_scale", torch.tensor(float(out_scale)))

    def config(self) -> dict:
        return {"slot_embed_dim": self.slot_embed_dim, "ctx_dim": self.ctx_dim, "width": self.width, "n_blocks": self.n_blocks, "n_heads": self.n_heads,
                "head_mode": self.head_mode, "use_slot_embedding": self.use_slot_embedding, "pooled_only": self.pooled_only, "out_scale": float(self.out_scale), "K": self.K,
                "timing_feats": self.timing_feats, "scaled_head": self.scaled_head, "head_scale": self.head_scale, "dropout": self.dropout_p, "chrono_gap_feats": self.chrono_gap_feats}

    def forward(self, contacts: dict[str, torch.Tensor], state: torch.Tensor, half_extents: torch.Tensor, phys: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """contacts: active (...,K), d (...,K), n (...,K,3), p_box_local (...,K,3), optional slot_embedding (...,K,D);
        state (...,13); half_extents (B,3) or (...,3); phys: PHYS_KEYS tensors with leading (B,).
        Returns j_slot (...,K,3), r (...,K,3), dv_c (...,3), dL (...,3), dw_c (...,3), wrench (...,6)."""
        lead = state.shape[:-1]
        he = half_extents if half_extents.shape[:-1] == lead else broadcast_phys(half_extents, lead)
        active = contacts["active"]
        geom = {k: contacts[k] for k in ("active", "d", "n", "p_box_local")}
        feats = slot_features(geom, state, he)                                   # (..., K, 19), zero for inactive slots
        ctx = self.ctx_pool(feats, active)                                       # (..., ctx_dim)
        pf = phys_features(phys, he)                                             # (..., 7)
        K = feats.shape[-2]
        dt = broadcast_phys(phys["dt"], lead)
        env = broadcast_phys(phys["envelope"], lead) if (self.chrono_gap_feats and (self.timing_feats or self.head_scale == "solver")) else None
        tf = slot_timing_features(feats, state, dt, env) if self.timing_feats else None
        j_solver = None
        if self.head_scale == "solver":   # analytic single-contact frictional impulse as prior features + normal scale
            g_dt0 = broadcast_phys(phys["gravity"], lead) * dt
            d_ch = (chrono_gap(feats, state, env) if env is not None else state[..., 2:3].unsqueeze(-2) + feats[..., 10:11])[..., 0]
            j_solver, sf = single_contact_solver(feats, state, broadcast_phys(phys["inertia_diag_over_m"], lead), broadcast_phys(phys["mu"], lead),
                                                 broadcast_phys(phys["restitution"], lead), g_dt0, d_ch, dt)
            tf = torch.cat([tf, sf], -1) if tf is not None else sf
        ex = lambda x: x.unsqueeze(-2).expand(*x.shape[:-1], K, x.shape[-1])   # noqa: E731
        if self.pooled_only:
            onehot = torch.eye(K, device=state.device, dtype=state.dtype).expand(*lead, K, K)
            x = torch.cat([ex(ctx), ex(pf), onehot], -1)
        else:
            parts = []
            if self.use_slot_embedding:
                emb = contacts.get("slot_embedding")
                if emb is None:
                    emb = self.null_embed.expand(*lead, K, self.slot_embed_dim)
                parts.append(emb)
            parts += [feats] + ([tf] if tf is not None else []) + [ex(pf), ex(ctx)]
            x = torch.cat(parts, -1)
        x = self.drop(self.inp(x.reshape(-1, K, self.in_dim)))
        # inactive slots are excluded as keys (unless a frame has no active slot: then attend to all, output is masked anyway)
        kpm = (active.reshape(-1, K) <= 0)
        kpm = kpm & ~kpm.all(-1, keepdim=True)
        for blk in self.blocks:
            x = blk["n1"](x + blk["attn"](x, x, x, key_padding_mask=kpm, need_weights=False)[0])
            x = blk["n2"](x + self.drop(blk["ff"](x)))
        a = self.head(x).reshape(*lead, K, 3)
        n = contacts["n"]
        if self.scaled_head:   # per-slot physical scale from exact inputs
            g_dt = (broadcast_phys(phys["gravity"], lead) * dt).unsqueeze(-1)
            if self.head_scale == "solver":
                scale = (j_solver * n).sum(-1) + g_dt                                          # analytic normal impulse + gravity step (..., K)
            elif self.head_scale == "delassus":
                scale = single_contact_scale(feats, state[..., 3:7], broadcast_phys(phys["inertia_diag_over_m"], lead), broadcast_phys(phys["restitution"], lead), g_dt)
            else:
                scale = torch.relu(-feats[..., 14]) + g_dt                       # (..., K)
        else:
            scale = self.out_scale.expand(*lead, K)
        if self.head_mode == "cone":
            t1, t2 = tangent_basis(n)
            mu = broadcast_phys(phys["mu"], lead).unsqueeze(-1)                  # (..., 1)
            jn = F.softplus(a[..., 0]) * scale                                   # (..., K) >= 0
            at = a[..., 1:3]
            mag = at.norm(dim=-1, keepdim=True)
            jt = mu.unsqueeze(-1) * jn.unsqueeze(-1) * torch.tanh(mag) * at / mag.clamp_min(1e-9)   # inside the friction disk
            j = jn.unsqueeze(-1) * n + jt[..., 0:1] * t1 + jt[..., 1:2] * t2
        else:
            j = a * scale.unsqueeze(-1)
        j = j * active.unsqueeze(-1)
        r = lever_arms(state[..., 3:7], contacts["p_box_local"])
        dv_c, dL = aggregate_wrench(j, r, active)
        I_over_m = broadcast_phys(phys["inertia_diag_over_m"], lead)
        w6 = wrench_to_delta(dv_c, dL, state[..., 3:7], I_over_m)
        out = {"j_slot": j, "r": r, "dv_c": dv_c, "dL": dL, "dw_c": w6[..., 3:6], "wrench": w6}
        if j_solver is not None:   # aggregated physics-only wrench (solver prior alone), for anchoring losses
            dv_s, dL_s = aggregate_wrench(j_solver * active.unsqueeze(-1), r, active)
            out["wrench_solver"] = wrench_to_delta(dv_s, dL_s, state[..., 3:7], I_over_m)
        return out


def build_decoder(cfg: dict) -> ContactImpulseDecoder:
    keys = ("slot_embed_dim", "ctx_dim", "width", "n_blocks", "n_heads", "head_mode", "use_slot_embedding", "pooled_only", "out_scale", "K")
    kw = {k: cfg[k] for k in keys if k in cfg}
    kw["timing_feats"], kw["scaled_head"] = cfg.get("timing_feats", False), cfg.get("scaled_head", False)   # v1 checkpoints predate both
    kw["head_scale"], kw["dropout"] = cfg.get("head_scale", "vn"), cfg.get("dropout", 0.0)
    kw["chrono_gap_feats"] = cfg.get("chrono_gap_feats", False)
    return ContactImpulseDecoder(**kw)
