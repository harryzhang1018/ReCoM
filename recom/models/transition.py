"""NeDM-compatible recurrent transition model for the box (Section 3 / Experiments A, B, D, E).

tokens_t = [normalized state_t (13), half_extents (3), contact context_t (C)]
A causal continuous-token transformer (same design as NeDM's ContinuousTransformer) maps the window
of tokens to per-position deltas [dv (3), dw (3)] (normalized).  Pose is integrated exactly outside
the network with Chrono's semi-implicit rule (verified on data):
    v' = v + dv ; w' = w + dw ; pos' = pos + dt v' ; q' = exp(dt w') (x) q

Contact context C by contact_mode:
    explicit / latent / explicit+latent : DeepSets-pooled 64-D context from the K contact slots (baseline, BASE-64)
    wrench      : the 6-D contact-induced (delta v, delta omega) predicted by the impulse decoder (JL-6)
    wrench_lin  : its 3-D linear part only (J-3)
physics_residual: the decoder wrench is applied as a Newton-Euler step and the network predicts the residual (JL-6-R).
Priors (exact for Chrono, applied outside the network): gravity [0,0,-g dt] and, with gyro_prior, the torque-free
gyroscopic delta omega_world = -dt R I_b^-1 (w_b x I_b w_b) (a tumbling box changes omega_world in free flight).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import K_SLOTS
from ..geometry.transforms import quat_from_omega_step, quat_normalize
from .impulse_decoder import gyro_delta_omega_world
from .nedm_adapter import SLOT_FEAT_DIM, ContactPooling, slot_features

STATE_DIM, TARGET_DIM = 13, 6
WRENCH_MODES = ("wrench", "wrench_lin")


def box_inertia_diag_over_m(half_extents: torch.Tensor) -> torch.Tensor:
    """I_b/m of a homogeneous box, (...,3) -> (...,3) (only inertia ratios matter for the gyroscopic prior)."""
    x2, y2, z2 = half_extents[..., 0] ** 2, half_extents[..., 1] ** 2, half_extents[..., 2] ** 2
    return torch.stack([y2 + z2, x2 + z2, x2 + y2], -1) / 3.0


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd: int, n_head: int, dropout: float) -> None:
        super().__init__()
        self.qkv = nn.Linear(n_embd, 3 * n_embd)
        self.proj = nn.Linear(n_embd, n_embd)
        self.n_head, self.dropout = n_head, dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q, k, v = (t.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) for t in (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0)
        return self.proj(y.transpose(1, 2).reshape(B, T, C))


class Block(nn.Module):
    def __init__(self, n_embd: int, n_head: int, dropout: float) -> None:
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(n_embd), nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, dropout)
        self.mlp = nn.Sequential(nn.Linear(n_embd, 4 * n_embd), nn.GELU(), nn.Linear(4 * n_embd, n_embd), nn.Dropout(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class BoxTransitionModel(nn.Module):
    def __init__(self, normalization: dict, contact_mode: str = "none", block_size: int = 32, n_layer: int = 3, n_head: int = 4,
                 n_embd: int = 128, dropout: float = 0.0, latent_dim: int = 0, ctx_dim: int = 64, dt: float = 1e-3,
                 gravity_prior: bool = True, gravity: float = 9.81, contact_gate: bool = True, soft_gate: bool = False,
                 gyro_prior: bool = False, physics_residual: bool = False) -> None:
        """contact_mode: 'none' (state-only), 'explicit' (explicit contact quantities), 'latent' (learned latent only),
        'explicit+latent', 'wrench' (6-D decoder wrench), 'wrench_lin' (3-D linear impulse only).
        gravity_prior: the network predicts the residual over the known free-flight delta [0, 0, -g dt, 0, 0, 0]
        (exact for Chrono's semi-implicit Euler); normalization statistics must then be computed on residual targets
        (see compute_state_normalization(..., prior, gyro)).  gyro_prior adds the exact state-dependent gyroscopic
        delta omega.  physics_residual (wrench modes): delta = prior + gate * (wrench + residual)."""
        super().__init__()
        self.contact_mode, self.block_size, self.dt, self.latent_dim = contact_mode, block_size, dt, latent_dim
        self.gravity_prior, self.gyro_prior = gravity_prior, gyro_prior
        self.physics_residual = physics_residual and contact_mode in WRENCH_MODES
        # contact_gate: multiply the predicted residual by the frame's contact activation (any slot active).
        # Exact for Chrono: a step with no reported contact is pure free flight (no contact impulse).
        self.contact_gate = contact_gate and contact_mode != "none"
        # soft_gate: use the encoder's activation probability as the gate (only consistent for models trained
        # jointly on soft learned contacts); otherwise the gate is the hard 0/1 active mask.
        self.soft_gate = soft_gate
        prior = torch.tensor([0.0, 0.0, -gravity * dt, 0.0, 0.0, 0.0]) if gravity_prior else torch.zeros(6)
        self.register_buffer("prior", prior)
        for k, v in normalization.items():
            self.register_buffer(k, torch.as_tensor(v, dtype=torch.float32))
        in_dim = STATE_DIM + 3
        if contact_mode in WRENCH_MODES:
            assert hasattr(self, "wrench_std"), "wrench modes need 'wrench_std' in the normalization dict (from the decoder checkpoint)"
            self.wrench_dim = 6 if contact_mode == "wrench" else 3
            in_dim += self.wrench_dim
        elif contact_mode != "none":
            slot_dim = {"explicit": SLOT_FEAT_DIM, "latent": 1 + latent_dim, "explicit+latent": SLOT_FEAT_DIM + latent_dim}[contact_mode]
            self.pool = ContactPooling(slot_dim, 64, ctx_dim)
            in_dim += ctx_dim
        self.inp = nn.Linear(in_dim, n_embd)
        self.pos_emb = nn.Parameter(torch.zeros(1, block_size, n_embd))
        self.blocks = nn.ModuleList([Block(n_embd, n_head, dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Sequential(nn.Linear(n_embd, n_embd), nn.GELU(), nn.Linear(n_embd, TARGET_DIM))

    # -- normalization ---------------------------------------------------------------
    def norm_state(self, s: torch.Tensor) -> torch.Tensor:
        return (s - self.state_mean) / self.state_std

    def prior_delta(self, states: torch.Tensor, half_extents: torch.Tensor) -> torch.Tensor:
        """Known part of the delta (...,6): gravity and (optionally) the exact gyroscopic delta omega from the current state."""
        p = self.prior.expand(*states.shape[:-1], 6)
        if not self.gyro_prior:
            return p
        he = half_extents if half_extents.shape[:-1] == states.shape[:-1] else half_extents.reshape(half_extents.shape[0], *([1] * (states.dim() - 2)), 3).expand(*states.shape[:-1], 3)
        dw = gyro_delta_omega_world(states[..., 3:7], states[..., 10:13], box_inertia_diag_over_m(he), self.dt)
        return torch.cat([p[..., :3], p[..., 3:] + dw], -1)

    def norm_target(self, t: torch.Tensor, states: torch.Tensor | None = None, half_extents: torch.Tensor | None = None) -> torch.Tensor:
        """Full delta -> normalized residual (after removing the priors; states/half_extents needed with gyro_prior)."""
        prior = self.prior_delta(states, half_extents) if self.gyro_prior else self.prior
        return (t - prior - self.target_mean) / self.target_std

    def denorm_target(self, t: torch.Tensor, states: torch.Tensor | None = None, half_extents: torch.Tensor | None = None) -> torch.Tensor:
        """Normalized residual -> full delta."""
        prior = self.prior_delta(states, half_extents) if self.gyro_prior else self.prior
        return t * self.target_std + self.target_mean + prior

    def wrench_delta(self, contacts: dict[str, torch.Tensor]) -> torch.Tensor:
        """Decoder wrench as a 6-D delta (omega part zero for wrench_lin)."""
        w = contacts["wrench"]
        if self.contact_mode == "wrench_lin":
            w = torch.cat([w[..., :3], torch.zeros_like(w[..., :3])], -1)
        return w

    # -- contact context -------------------------------------------------------------
    def contact_context(self, contacts: dict[str, torch.Tensor] | None, states: torch.Tensor, half_extents: torch.Tensor) -> torch.Tensor | None:
        if self.contact_mode == "none":
            return None
        assert contacts is not None, "contact-conditioned model needs contacts"
        if self.contact_mode in WRENCH_MODES:
            assert "wrench" in contacts, "wrench modes need contacts['wrench'] from the impulse decoder"
            return (contacts["wrench"] / self.wrench_std)[..., : self.wrench_dim]
        if self.contact_mode == "latent":
            f = torch.cat([contacts["active"].unsqueeze(-1), contacts["latent"]], -1) * contacts["active"].unsqueeze(-1)
        else:
            c = dict(contacts)
            if self.contact_mode == "explicit":
                c.pop("latent", None)
            f = slot_features(c, states, half_extents)
        return self.pool(f, contacts["active"])

    def forward(self, states: torch.Tensor, half_extents: torch.Tensor, contacts: dict[str, torch.Tensor] | None = None) -> torch.Tensor:
        """states (B,T,13), half_extents (B,3) or (B,T,3), contacts dict of (B,T,K,...) -> normalized deltas (B,T,6)."""
        B, T, _ = states.shape
        assert T <= self.block_size
        he = half_extents if half_extents.dim() == 3 else half_extents[:, None, :].expand(B, T, 3)
        x = [self.norm_state(states), he / 0.15]
        ctx = self.contact_context(contacts, states, he)
        if ctx is not None:
            x.append(ctx)
        h = self.inp(torch.cat(x, -1)) + self.pos_emb[:, :T]
        for blk in self.blocks:
            h = blk(h)
        return self.head(self.ln_f(h))

    def gate(self, contacts: dict[str, torch.Tensor] | None) -> torch.Tensor | None:
        """(B,T,1) activation gate from the contact set (soft probability if available, else hard mask)."""
        if not self.contact_gate or contacts is None:
            return None
        a = contacts.get("prob", contacts["active"]) if self.soft_gate else contacts["active"]
        return a.max(dim=-1, keepdim=True).values.clamp(0, 1)

    def predict_delta(self, states, half_extents, contacts=None) -> torch.Tensor:
        """Full delta (B,T,6) = prior(state) + gate * (residual [+ decoder wrench with physics_residual])."""
        res = self.forward(states, half_extents, contacts) * self.target_std + self.target_mean
        if self.physics_residual:
            res = res + self.wrench_delta(contacts)
        g = self.gate(contacts)
        if g is not None:
            res = res * g
        return res + self.prior_delta(states, half_extents)

    def integrate(self, state: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        """Exact Chrono-style semi-implicit update from state (B,13) and delta (B,6)."""
        pos, quat, v, w = state[:, 0:3], state[:, 3:7], state[:, 7:10], state[:, 10:13]
        v1, w1 = v + delta[:, :3], w + delta[:, 3:]
        pos1 = pos + self.dt * v1
        q1 = quat_from_omega_step(quat, w1, self.dt)
        q1 = q1 * torch.sign(q1[:, :1] + 1e-12)
        return torch.cat([pos1, q1, v1, w1], -1)


def gt_contacts_from_batch(batch: dict[str, torch.Tensor], latent_dim: int = 0) -> dict[str, torch.Tensor]:
    """Oracle (Chrono) contacts in the adapter format from a TransitionWindowDataset batch."""
    c = {"active": batch["c_active"], "d": batch["c_d"], "n": batch["c_n"], "p_box_local": batch["c_p_box_local"]}
    if latent_dim > 0:
        c["latent"] = torch.zeros(*batch["c_active"].shape, latent_dim, device=batch["c_active"].device)
    return c
