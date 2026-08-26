"""NeDM-compatible recurrent transition model for the box (Section 3 / Experiments A, B, D, E).

tokens_t = [normalized state_t (13), half_extents (3), contact context_t (C)]
A causal continuous-token transformer (same design as NeDM's ContinuousTransformer) maps the window
of tokens to per-position deltas [dv (3), dw (3)] (normalized).  Pose is integrated exactly outside
the network with Chrono's semi-implicit rule (verified on data):
    v' = v + dv ; w' = w + dw ; pos' = pos + dt v' ; q' = exp(dt w') (x) q
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import K_SLOTS
from ..geometry.transforms import quat_from_omega_step, quat_normalize
from .nedm_adapter import SLOT_FEAT_DIM, ContactPooling, slot_features

STATE_DIM, TARGET_DIM = 13, 6


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
                 gravity_prior: bool = True, gravity: float = 9.81, contact_gate: bool = True) -> None:
        """contact_mode: 'none' (state-only), 'explicit' (explicit contact quantities), 'latent' (learned latent only),
        'explicit+latent'.  gravity_prior: the network predicts the residual over the known free-flight delta
        [0, 0, -g dt, 0, 0, 0] (exact for Chrono's semi-implicit Euler); normalization statistics must then be
        computed on residual targets (see compute_state_normalization(..., prior))."""
        super().__init__()
        self.contact_mode, self.block_size, self.dt, self.latent_dim = contact_mode, block_size, dt, latent_dim
        self.gravity_prior = gravity_prior
        # contact_gate: multiply the predicted residual by the frame's contact activation (any slot active).
        # Exact for Chrono: a step with no reported contact is pure free flight (no contact impulse).
        self.contact_gate = contact_gate and contact_mode != "none"
        prior = torch.tensor([0.0, 0.0, -gravity * dt, 0.0, 0.0, 0.0]) if gravity_prior else torch.zeros(6)
        self.register_buffer("prior", prior)
        for k, v in normalization.items():
            self.register_buffer(k, torch.as_tensor(v, dtype=torch.float32))
        in_dim = STATE_DIM + 3
        if contact_mode != "none":
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

    def norm_target(self, t: torch.Tensor) -> torch.Tensor:
        """Full delta -> normalized residual (after removing the gravity prior)."""
        return (t - self.prior - self.target_mean) / self.target_std

    def denorm_target(self, t: torch.Tensor) -> torch.Tensor:
        """Normalized residual -> full delta."""
        return t * self.target_std + self.target_mean + self.prior

    # -- contact context -------------------------------------------------------------
    def contact_context(self, contacts: dict[str, torch.Tensor] | None, states: torch.Tensor, half_extents: torch.Tensor) -> torch.Tensor | None:
        if self.contact_mode == "none":
            return None
        assert contacts is not None, "contact-conditioned model needs contacts"
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
        a = contacts.get("prob", contacts["active"])
        return a.max(dim=-1, keepdim=True).values.clamp(0, 1)

    def predict_delta(self, states, half_extents, contacts=None) -> torch.Tensor:
        """Full delta (B,T,6) = prior + gate * residual."""
        res = self.forward(states, half_extents, contacts) * self.target_std + self.target_mean
        g = self.gate(contacts)
        if g is not None:
            res = res * g
        return res + self.prior

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
