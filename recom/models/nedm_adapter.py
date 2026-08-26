"""Contact-to-NeDM adapter (Section 12.5): per-contact deterministic features from the box state,
followed by permutation-invariant pooling over the K slots -> one context vector per timestep.

Per-slot feature (dim 19 + latent):
    active, d, n(3), p_box_local/he(3), r(3) lever arm (world), v_rel(3), v_n, v_t(3), |v_t|, ... latent(L)
The ground is static, so v_rel = v + w x r.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ..config import K_SLOTS
from ..geometry.transforms import quat_to_rotmat

SLOT_FEAT_DIM = 19


def slot_features(contacts: dict[str, torch.Tensor], state: torch.Tensor, half_extents: torch.Tensor) -> torch.Tensor:
    """contacts: dict with active (..., K), d (..., K), n (..., K, 3), p_box_local (..., K, 3), optional latent.
    state: (..., 13) = [pos(3), quat(4), v(3), w(3)].  Returns (..., K, SLOT_FEAT_DIM[+L])."""
    pos, quat, v, w = state[..., 0:3], state[..., 3:7], state[..., 7:10], state[..., 10:13]
    lead = state.shape[:-1]
    R = quat_to_rotmat(quat.reshape(-1, 4)).reshape(*lead, 3, 3)
    pbl = contacts["p_box_local"]
    r = torch.einsum("...ij,...kj->...ki", R, pbl)                       # lever arm in world (box frame origin = COM)
    v_rel = v.unsqueeze(-2) + torch.cross(w.unsqueeze(-2).expand_as(r), r, dim=-1)
    n = contacts["n"]
    v_n = (v_rel * n).sum(-1, keepdim=True)
    v_t = v_rel - v_n * n
    he = half_extents.unsqueeze(-2)
    feats = [contacts["active"].unsqueeze(-1), contacts["d"].unsqueeze(-1), n, pbl / he, r, v_rel, v_n, v_t, v_t.norm(dim=-1, keepdim=True)]
    f = torch.cat(feats, -1)
    if "latent" in contacts:
        f = torch.cat([f, contacts["latent"]], -1)
    return f * contacts["active"].unsqueeze(-1)


class ContactPooling(nn.Module):
    """phi per slot -> masked sum + max pooling + count -> context vector."""

    def __init__(self, in_dim: int, hidden: int = 64, out_dim: int = 64) -> None:
        super().__init__()
        self.phi = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU())
        self.rho = nn.Sequential(nn.Linear(2 * hidden + 1, out_dim), nn.GELU())
        self.out_dim = out_dim

    def forward(self, slot_feats: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
        h = self.phi(slot_feats) * active.unsqueeze(-1)
        s = h.sum(-2)
        m = (h + (active.unsqueeze(-1) - 1.0) * 1e4).max(-2).values * (active.sum(-1, keepdim=True) > 0).float()
        cnt = active.sum(-1, keepdim=True) / K_SLOTS
        return self.rho(torch.cat([s, m, cnt], -1))
