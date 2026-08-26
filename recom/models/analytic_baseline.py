"""Exact analytic box-plane contact in the encoder output contract (torch, batched).  Correctness baseline."""
from __future__ import annotations

import torch
import torch.nn as nn

from ..config import K_SLOTS
from ..geometry.box_plane_analytic import CORNER_SIGNS
from ..geometry.transforms import quat_to_rotmat


class AnalyticBoxPlaneEncoder(nn.Module):
    def __init__(self, margin: float = 0.002, K: int = K_SLOTS, latent_dim: int = 16, sharpness: float = 2000.0) -> None:
        super().__init__()
        self.margin, self.K, self.latent_dim, self.sharpness = margin, K, latent_dim, sharpness
        self.register_buffer("signs", torch.tensor(CORNER_SIGNS, dtype=torch.float32))

    def canonical_tokens(self, half_extents: torch.Tensor) -> dict:
        return {}

    def forward(self, half_extents: torch.Tensor, pos: torch.Tensor, quat: torch.Tensor, tokens: dict | None = None) -> dict[str, torch.Tensor]:
        B = pos.shape[0]
        R = quat_to_rotmat(quat)
        c_local = self.signs[None] * half_extents[:, None, :]                       # (B, 8, 3)
        c_pair = torch.einsum("bij,bkj->bki", R, c_local)
        c_pair = c_pair + torch.stack([torch.zeros_like(pos[:, 2]), torch.zeros_like(pos[:, 2]), pos[:, 2]], -1)[:, None, :]
        gaps = c_pair[..., 2]                                                       # (B, 8)
        order = gaps.argsort(dim=1)[:, : self.K]
        g = torch.gather(gaps, 1, order)
        active = (g <= self.margin).float()
        logit = (self.margin - g) * self.sharpness
        p_box = torch.gather(c_local, 1, order.unsqueeze(-1).expand(-1, -1, 3))
        p_pair = torch.gather(c_pair, 1, order.unsqueeze(-1).expand(-1, -1, 3))
        p_gnd = torch.cat([p_pair[..., :2], torch.zeros_like(p_pair[..., 2:3])], -1)
        n = torch.zeros(B, self.K, 3, device=pos.device)
        n[..., 2] = 1.0
        card = torch.nn.functional.one_hot(active.sum(1).long(), self.K + 1).float() * 20.0
        return {"logit": logit, "p_box_local": p_box * active.unsqueeze(-1), "p_ground_rel": p_gnd * active.unsqueeze(-1), "n": n, "d": g,
                "latent": torch.zeros(B, self.K, self.latent_dim, device=pos.device), "log_var": torch.full((B, self.K), -10.0, device=pos.device), "cardinality": card}
