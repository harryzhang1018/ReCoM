"""Contact encoders for Study 1 (Section 12).

Common output contract (dict of tensors, K = 4 slots, batch B):
    logit         (B, K)      active / near-contact logit
    p_box_local   (B, K, 3)   contact point in the box frame (inside the box by construction)
    p_ground_rel  (B, K, 3)   ground point in the pair frame (world axes, origin = box xy projected on z = 0)
    n             (B, K, 3)   unit normal (ground -> box)
    d             (B, K)      signed distance
    latent        (B, K, L)   learned local contact latent
    log_var       (B, K)      log-variance of the normalized box-point error (uncertainty)
    cardinality   (B, K+1)    logits for the number of contacts 0..K
    slot_embedding (B, K, d_model)  post-attention slot query (unsupervised; consumed by the impulse decoder)

Pair-relative frame: world axes with origin (pos_x, pos_y, 0).  All pose-dependent features are
functions of (R, pos_z) only, so the prediction is exactly invariant to in-plane translation.
Surface tokens are processed by shared per-token MLPs + permutation-invariant aggregation, so
the result is invariant to token order.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import K_SLOTS
from ..geometry.box_mesh import PATCH_FEATURE_DIM, patch_feature_matrix, surface_patch_tokens, surface_points
from ..geometry.transforms import quat_to_rotmat


def mlp(i: int, h: int, o: int, layers: int = 2, norm: bool = True) -> nn.Sequential:
    mods: list[nn.Module] = []
    d = i
    for _ in range(layers - 1):
        mods += [nn.Linear(d, h)] + ([nn.LayerNorm(h)] if norm else []) + [nn.GELU()]
        d = h
    mods.append(nn.Linear(d, o))
    return nn.Sequential(*mods)


class GeometryCache:
    """Per-geometry cached canonical tokens (computed once per half-extent triple)."""

    def __init__(self, kind: str = "patch", n_face: int = 16, n_edge: int = 4) -> None:
        self.kind, self.n_face, self.n_edge = kind, n_face, n_edge
        self._cache: dict[tuple, dict[str, torch.Tensor]] = {}

    def get(self, half_extents: torch.Tensor) -> dict[str, torch.Tensor]:
        """half_extents (B, 3) -> batched token dict (B, N, ...)."""
        outs = []
        for he in half_extents.detach().cpu().numpy():
            key = tuple(np.round(he, 9).tolist())
            if key not in self._cache:
                self._cache[key] = self._build(np.asarray(he, dtype=np.float64))
            outs.append(self._cache[key])
        dev = half_extents.device
        return {k: torch.stack([o[k] for o in outs]).to(dev) for k in outs[0]}

    def _build(self, he: np.ndarray) -> dict[str, torch.Tensor]:
        if self.kind == "patch":
            tok = surface_patch_tokens(he)
            return {
                "feat": torch.from_numpy(patch_feature_matrix(tok)),                 # (F, 34)
                "centroid": torch.from_numpy(tok["centroid"].astype(np.float32)),   # (F, 3)
                "normal": torch.from_numpy(tok["normal"].astype(np.float32)),
                "verts": torch.from_numpy((tok["verts_rel"] + tok["centroid"][:, None]).astype(np.float32)),  # (F, 3, 3) box frame
                "adj": torch.from_numpy(tok["adj"]),                                # (F, 3)
            }
        sp = surface_points(he, self.n_face, self.n_edge)
        kind = np.eye(3, dtype=np.float32)[sp["kind"]]
        feat = np.concatenate([sp["pos"], sp["normal"], sp["weight"][:, None], kind, sp["scale"]], 1).astype(np.float32)
        return {"feat": torch.from_numpy(feat), "pos": torch.from_numpy(sp["pos"].astype(np.float32)), "normal": torch.from_numpy(sp["normal"].astype(np.float32))}


class SetDecoder(nn.Module):
    """K learned slot queries cross-attend over surface tokens and decode the contact contract."""

    def __init__(self, d_model: int, K: int = K_SLOTS, latent_dim: int = 16, n_heads: int = 4, n_layers: int = 2) -> None:
        super().__init__()
        self.K, self.latent_dim = K, latent_dim
        self.queries = nn.Parameter(torch.randn(K, d_model) * 0.02)
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(nn.ModuleDict({
                "cross": nn.MultiheadAttention(d_model, n_heads, batch_first=True),
                "self": nn.MultiheadAttention(d_model, n_heads, batch_first=True),
                "ff": mlp(d_model, 2 * d_model, d_model),
                "n1": nn.LayerNorm(d_model), "n2": nn.LayerNorm(d_model), "n3": nn.LayerNorm(d_model),
            }))
        self.head = nn.Linear(d_model, 1 + 3 + 3 + 3 + 1 + latent_dim + 1)
        self.card = mlp(d_model, d_model, K + 1)

    def forward(self, tokens: torch.Tensor, half_extents: torch.Tensor, pos_z: torch.Tensor) -> dict[str, torch.Tensor]:
        B = tokens.shape[0]
        q = self.queries.unsqueeze(0).expand(B, -1, -1)
        for L in self.layers:
            q = L["n1"](q + L["cross"](q, tokens, tokens, need_weights=False)[0])
            q = L["n2"](q + L["self"](q, q, q, need_weights=False)[0])
            q = L["n3"](q + L["ff"](q))
        h = self.head(q)
        he = half_extents.unsqueeze(1)
        logit = h[..., 0]
        p_box = torch.tanh(h[..., 1:4]) * he                       # inside the box
        p_gnd = h[..., 4:7] * he.max(-1, keepdim=True).values      # scale by box size
        n = F.normalize(h[..., 7:10] + torch.tensor([0.0, 0.0, 1.0], device=h.device), dim=-1)  # bias toward +Z init
        d = h[..., 10] * he.max(-1).values
        latent = h[..., 11:11 + self.latent_dim]
        log_var = 6.0 * torch.tanh(h[..., 11 + self.latent_dim] / 6.0)   # bounded in (-6, 6): keeps the NLL bounded below
        pooled = tokens.mean(1)
        return {"logit": logit, "p_box_local": p_box, "p_ground_rel": p_gnd, "n": n, "d": d, "latent": latent, "log_var": log_var, "cardinality": self.card(pooled), "slot_embedding": q}


def pose_features(R: torch.Tensor, pos_z: torch.Tensor, points_box: torch.Tensor, normals_box: torch.Tensor | None = None) -> torch.Tensor:
    """Pose-dependent per-token features in the pair frame: rotated point (3), height above plane (1),
    rotated normal (3), normal.z (1)."""
    p = torch.einsum("bij,bnj->bni", R, points_box)
    p = p + torch.stack([torch.zeros_like(pos_z), torch.zeros_like(pos_z), pos_z], -1).unsqueeze(1)
    feats = [p, p[..., 2:3]]
    if normals_box is not None:
        nw = torch.einsum("bij,bnj->bni", R, normals_box)
        feats += [nw, nw[..., 2:3]]
    return torch.cat(feats, -1)


class PatchContactEncoder(nn.Module):
    """Surface-patch (face) encoder: cached per-face MLP, adjacency message passing, plane interaction, set decoder."""

    def __init__(self, d_model: int = 128, n_mp: int = 3, latent_dim: int = 16, K: int = K_SLOTS) -> None:
        super().__init__()
        self.cache = GeometryCache("patch")
        self.geo_mlp = mlp(PATCH_FEATURE_DIM, d_model, d_model)        # cached canonical patch embedding
        pose_dim = 8 + 9 + 3                                           # centroid feats (8) + 3 vertices in pair frame (9) + vertex heights (3)
        self.pose_mlp = mlp(pose_dim + 4, d_model, d_model)            # + global (pos_z, R[:,2]) 
        self.fuse = mlp(2 * d_model, d_model, d_model)
        self.mp = nn.ModuleList([mlp(2 * d_model, d_model, d_model) for _ in range(n_mp)])
        self.mp_norm = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_mp)])
        self.decoder = SetDecoder(d_model, K, latent_dim)

    def canonical_tokens(self, half_extents: torch.Tensor) -> dict[str, torch.Tensor]:
        tok = self.cache.get(half_extents)
        tok["emb"] = self.geo_mlp(tok["feat"])
        return tok

    def forward(self, half_extents: torch.Tensor, pos: torch.Tensor, quat: torch.Tensor, tokens: dict | None = None) -> dict[str, torch.Tensor]:
        tok = tokens if tokens is not None else self.canonical_tokens(half_extents)
        R = quat_to_rotmat(quat)
        pos_z = pos[:, 2]
        B, Fn = tok["emb"].shape[:2]
        cf = pose_features(R, pos_z, tok["centroid"], tok["normal"])                      # (B, F, 8)
        vw = torch.einsum("bij,bfkj->bfki", R, tok["verts"]) + torch.stack([torch.zeros_like(pos_z), torch.zeros_like(pos_z), pos_z], -1)[:, None, None, :]
        glob = torch.cat([pos_z[:, None], R[:, 2, :]], -1)[:, None, :].expand(B, Fn, 4)  # height + world-down in box frame
        pf = torch.cat([cf, vw.reshape(B, Fn, 9), vw[..., 2], glob], -1)
        h = self.fuse(torch.cat([tok["emb"], self.pose_mlp(pf)], -1))
        adj = tok["adj"]                                                                  # (B, F, 3)
        for layer, norm in zip(self.mp, self.mp_norm):
            nb = torch.gather(h, 1, adj.reshape(B, -1, 1).expand(-1, -1, h.shape[-1])).reshape(B, Fn, 3, -1).mean(2)
            h = norm(h + layer(torch.cat([h, nb], -1)))
        out = self.decoder(h, half_extents, pos_z)
        out["tokens"] = h
        return out


class PointContactEncoder(nn.Module):
    """Surface-point baseline (PointNet++-lite: shared MLP, one kNN set-abstraction level, set decoder)."""

    def __init__(self, d_model: int = 128, k_nn: int = 16, latent_dim: int = 16, K: int = K_SLOTS, n_face: int = 16, n_edge: int = 4) -> None:
        super().__init__()
        self.cache = GeometryCache("point", n_face, n_edge)
        self.k_nn = k_nn
        self.geo_mlp = mlp(13, d_model, d_model)
        self.pose_mlp = mlp(8 + 4, d_model, d_model)
        self.fuse = mlp(2 * d_model, d_model, d_model)
        self.local = mlp(2 * d_model + 3, d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.decoder = SetDecoder(d_model, K, latent_dim)

    def canonical_tokens(self, half_extents: torch.Tensor) -> dict[str, torch.Tensor]:
        tok = self.cache.get(half_extents)
        tok["emb"] = self.geo_mlp(tok["feat"])
        return tok

    def forward(self, half_extents: torch.Tensor, pos: torch.Tensor, quat: torch.Tensor, tokens: dict | None = None) -> dict[str, torch.Tensor]:
        tok = tokens if tokens is not None else self.canonical_tokens(half_extents)
        R = quat_to_rotmat(quat)
        pos_z = pos[:, 2]
        B, N = tok["emb"].shape[:2]
        pf = pose_features(R, pos_z, tok["pos"], tok["normal"])
        glob = torch.cat([pos_z[:, None], R[:, 2, :]], -1)[:, None, :].expand(B, N, 4)
        h = self.fuse(torch.cat([tok["emb"], self.pose_mlp(torch.cat([pf, glob], -1))], -1))
        # kNN grouping in the box frame (pose independent -> could be cached too)
        p = tok["pos"]
        dist = torch.cdist(p, p)
        idx = dist.topk(self.k_nn, largest=False).indices                                # (B, N, k)
        nb = torch.gather(h, 1, idx.reshape(B, -1, 1).expand(-1, -1, h.shape[-1])).reshape(B, N, self.k_nn, -1)
        rel = torch.gather(p, 1, idx.reshape(B, -1, 1).expand(-1, -1, 3)).reshape(B, N, self.k_nn, 3) - p[:, :, None, :]
        loc = self.local(torch.cat([h[:, :, None, :].expand(-1, -1, self.k_nn, -1), nb, rel], -1)).max(2).values
        h = self.norm(h + loc)
        out = self.decoder(h, half_extents, pos_z)
        out["tokens"] = h
        return out


def build_encoder(name: str, **kw) -> nn.Module:
    if name == "patch":
        return PatchContactEncoder(**kw)
    if name == "point":
        return PointContactEncoder(**kw)
    if name == "analytic":
        from .analytic_baseline import AnalyticBoxPlaneEncoder
        return AnalyticBoxPlaneEncoder(**kw)
    raise ValueError(name)
