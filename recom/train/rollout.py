"""Closed-loop recurrent simulation:  state history -> transition model -> exact pose integration ->
contact source -> next contact token (Section 3).

contact_source:
    'gt'       : replay the recorded Chrono contacts (open loop w.r.t. the predicted pose; NeDM-style)
    'analytic' : exact box-plane contacts recomputed from the predicted pose (exact fallback)
    'learned'  : contact encoder evaluated on the predicted pose
"""
from __future__ import annotations

import numpy as np
import torch

from ..config import K_SLOTS
from ..data.dataset import CAT_NAMES, EpisodeArrays
from ..eval.metrics import rollout_errors
from ..geometry.box_plane_analytic import CORNER_SIGNS
from ..geometry.transforms import quat_to_rotmat
from ..models.analytic_baseline import AnalyticBoxPlaneEncoder

HORIZONS = (50, 100, 200, 500, 1000)


def contacts_from_encoder_output(out: dict[str, torch.Tensor], pos: torch.Tensor, threshold: float = 0.5, hard: bool = True) -> dict[str, torch.Tensor]:
    """Encoder output (B,K,...) -> adapter contact dict (B,K,...)."""
    prob = torch.sigmoid(out["logit"])
    active = (prob > threshold).float() if hard else prob
    return {"active": active, "d": out["d"], "n": out["n"], "p_box_local": out["p_box_local"], "latent": out["latent"], "prob": prob}


class ContactSource:
    def __init__(self, kind: str, encoder=None, margin: float = 0.002, latent_dim: int = 0) -> None:
        self.kind, self.encoder, self.latent_dim = kind, encoder, latent_dim
        self.analytic = AnalyticBoxPlaneEncoder(margin=margin, latent_dim=max(latent_dim, 1))

    def __call__(self, state: torch.Tensor, half_extents: torch.Tensor, gt_contacts: dict | None = None) -> dict[str, torch.Tensor]:
        if self.kind == "gt":
            assert gt_contacts is not None
            c = dict(gt_contacts)
            if self.latent_dim and "latent" not in c:
                c["latent"] = torch.zeros(*c["active"].shape, self.latent_dim, device=state.device)
            return c
        enc = self.analytic.to(state.device) if self.kind == "analytic" else self.encoder
        out = enc(half_extents, state[:, 0:3], state[:, 3:7])
        c = contacts_from_encoder_output(out, state[:, 0:3])
        if self.kind == "analytic":
            c["latent"] = torch.zeros(*c["active"].shape, max(self.latent_dim, 1), device=state.device)[..., : self.latent_dim] if self.latent_dim else c["latent"][..., :0]
        return c


def _stack_episodes(episodes: list[EpisodeArrays], device) -> dict:
    B = len(episodes)
    N = max(ep.n_steps for ep in episodes)
    states = torch.zeros(B, N + 1, 13)
    contacts = {"active": torch.zeros(B, N, K_SLOTS), "d": torch.zeros(B, N, K_SLOTS), "n": torch.zeros(B, N, K_SLOTS, 3), "p_box_local": torch.zeros(B, N, K_SLOTS, 3)}
    cats = np.full((B, N), -1, dtype=np.int64)
    he = torch.zeros(B, 3)
    lengths = torch.zeros(B, dtype=torch.long)
    for i, ep in enumerate(episodes):
        n = ep.n_steps
        states[i, : n + 1] = torch.from_numpy(ep.state)
        states[i, n + 1:] = torch.from_numpy(ep.state[-1])
        for k in contacts:
            contacts[k][i, :n] = torch.from_numpy(ep.contact[k])
        cats[i, :n] = ep.category
        he[i] = torch.from_numpy(ep.half_extents)
        lengths[i] = n
    return {"states": states.to(device), "contacts": {k: v.to(device) for k, v in contacts.items()}, "cats": cats, "half_extents": he.to(device), "lengths": lengths, "N": N}


@torch.no_grad()
def rollout_batch(model, source: ContactSource, episodes: list[EpisodeArrays], device, T: int | None = None) -> np.ndarray:
    """Return predicted states (B, N+1, 13); the first T states are the ground-truth priming context."""
    model.eval()
    T = T or model.block_size
    D = _stack_episodes(episodes, device)
    S, C, he, N = D["states"], D["contacts"], D["half_extents"], D["N"]
    B = S.shape[0]
    pred = S.clone()
    hist_s = S[:, :T].clone()
    gt_c = lambda t: {k: v[:, t] for k, v in C.items()}  # noqa: E731
    hist_c = [source(hist_s[:, t], he, gt_c(t)) for t in range(T)]
    keys = list(hist_c[0].keys())
    hist_c = {k: torch.stack([c[k] for c in hist_c], 1) for k in keys}
    for t in range(T - 1, N):
        delta = model.predict_delta(hist_s, he, hist_c if model.contact_mode != "none" else None)[:, -1]
        s_next = model.integrate(hist_s[:, -1], delta)
        pred[:, t + 1] = s_next
        c_next = source(s_next, he, gt_c(min(t + 1, N - 1)))
        hist_s = torch.cat([hist_s[:, 1:], s_next[:, None]], 1)
        hist_c = {k: torch.cat([hist_c[k][:, 1:], c_next[k][:, None]], 1) for k in keys}
    return pred.cpu().numpy()


def box_energy(state: np.ndarray, mass: float, inertia: np.ndarray, g: float = 9.81) -> np.ndarray:
    v, w = state[:, 7:10], state[:, 10:13]
    from ..geometry.transforms import quat_to_rotmat_np
    R = quat_to_rotmat_np(state[:, 3:7])
    wl = np.einsum("nji,nj->ni", R, w)
    return 0.5 * mass * (v**2).sum(1) + 0.5 * (inertia * wl**2).sum(1) + mass * g * state[:, 2]


def min_gap_np(state: np.ndarray, he: np.ndarray) -> np.ndarray:
    from ..geometry.transforms import quat_to_rotmat_np
    R = quat_to_rotmat_np(state[:, 3:7])
    c = np.einsum("nij,kj->nki", R, CORNER_SIGNS * he) + state[:, None, 0:3]
    return c[..., 2].min(1)


def episode_rollout_metrics(pred: np.ndarray, ep: EpisodeArrays, T: int, impact_window: int = 5) -> dict:
    n = ep.n_steps
    gt = ep.state[: n + 1]
    pr = pred[: n + 1]
    err = rollout_errors(pr[T:], gt[T:])
    res = {"episode_id": ep.episode_id, "n_steps": n}
    for h in HORIZONS:
        if T + h <= n:
            for k, v in err.items():
                res[f"{k}@{h}"] = float(v[h])
    for k, v in err.items():
        res[f"{k}_final"] = float(v[-1])
        res[f"{k}_mean"] = float(v.mean())
    # regime breakdown (GT categories)
    cats = ep.category[T:n]
    for c, name in enumerate(CAT_NAMES):
        m = cats == c
        if m.any():
            res[f"v_err_{name}"] = float(err["v_err"][:-1][m].mean())
            res[f"w_err_{name}"] = float(err["w_err"][:-1][m].mean())
    fi = ep.events["first_impact_step"]
    if fi >= T and fi + impact_window <= n:
        dv_gt = gt[fi + impact_window, 7:10] - gt[fi - 1, 7:10]
        dv_pr = pr[fi + impact_window, 7:10] - pr[fi - 1, 7:10]
        res["impact_dv_err"] = float(np.linalg.norm(dv_gt - dv_pr))
        res["impact_dv_gt"] = float(np.linalg.norm(dv_gt))
        dw_gt = gt[fi + impact_window, 10:13] - gt[fi - 1, 10:13]
        dw_pr = pr[fi + impact_window, 10:13] - pr[fi - 1, 10:13]
        res["impact_dw_err"] = float(np.linalg.norm(dw_gt - dw_pr))
        # rebound apex (max COM height after first impact)
        res["apex_err"] = float(abs(pr[fi:, 2].max() - gt[fi:, 2].max()))
        res["apex_time_err_steps"] = float(abs(pr[fi:, 2].argmax() - gt[fi:, 2].argmax()))
    he = ep.half_extents.astype(np.float64)
    res["max_penetration_pred"] = float(max(0.0, -min_gap_np(pr[T:], he).min()))
    res["max_penetration_gt"] = float(max(0.0, -min_gap_np(gt[T:], he).min()))
    mass, inertia = ep.meta["mass"], np.asarray(ep.meta["inertia_xx"])
    E_pr, E_gt = box_energy(pr[T:], mass, inertia), box_energy(gt[T:], mass, inertia)
    res["artificial_energy_max"] = float(np.maximum.accumulate(E_pr).max() - E_pr[0]) if len(E_pr) else 0.0
    res["energy_gt_initial"] = float(E_gt[0])
    res["final_rot_err_deg"] = float(err["rot_err_deg"][-1])
    return res


def summarize_rollouts(rows: list[dict]) -> dict:
    keys = sorted({k for r in rows for k in r if isinstance(r[k], (int, float)) and k != "n_steps"})
    out = {}
    for k in keys:
        vals = np.array([r[k] for r in rows if k in r], dtype=np.float64)
        if len(vals):
            out[k] = {"mean": float(vals.mean()), "median": float(np.median(vals)), "p90": float(np.percentile(vals, 90)), "n": int(len(vals))}
    return out


def evaluate_rollouts(model, source: ContactSource, cache, device, batch_size: int = 64, T: int | None = None) -> tuple[dict, list[dict]]:
    T = T or model.block_size
    rows = []
    eps = cache.episodes
    for s in range(0, len(eps), batch_size):
        chunk = eps[s: s + batch_size]
        pred = rollout_batch(model, source, chunk, device, T)
        for i, ep in enumerate(chunk):
            rows.append(episode_rollout_metrics(pred[i], ep, T))
    return summarize_rollouts(rows), rows
