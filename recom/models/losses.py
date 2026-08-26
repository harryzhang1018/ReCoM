"""Set-based contact losses (Section 13): Hungarian matching of K predicted slots to GT contacts."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from ..config import K_SLOTS


def focal_bce_with_logits(logits: torch.Tensor, targets: torch.Tensor, alpha: float = 0.5, gamma: float = 2.0) -> torch.Tensor:
    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    a_t = alpha * targets + (1 - alpha) * (1 - targets)
    return (a_t * (1 - p_t) ** gamma * ce).mean()


@torch.no_grad()
def hungarian_match(pred: dict[str, torch.Tensor], gt: dict[str, torch.Tensor], scale: torch.Tensor,
                    w_point: float = 1.0, w_d: float = 1.0, w_act: float = 0.5) -> torch.Tensor:
    """Return perm (B, K): perm[b, j] = index of the predicted slot assigned to GT slot j.

    GT slots that are inactive are matched to the leftover predicted slots (they only receive the
    'inactive' classification loss).  Cost uses box-local points normalized by the smallest half-extent.
    """
    B, K = gt["active"].shape
    s = scale.view(B, 1, 1, 1)
    pb = (pred["p_box_local"][:, :, None, :] - gt["p_box_local"][:, None, :, :]).abs().sum(-1) / s.view(B, 1, 1)   # (B, Kp, Kg)
    pg = (pred["p_ground_rel"][:, :, None, :] - gt["p_ground_rel"][:, None, :, :]).abs().sum(-1) / s.view(B, 1, 1)
    dd = (pred["d"][:, :, None] - gt["d"][:, None, :]).abs() / s.view(B, 1, 1)
    act = -torch.log_softmax(torch.stack([torch.zeros_like(pred["logit"]), pred["logit"]], -1), -1)[..., 1][:, :, None].expand(B, K, K)
    cost = w_point * (pb + pg) + w_d * dd + w_act * act
    cost = cost * gt["active"][:, None, :] + (1.0 - gt["active"][:, None, :]) * 0.0
    perm = torch.zeros(B, K, dtype=torch.long, device=cost.device)
    cost_np = cost.detach().cpu().numpy()
    for b in range(B):
        r, c = linear_sum_assignment(cost_np[b].T)  # rows = gt, cols = pred
        perm[b, r] = torch.as_tensor(c, device=cost.device)
    return perm


def contact_set_loss(pred: dict[str, torch.Tensor], gt: dict[str, torch.Tensor], scale: torch.Tensor, weights: dict | None = None) -> tuple[torch.Tensor, dict[str, float]]:
    """Weighted multi-task loss; pred/gt keys: logit/active, d, p_box_local, p_ground_rel, n, (cardinality logits), log_var.
    `scale` = smallest half-extent per sample (B,) used to normalize point/distance errors."""
    w = {"active": 2.0, "d": 5.0, "points": 5.0, "normal": 1.0, "cardinality": 0.5, "uncertainty": 0.2}
    if weights:
        w.update(weights)
    B, K = gt["active"].shape
    perm = hungarian_match(pred, gt, scale)
    gather = lambda x: torch.gather(x, 1, perm.view(B, K, *([1] * (x.dim() - 2))).expand(B, K, *x.shape[2:]))  # noqa: E731
    p = {k: gather(v) for k, v in pred.items() if k not in ("cardinality",)}
    act = gt["active"]
    m = act.sum().clamp_min(1.0)
    s = scale.view(B, 1)
    l_act = focal_bce_with_logits(p["logit"], act)
    l_d = (F.huber_loss(p["d"] / s, gt["d"] / s, reduction="none", delta=1.0) * act).sum() / m
    l_pb = ((p["p_box_local"] - gt["p_box_local"]).abs().sum(-1) / s * act).sum() / m
    l_pg = ((p["p_ground_rel"] - gt["p_ground_rel"]).abs().sum(-1) / s * act).sum() / m
    l_n = ((1.0 - F.cosine_similarity(p["n"], gt["n"], dim=-1)) * act).sum() / m
    l_card = F.cross_entropy(pred["cardinality"], gt["n_contacts"]) if "cardinality" in pred else torch.zeros((), device=act.device)
    # heteroscedastic uncertainty on the (normalized) box point error
    if "log_var" in p:
        err2 = (((p["p_box_local"] - gt["p_box_local"]) / s.unsqueeze(-1)) ** 2).sum(-1).detach()
        l_unc = ((0.5 * torch.exp(-p["log_var"]) * err2 + 0.5 * p["log_var"]) * act).sum() / m
    else:
        l_unc = torch.zeros((), device=act.device)
    total = w["active"] * l_act + w["d"] * l_d + w["points"] * (l_pb + l_pg) + w["normal"] * l_n + w["cardinality"] * l_card + w["uncertainty"] * l_unc
    parts = {"active": l_act.item(), "d": l_d.item(), "p_box": l_pb.item(), "p_ground": l_pg.item(), "normal": l_n.item(), "cardinality": l_card.item(), "uncertainty": l_unc.item()}
    return total, parts
