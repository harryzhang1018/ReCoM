"""Evaluation metrics (Section 16)."""
from __future__ import annotations

import numpy as np
import torch

from ..geometry.transforms import quat_angle_np, quat_conj_np, quat_mul_np


def _npy(x):
    return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)


def contact_frame_metrics(pred: dict, gt: dict, scale, perm=None) -> dict[str, np.ndarray]:
    """Per-sample raw quantities (to be aggregated).  pred/gt: batched dicts (after Hungarian matching if perm given).
    Returns arrays: tp/fp/fn counts per sample, |d| error, point error (normalized by smallest half-extent),
    normal angle (deg), cardinality correct, predicted probability & correctness for calibration."""
    from ..models.losses import hungarian_match
    if perm is None:
        perm = hungarian_match(pred, gt, scale)
    B, K = gt["active"].shape
    g = lambda x: torch.gather(x, 1, perm.view(B, K, *([1] * (x.dim() - 2))).expand(B, K, *x.shape[2:]))  # noqa: E731
    p = {k: g(v) for k, v in pred.items() if k not in ("cardinality", "tokens")}
    prob = torch.sigmoid(p["logit"])
    pa, ga = (prob > 0.5).float(), gt["active"]
    s = scale.view(B, 1)
    out = {
        "tp": (pa * ga).sum(1), "fp": (pa * (1 - ga)).sum(1), "fn": ((1 - pa) * ga).sum(1),
        "frame_gt_active": (ga.sum(1) > 0).float(), "frame_pred_active": (pa.sum(1) > 0).float(),
        "d_abs_err": ((p["d"] - gt["d"]).abs() * ga).sum(1) / ga.sum(1).clamp_min(1),
        "point_err_norm": (((p["p_box_local"] - gt["p_box_local"]).norm(dim=-1) / s) * ga).sum(1) / ga.sum(1).clamp_min(1),
        "ground_point_err_norm": (((p["p_ground_rel"] - gt["p_ground_rel"]).norm(dim=-1) / s) * ga).sum(1) / ga.sum(1).clamp_min(1),
        "normal_deg": (torch.rad2deg(torch.arccos((p["n"] * gt["n"]).sum(-1).clamp(-1, 1))) * ga).sum(1) / ga.sum(1).clamp_min(1),
        "n_gt": ga.sum(1),
        "card_correct": (pred["cardinality"].argmax(-1) == gt["n_contacts"]).float() if "cardinality" in pred else torch.zeros(B),
        "count_correct": (pa.sum(1) == ga.sum(1)).float(),
        "prob": prob.reshape(-1), "prob_target": ga.reshape(-1),
        "log_var": p.get("log_var", torch.zeros(B, K)).reshape(-1), "slot_point_err": ((p["p_box_local"] - gt["p_box_local"]).norm(dim=-1) / s).reshape(-1), "slot_active": ga.reshape(-1),
    }
    return {k: _npy(v) for k, v in out.items()}


def aggregate_contact_metrics(acc: dict[str, list], categories: np.ndarray | None = None) -> dict:
    A = {k: np.concatenate(v) for k, v in acc.items()}
    has = A["n_gt"] > 0
    tp, fp, fn = A["tp"].sum(), A["fp"].sum(), A["fn"].sum()
    ftp = (A["frame_gt_active"] * A["frame_pred_active"]).sum()
    res = {
        "slot_precision": float(tp / max(tp + fp, 1)), "slot_recall": float(tp / max(tp + fn, 1)),
        "frame_precision": float(ftp / max(A["frame_pred_active"].sum(), 1)), "frame_recall": float(ftp / max(A["frame_gt_active"].sum(), 1)),
        "d_mae": float(A["d_abs_err"][has].mean()) if has.any() else None,
        "d_p95": float(np.percentile(A["d_abs_err"][has], 95)) if has.any() else None,
        "point_err_median_pct_min_dim": float(100 * np.median(A["point_err_norm"][has]) / 2) if has.any() else None,   # % of smallest side (= 2*he_min)
        "point_err_p95_pct_min_dim": float(100 * np.percentile(A["point_err_norm"][has], 95) / 2) if has.any() else None,
        "ground_point_err_median_pct_min_dim": float(100 * np.median(A["ground_point_err_norm"][has]) / 2) if has.any() else None,
        "normal_deg_median": float(np.median(A["normal_deg"][has])) if has.any() else None,
        "normal_deg_p95": float(np.percentile(A["normal_deg"][has], 95)) if has.any() else None,
        "cardinality_acc": float(A["card_correct"].mean()), "count_acc": float(A["count_correct"].mean()),
        "ece": expected_calibration_error(A["prob"], A["prob_target"]),
        "n_frames": int(len(A["n_gt"])),
    }
    # error-vs-uncertainty: Spearman-like rank correlation between predicted log-var and point error on active slots
    m = A["slot_active"] > 0
    if m.sum() > 10:
        lv, e = A["log_var"][m], A["slot_point_err"][m]
        res["uncertainty_error_corr"] = float(np.corrcoef(np.argsort(np.argsort(lv)), np.argsort(np.argsort(e)))[0, 1])
    if categories is not None:
        from ..data.dataset import CAT_NAMES
        res["by_category"] = {}
        for c, name in enumerate(CAT_NAMES):
            mc = categories == c
            if mc.sum() == 0:
                continue
            h = has & mc
            res["by_category"][name] = {
                "n": int(mc.sum()),
                "frame_recall": float((A["frame_gt_active"][mc] * A["frame_pred_active"][mc]).sum() / max(A["frame_gt_active"][mc].sum(), 1)),
                "false_positive_rate": float((A["frame_pred_active"][mc] * (1 - A["frame_gt_active"][mc])).sum() / max((1 - A["frame_gt_active"][mc]).sum(), 1)),
                "d_mae": float(A["d_abs_err"][h].mean()) if h.any() else None,
                "point_err_median_pct_min_dim": float(100 * np.median(A["point_err_norm"][h]) / 2) if h.any() else None,
            }
    return res


def expected_calibration_error(prob: np.ndarray, target: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (prob >= lo) & (prob < hi) if hi < 1 else (prob >= lo) & (prob <= hi)
        if m.any():
            ece += m.mean() * abs(prob[m].mean() - target[m].mean())
    return float(ece)


def first_impact_timing(pred_active_frames: np.ndarray, gt_first_step: int) -> float | None:
    """pred_active_frames: (N,) bool per frame. Returns error in steps (pred - gt) or None."""
    idx = np.where(pred_active_frames)[0]
    if gt_first_step < 0 or len(idx) == 0:
        return None
    return float(idx[0] - gt_first_step)


# ---- dynamics --------------------------------------------------------------------------
def rollout_errors(pred: np.ndarray, gt: np.ndarray) -> dict[str, np.ndarray]:
    """pred/gt (T, 13) -> per-step error arrays."""
    dq = quat_mul_np(pred[:, 3:7], quat_conj_np(gt[:, 3:7]))
    return {
        "pos_err": np.linalg.norm(pred[:, 0:3] - gt[:, 0:3], axis=1),
        "rot_err_deg": np.rad2deg(quat_angle_np(dq)),
        "v_err": np.linalg.norm(pred[:, 7:10] - gt[:, 7:10], axis=1),
        "w_err": np.linalg.norm(pred[:, 10:13] - gt[:, 10:13], axis=1),
    }
