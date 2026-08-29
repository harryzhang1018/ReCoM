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
    p = {k: g(v) for k, v in pred.items() if k not in ("cardinality", "tokens", "slot_embedding")}
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


# ---- contact impulses (encoder-decoder plan, Section 9.4) ----------------------------------------
def impulse_frame_metrics(out: dict, target: dict, active: torch.Tensor, n: torch.Tensor, mu: torch.Tensor, label_thr: float = 1e-4) -> dict[str, np.ndarray]:
    """Per-frame raw quantities.  out: decoder output (dv_c, dL, dw_c, j_slot) with leading dims (...);
    target: target_dv_contact / target_dL_contact / target_dw_contact (...,3); active (...,K); n (...,K,3); mu (...)."""
    from ..models.impulse_decoder import cone_violation
    f = lambda x: x.reshape(-1, x.shape[-1]) if x.dim() > 1 else x.reshape(-1)  # noqa: E731
    dv_p, dL_p, dw_p = f(out["dv_c"]), f(out["dL"]), f(out["dw_c"])
    dv_t, dL_t, dw_t = f(target["target_dv_contact"]), f(target["target_dL_contact"]), f(target["target_dw_contact"])
    act = active.reshape(-1, active.shape[-1])
    K = act.shape[-1]
    pred_active = act.amax(-1) > 0
    label = dv_t.norm(dim=-1) > label_thr
    cosang = (dv_p * dv_t).sum(-1) / (dv_p.norm(dim=-1) * dv_t.norm(dim=-1)).clamp_min(1e-12)
    viol = cone_violation(out["j_slot"].reshape(-1, K, 3), n.reshape(-1, K, 3), mu.reshape(-1), act)
    res = {
        "label": label.float(), "pred_active": pred_active.float(),
        "dv_abs": (dv_p - dv_t).norm(dim=-1), "dL_abs": (dL_p - dL_t).norm(dim=-1), "dw_abs": (dw_p - dw_t).norm(dim=-1),
        "dv_rel": (dv_p - dv_t).norm(dim=-1) / dv_t.norm(dim=-1).clamp_min(1e-12), "dL_rel": (dL_p - dL_t).norm(dim=-1) / dL_t.norm(dim=-1).clamp_min(1e-12),
        "dir_err_deg": torch.rad2deg(torch.arccos(cosang.clamp(-1, 1))),
        "dv_pred_mag": dv_p.norm(dim=-1), "dv_target_mag": dv_t.norm(dim=-1), "dL_target_mag": dL_t.norm(dim=-1), "dw_target_mag": dw_t.norm(dim=-1),
        "cone_viol_frac": viol.sum(-1) / act.sum(-1).clamp_min(1),
    }
    return {k: _npy(v) for k, v in res.items()}


def aggregate_impulse_metrics(acc: dict[str, list], categories: np.ndarray | None = None, label_thr: float = 1e-4) -> dict:
    A = {k: np.concatenate(v) for k, v in acc.items()}

    def summ(m: np.ndarray) -> dict:
        lab, pa = A["label"][m] > 0, A["pred_active"][m] > 0
        ev = lab & pa                       # decoder is evaluated where the encoder activated and an impulse exists
        r = {"n": int(m.sum()), "n_label": int(lab.sum()), "n_pred_active": int(pa.sum()),
             "missed_impulse_rate": float((lab & ~pa).sum() / max(lab.sum(), 1)),
             "spurious_impulse_rate": float(((~lab) & pa & (A["dv_pred_mag"][m] > label_thr)).sum() / max((~lab).sum(), 1)),   # nonzero prediction on zero-label frames
             "cone_violation_rate": float(A["cone_viol_frac"][m][pa].mean()) if pa.any() else None}
        for k in ("dv", "dL", "dw"):
            r[f"{k}_mae"] = float(A[f"{k}_abs"][m][ev].mean()) if ev.any() else None
            r[f"{k}_p90"] = float(np.percentile(A[f"{k}_abs"][m][ev], 90)) if ev.any() else None
        # including encoder misses (error = full label) -> what the transition model actually sees
        r["dv_mae_incl_missed"] = float(np.where(pa, A["dv_abs"][m], A["dv_target_mag"][m])[lab].mean()) if lab.any() else None
        r["dv_rel_err_median"] = float(np.median(A["dv_rel"][m][ev])) if ev.any() else None
        r["dL_rel_err_median"] = float(np.median(A["dL_rel"][m][ev])) if ev.any() else None
        r["dir_err_deg_median"] = float(np.median(A["dir_err_deg"][m][ev])) if ev.any() else None
        r["zero_baseline_dv_mae"] = float(A["dv_target_mag"][m][ev].mean()) if ev.any() else None
        r["zero_baseline_dL_mae"] = float(A["dL_target_mag"][m][ev].mean()) if ev.any() else None
        r["zero_baseline_dw_mae"] = float(A["dw_target_mag"][m][ev].mean()) if ev.any() else None
        return r

    res = summ(np.ones(len(A["label"]), dtype=bool))
    if categories is not None:
        from ..data.dataset import CAT_NAMES
        res["by_category"] = {}
        for c, name in enumerate(CAT_NAMES):
            mc = categories == c
            if mc.any():
                res["by_category"][name] = summ(mc)
    return res


# ---- orientation symmetry (box) ---------------------------------------------------------------------
def settled_face_np(quat: np.ndarray) -> np.ndarray:
    """Which local face points down: index 2a (+axis a down) or 2a+1 (-axis a down), a in {x, y, z}.  (...,4) -> (...)."""
    from ..geometry.transforms import quat_to_rotmat_np
    R = quat_to_rotmat_np(quat)
    z = R[..., 2, :]                        # world-z component of each local axis
    a = np.abs(z).argmax(-1)
    s = np.take_along_axis(z, a[..., None], -1)[..., 0]
    return 2 * a + (s > 0).astype(np.int64)  # +axis pointing up (s>0) means the -axis face is down


def _symmetry_group(group: str) -> np.ndarray:
    import itertools
    if group == "d2":
        return np.stack([np.diag(d) for d in ([1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1])]).astype(np.float64)
    if group == "octahedral":
        mats = []
        for perm in itertools.permutations(range(3)):
            for signs in itertools.product([1, -1], repeat=3):
                M = np.zeros((3, 3))
                for i, p in enumerate(perm):
                    M[i, p] = signs[i]
                if np.linalg.det(M) > 0:
                    mats.append(M)
        return np.stack(mats)
    raise ValueError(group)


def symmetry_rot_err_deg_np(q_pred: np.ndarray, q_gt: np.ndarray, group: str = "d2") -> np.ndarray:
    """Orientation error modulo the box's rotational symmetries: min_g angle(R_gt^T R_pred g).
    'd2' (4 elements) is exact for a general cuboid; 'octahedral' (24) is exact only for a cube (shape-agnostic lower bound)."""
    from ..geometry.transforms import quat_to_rotmat_np
    Rp, Rg = quat_to_rotmat_np(q_pred), quat_to_rotmat_np(q_gt)
    rel = np.einsum("...ji,...jk->...ik", Rg, Rp)                     # R_gt^T R_pred
    G = _symmetry_group(group)
    tr = np.einsum("...ik,gki->...g", rel, G)                          # trace(rel @ g)
    ang = np.arccos(np.clip((tr - 1.0) / 2.0, -1.0, 1.0))
    return np.rad2deg(ang.min(-1))
