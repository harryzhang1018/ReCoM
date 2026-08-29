#!/usr/bin/env python
"""Stage ED0: audit contact-impulse labels on a recorded dataset.

Compares, per split and per frame category, the state-derived net wrench (J, L) [frozen formulas in
recom/data/impulse_targets.py] with the force-derived wrench integrated from Chrono's recorded per-contact
forces (raw contact points / canonical clamped points) and with the naive world angular-momentum difference.
Also reports inactive-frame residuals, the zero-force active-slot fraction and mass-normalized magnitudes.

    python scripts/audit_impulse_labels.py --data data/pilot1b --out results/audit_impulse_labels/pilot1b
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from recom.config import dump_json  # noqa: E402
from recom.data.dataset import CAT_NAMES, frame_categories  # noqa: E402
from recom.data.impulse_targets import LABEL_VERSION, force_derived_labels, impulse_targets_from_record, naive_dH_world, phys_from_meta, state_derived_wrench  # noqa: E402
from recom.data.splits import load_splits  # noqa: E402
from recom.data.storage import load_episode  # noqa: E402

EPS = 1e-12


def rel_err(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.linalg.norm(a - b, axis=-1) / (np.linalg.norm(b, axis=-1) + EPS)


def pct(x: np.ndarray, qs=(50, 90, 100)) -> dict:
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return {f"p{q}": None for q in qs}
    return {f"p{q}": float(np.percentile(x, q)) if q < 100 else float(x.max()) for q in qs}


def audit_split(root: str, ids: list[str], max_episodes: int | None) -> dict:
    acc: dict[str, list] = {}
    masses = []
    for eid in ids[:max_episodes] if max_episodes else ids:
        rec = load_episode(root, eid)
        N = rec.n_steps
        ph = phys_from_meta(rec.meta)
        masses.append(ph["mass"])
        cat = frame_categories(rec)
        J_s, L_s = state_derived_wrench(rec)
        J_r, L_r, j_slot = force_derived_labels(rec, "raw")
        _, L_c, _ = force_derived_labels(rec, "canonical")
        L_n = naive_dH_world(rec.states["ang_vel_local"][: N + 1], rec.states["quat"][: N + 1], ph["inertia_diag"])
        tg = impulse_targets_from_record(rec)
        active = rec.canon["c_active"][:N].astype(bool)
        nc = rec.canon["n_contacts"][:N]
        rows = {
            "category": cat, "n_contacts": nc,
            "relJ_force": rel_err(J_r, J_s), "relL_raw": rel_err(L_r, L_s), "relL_canon": rel_err(L_c, L_s), "relL_naive": rel_err(L_n, L_s),
            "absJ_force_over_m": np.linalg.norm(J_r - J_s, axis=-1) / ph["mass"], "absL_canon_over_m": np.linalg.norm(L_c - L_s, axis=-1) / ph["mass"],
            "dv_mag": np.linalg.norm(tg["target_dv_contact"], axis=-1), "dL_mag": np.linalg.norm(tg["target_dL_contact"], axis=-1),
            "dw_mag": np.linalg.norm(tg["target_dw_contact"], axis=-1),
            "inactive_dv": np.where(nc == 0, np.linalg.norm(tg["target_dv_contact"], axis=-1), np.nan),
            "inactive_dw": np.where(nc == 0, np.linalg.norm(tg["target_dw_contact"], axis=-1), np.nan),
            "zero_force_slots": np.array([(np.linalg.norm(j_slot[k][active[k]], axis=-1) == 0).sum() for k in range(N)], dtype=np.float64),
            "active_slots": active.sum(1).astype(np.float64),
            "jslot_mag": np.where(active, np.linalg.norm(tg["target_j_slot"], axis=-1), np.nan).max(1),
        }
        for k, v in rows.items():
            acc.setdefault(k, []).append(np.asarray(v))
    A = {k: np.concatenate(v) for k, v in acc.items()}
    out = {"n_episodes": len(masses), "n_frames": int(len(A["category"])), "mass_min": float(min(masses)), "mass_max": float(max(masses)),
           "mass_ratio": float(max(masses) / min(masses))}
    contact = A["n_contacts"] > 0
    out["overall"] = _summ(A, np.ones_like(contact), contact)
    out["by_category"] = {}
    for c, name in enumerate(CAT_NAMES):
        m = A["category"] == c
        if m.any():
            out["by_category"][name] = _summ(A, m, m & contact)
    return out


def _summ(A: dict, m: np.ndarray, mc: np.ndarray) -> dict:
    inactive = m & (A["n_contacts"] == 0)
    label = mc & (A["dv_mag"] > 1e-4)
    res = {
        "n": int(m.sum()), "n_contact": int(mc.sum()), "n_label": int(label.sum()),
        "relJ_force": pct(A["relJ_force"][label]), "relL_raw": pct(A["relL_raw"][label]), "relL_canon": pct(A["relL_canon"][label]), "relL_naive": pct(A["relL_naive"][label]),
        "absJ_force_over_m_max": float(A["absJ_force_over_m"][mc].max()) if mc.any() else None,
        "dv_mag": pct(A["dv_mag"][mc]), "dL_mag": pct(A["dL_mag"][mc]), "dw_mag": pct(A["dw_mag"][mc]), "jslot_mag": pct(A["jslot_mag"][mc]),
        "inactive_dv_max": float(np.nanmax(A["inactive_dv"][inactive])) if inactive.any() else None,
        "inactive_dw_max": float(np.nanmax(A["inactive_dw"][inactive])) if inactive.any() else None,
        "zero_force_slot_frac": float(A["zero_force_slots"][mc].sum() / max(A["active_slots"][mc].sum(), 1)) if mc.any() else None,
        "zero_label_contact_frame_frac": float(((A["dv_mag"] <= 1e-4) & mc).sum() / max(mc.sum(), 1)) if mc.any() else None,
    }
    return res


def to_markdown(report: dict) -> str:
    lines = [f"# Impulse-label audit ({LABEL_VERSION})", ""]
    for split, r in report["splits"].items():
        lines += [f"## {split}: {r['n_episodes']} episodes, {r['n_frames']} frames, mass {r['mass_min']:.3f}-{r['mass_max']:.2f} kg ({r['mass_ratio']:.0f}x)", "",
                  "| category | n | n_contact | relJ force p50/max | relL raw p50/max | relL canonical p50/p90 | relL naive dH p50/max | inactive max dv | zero-force slot frac | dv p50/p90/max [m/s] | dw p50/p90/max [rad/s] |",
                  "|---|---|---|---|---|---|---|---|---|---|---|"]
        rows = [("overall", r["overall"])] + list(r["by_category"].items())
        for name, s in rows:
            f = lambda d, a, b: f"{d[a]:.1e}/{d[b]:.1e}" if d[a] is not None else "-"  # noqa: E731
            g = lambda d: f"{d['p50']:.3g}/{d['p90']:.3g}/{d['p100']:.3g}" if d["p50"] is not None else "-"  # noqa: E731
            lines.append(f"| {name} | {s['n']} | {s['n_contact']} | {f(s['relJ_force'], 'p50', 'p100')} | {f(s['relL_raw'], 'p50', 'p100')} | {f(s['relL_canon'], 'p50', 'p90')} | {f(s['relL_naive'], 'p50', 'p100')} | "
                         f"{'-' if s['inactive_dv_max'] is None else f'{s['inactive_dv_max']:.1e}'} | {'-' if s['zero_force_slot_frac'] is None else f'{s['zero_force_slot_frac']:.3f}'} | {g(s['dv_mag'])} | {g(s['dw_mag'])} |")
        lines.append("")
    return "\n".join(lines)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/pilot1b")
    ap.add_argument("--splits", default="train,val,test,test_geometry")
    ap.add_argument("--max-episodes", type=int, default=None)
    ap.add_argument("--out", default=None, help="output prefix (default results/audit_impulse_labels/<dataset name>)")
    a = ap.parse_args(argv)
    out = Path(a.out or f"results/audit_impulse_labels/{Path(a.data).name}")
    sp = load_splits(Path(a.data) / "splits.json")
    report = {"label_version": LABEL_VERSION, "data": a.data, "splits": {}}
    for name in a.splits.split(","):
        ids = sp.get(name, [])
        if not ids:
            continue
        report["splits"][name] = audit_split(a.data, ids, a.max_episodes)
        print(f"[{name}] " + " ".join(f"{k}={v}" for k, v in report["splits"][name]["overall"].items() if not isinstance(v, dict)), flush=True)
    dump_json(report, out.with_suffix(".json"))
    md = to_markdown(report)
    out.with_suffix(".md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
