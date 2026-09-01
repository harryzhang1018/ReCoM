#!/usr/bin/env python
"""Closed-loop bottleneck ablation: attribute rollout drift to the pipeline stages
E (contact encoder: pose -> geometry), D (impulse decoder: geometry -> wrench),
N (NRD transformer: history + wrench -> residual delta).

Each cell swaps one stage for an oracle/analytic reference and rolls out the SAME frozen
transition model closed-loop; differences between cells are the closed-loop cost of the
swapped stage.

    cell      geometry   wrench          NRD residual
    FULL      learned    decoder         on            <- current neural sim (baseline)
    E-ANA     analytic   decoder         on            E cost = FULL - E-ANA
    E-GT      gt replay  decoder         on            open-loop geometry (NeDM-style)
    D-ORACLE  analytic   Chrono labels   on            D cost = E-ANA - D-ORACLE
    D-SOLVER  analytic   solver prior    on            learned decoder head value
    N-OFF     analytic   decoder         off           N contribution = E-ANA - N-OFF
    PHYS      analytic   Chrono labels   off           physics path alone (prior + gate*oracle)

Oracle-wrench cells replay the recorded per-step labels by time index: exact while the
predicted pose tracks Chrono, misaligned after divergence (same caveat as gt geometry replay).

    python scripts/ablate_bottleneck.py --split val --out results/ablate_bottleneck
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from recom.config import dump_json                                          # noqa: E402
from recom.data.dataset import CAT_NAMES                                    # noqa: E402
from recom.models.analytic_baseline import AnalyticBoxPlaneEncoder          # noqa: E402
from recom.models.impulse_decoder import (                                  # noqa: E402
    aggregate_wrench, broadcast_phys, chrono_gap, lever_arms, single_contact_solver, wrench_to_delta)
from recom.models.nedm_adapter import slot_features                         # noqa: E402
from recom.train.common import load_caches                                  # noqa: E402
from recom.train.rollout import (                                           # noqa: E402
    _stack_episodes, attach_wrench, contacts_from_encoder_output, episode_rollout_metrics, min_gap_np, summarize_rollouts)
from recom.train.train_impulse import load_decoder                          # noqa: E402
from recom.train.train_transition import load_encoder                       # noqa: E402
from recom.eval.metrics import rollout_errors                               # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_overlay_video import load_model                                   # noqa: E402

CELLS = {  # name -> (geometry, wrench, nrd)
    "FULL": ("learned", "decoder", True),
    "E-ANA": ("analytic", "decoder", True),
    "E-GT": ("gt", "decoder", True),
    "D-ORACLE": ("analytic", "oracle", True),
    "D-SOLVER": ("analytic", "solver", True),
    "N-OFF": ("analytic", "decoder", False),
    "PHYS": ("analytic", "oracle", False),
    "REPLAY": ("gt", "oracle", False),   # pure label replay with Chrono's own contact gate: integrator exactness bound
    "D-HYB": ("analytic", "hybrid", True),    # speed-gated wrench: decoder at impact speed, solver prior near rest
    "FULL-HYB": ("learned", "hybrid", True),  # the deployable form of the hybrid (learned geometry)
}
HYB_V0, HYB_W = 0.15, 0.05   # blend threshold ~ Chrono min_bounce_speed, sigmoid width (m/s)


def solver_wrench(contacts: dict, state: torch.Tensor, half_extents: torch.Tensor, phys: dict) -> torch.Tensor:
    """Physics-only wrench: the decoder's closed-form frictional single-contact impulse per active slot,
    aggregated (no learned head).  Multi-contact frames overcount (each slot solved as if alone)."""
    lead = state.shape[:-1]
    he = half_extents if half_extents.shape[:-1] == lead else broadcast_phys(half_extents, lead)
    active = contacts["active"]
    feats = slot_features({k: contacts[k] for k in ("active", "d", "n", "p_box_local")}, state, he)
    dt = broadcast_phys(phys["dt"], lead)
    env = broadcast_phys(phys["envelope"], lead)
    g_dt0 = broadcast_phys(phys["gravity"], lead) * dt
    d_ch = chrono_gap(feats, state, env)[..., 0]
    j_solver, _ = single_contact_solver(feats, state, broadcast_phys(phys["inertia_diag_over_m"], lead),
                                        broadcast_phys(phys["mu"], lead), broadcast_phys(phys["restitution"], lead), g_dt0, d_ch, dt)
    j = j_solver * active.unsqueeze(-1)
    r = lever_arms(state[..., 3:7], contacts["p_box_local"])
    dv_c, dL = aggregate_wrench(j, r, active)
    return wrench_to_delta(dv_c, dL, state[..., 3:7], broadcast_phys(phys["inertia_diag_over_m"], lead))


def stack_oracle_wrench(episodes, N: int, device) -> torch.Tensor:
    W = torch.zeros(len(episodes), N, 6)
    for i, ep in enumerate(episodes):
        n = ep.n_steps
        W[i, :n, :3] = torch.from_numpy(ep.impulse["target_dv_contact"][:n])
        W[i, :n, 3:] = torch.from_numpy(ep.impulse["target_dw_contact"][:n])
    return W.to(device)


@torch.no_grad()
def ablation_rollout(model, cell: str, encoder, decoder, episodes, device, T: int | None = None) -> np.ndarray:
    """rollout_batch with per-stage substitution; returns pred (B, N+1, 13)."""
    geom_kind, wrench_kind, use_nrd = CELLS[cell]
    T = T or model.block_size
    D = _stack_episodes(episodes, device)
    S, C, he, N, phys = D["states"], D["contacts"], D["half_extents"], D["N"], D["phys"]
    ana = AnalyticBoxPlaneEncoder(margin=0.002, latent_dim=1).to(device)
    OW = stack_oracle_wrench(episodes, N, device) if wrench_kind == "oracle" else None

    def contact(state, t):  # t = label index for gt geometry / oracle wrench
        if geom_kind == "gt":
            c = {k: v[:, min(t, N - 1)] for k, v in C.items()}
        else:
            enc = ana if geom_kind == "analytic" else encoder
            c = contacts_from_encoder_output(enc(he, state[:, 0:3], state[:, 3:7]), state[:, 0:3])
        if wrench_kind == "decoder":
            c = attach_wrench(c, state, he, phys, decoder)
        elif wrench_kind == "solver":
            c = dict(c)
            c["wrench"] = solver_wrench(c, state, he, phys)
        elif wrench_kind == "hybrid":
            ws = solver_wrench(c, state, he, phys)
            c = attach_wrench(c, state, he, phys, decoder)
            s = state[:, 7:10].norm(dim=-1) + state[:, 10:13].norm(dim=-1) * he.norm(dim=-1)
            w = torch.sigmoid((s - HYB_V0) / HYB_W).unsqueeze(-1)
            c["wrench"] = ws + w * (c["wrench"] - ws)
        else:
            c = dict(c)
            c["wrench"] = OW[:, min(t, N - 1)]
        return c

    pred = S.clone()
    hist_s = S[:, :T].clone()
    hist_c = [contact(hist_s[:, t], t) for t in range(T)]
    keys = [k for k in hist_c[0] if k in ("active", "d", "n", "p_box_local", "latent", "prob", "slot_embedding", "wrench")]
    hist_c = {k: torch.stack([c[k] for c in hist_c], 1) for k in keys}
    for t in range(T - 1, N):
        if use_nrd:
            delta = model.predict_delta(hist_s, he, hist_c)[:, -1]
        else:  # physics path only: prior + hard gate * wrench
            s_last = hist_s[:, -1]
            g = hist_c["active"][:, -1].amax(-1, keepdim=True).clamp(0, 1)
            delta = model.prior_delta(s_last, he) + g * hist_c["wrench"][:, -1]
        s_next = model.integrate(hist_s[:, -1], delta)
        pred[:, t + 1] = s_next
        c_next = contact(s_next, t + 1)
        hist_s = torch.cat([hist_s[:, 1:], s_next[:, None]], 1)
        hist_c = {k: torch.cat([hist_c[k][:, 1:], c_next[k][:, None]], 1) for k in keys}
    return pred.cpu().numpy()


def extra_metrics(pred: np.ndarray, ep, T: int, dt: float) -> dict:
    """Regime-resolved drift attribution + resting creep, appended to episode_rollout_metrics."""
    n = ep.n_steps
    gt, pr = ep.state[: n + 1].astype(np.float64), pred[: n + 1].astype(np.float64)
    err = rollout_errors(pr, gt)
    res = {}
    dpos = np.diff(err["pos_err"]) * 100.0    # cm accrued per step
    drot = np.diff(err["rot_err_deg"])
    cats = ep.category[:n]
    for c, name in enumerate(CAT_NAMES):
        m = (cats == c) & (np.arange(n) >= T)
        res[f"pos_accr_{name}"] = float(dpos[m].sum()) if m.any() else 0.0
        res[f"rot_accr_{name}"] = float(drot[m].sum()) if m.any() else 0.0
    rest = (cats == CAT_NAMES.index("resting")) & (np.arange(n) >= T)
    if rest.any():
        res["rest_creep_cm_s"] = float(np.linalg.norm(pr[:n][rest][:, 7:9], axis=1).mean() * 100)
        res["rest_creep_gt_cm_s"] = float(np.linalg.norm(gt[:n][rest][:, 7:9], axis=1).mean() * 100)
        res["rest_gap_err_mm"] = float(np.abs(min_gap_np(pr[:n][rest], ep.half_extents.astype(np.float64))).mean() * 1000)
    return res


@torch.no_grad()
def gate_confusion(pred: np.ndarray, episodes, encoder, device, T: int) -> dict:
    """Encoder vs analytic any-active agreement on the states the rollout actually visits."""
    ana = AnalyticBoxPlaneEncoder(margin=0.002, latent_dim=1).to(device)
    stats = {"gate_fp": 0, "gate_fn": 0, "n": 0, "gate_off_rest": 0, "n_rest": 0}
    for i, ep in enumerate(episodes):
        n = ep.n_steps
        s = torch.from_numpy(pred[i, T:n].astype(np.float32)).to(device)
        he = torch.from_numpy(ep.half_extents).to(device).expand(s.shape[0], 3)
        g_enc = (torch.sigmoid(encoder(he, s[:, 0:3], s[:, 3:7])["logit"]) > 0.5).any(-1)
        g_ana = (torch.sigmoid(ana(he, s[:, 0:3], s[:, 3:7])["logit"]) > 0.5).any(-1)
        stats["gate_fp"] += int((g_enc & ~g_ana).sum())
        stats["gate_fn"] += int((~g_enc & g_ana).sum())
        stats["n"] += int(g_enc.numel())
        rest = torch.from_numpy((ep.category[T:n] == CAT_NAMES.index("resting"))).to(device)
        stats["gate_off_rest"] += int((~g_enc & rest).sum())
        stats["n_rest"] += int(rest.sum())
    return {"gate_fp_rate": stats["gate_fp"] / max(stats["n"], 1), "gate_fn_rate": stats["gate_fn"] / max(stats["n"], 1),
            "gate_off_rest_rate": stats["gate_off_rest"] / max(stats["n_rest"], 1), "n_frames": stats["n"]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default="data/pilot1b")
    p.add_argument("--run", default="runs/local_ed3_jl6r_8k_v6_h8")
    p.add_argument("--split", default="val")
    p.add_argument("--cells", nargs="*", default=list(CELLS))
    p.add_argument("--decoder-ckpt", default=None, help="override the run's decoder (evaluate a retrained decoder closed-loop)")
    p.add_argument("--tag", default="", help="suffix for output files / summary keys (e.g. _v7rest)")
    p.add_argument("--max-episodes", type=int, default=0)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--out", default="results/ablate_bottleneck")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, cfg = load_model(Path(args.run), device)
    encoder, _ = load_encoder(cfg["encoder_ckpt"], device)
    decoder, _ = load_decoder(args.decoder_ckpt or cfg["decoder_ckpt"], device)
    cache = load_caches(args.data, [args.split], args.max_episodes or None)[args.split]
    eps, T, dt = cache.episodes, model.block_size, cfg["dt"]
    out = Path(args.out) / args.split
    out.mkdir(parents=True, exist_ok=True)

    summary = json.loads((out / "summary.json").read_text()) if (out / "summary.json").exists() else {}
    for cell in args.cells:
        rows, preds = [], []
        for s in range(0, len(eps), args.batch):
            chunk = eps[s: s + args.batch]
            pred = ablation_rollout(model, cell, encoder, decoder, chunk, device)
            preds.append((s, pred))
            for i, ep in enumerate(chunk):
                r = episode_rollout_metrics(pred[i], ep, T)
                r.update(extra_metrics(pred[i], ep, T, dt))
                rows.append(r)
            print(f"[{cell}] {min(s + args.batch, len(eps))}/{len(eps)} episodes", flush=True)
        key = cell + args.tag
        summary[key] = summarize_rollouts(rows)
        if CELLS[cell][0] == "learned":
            summary[key]["gate_confusion"] = gate_confusion(np.concatenate([p for _, p in preds]), eps, encoder, device, T)
        dump_json(rows, out / f"rows_{key}.json")
        # median error-vs-time curves (every 10 steps, over episodes still running)
        grid = np.arange(0, max(ep.n_steps for ep in eps) + 1, 10)
        curves = {}
        for key in ("pos_err", "rot_err_deg"):
            M = np.full((len(eps), len(grid)), np.nan)
            k = 0
            for s, pred in preds:
                for i, ep in enumerate(eps[s: s + pred.shape[0]]):
                    e = rollout_errors(pred[i, : ep.n_steps + 1], ep.state[: ep.n_steps + 1])[key]
                    g = grid[grid <= ep.n_steps]
                    M[k, : len(g)] = e[g]
                    k += 1
            curves[key] = np.nanmedian(M, axis=0)
            curves[f"{key}_n"] = np.isfinite(M).sum(0)
        np.savez(out / f"curves_{cell}{args.tag}.npz", grid=grid, **curves)

    dump_json(summary, out / "summary.json")
    keymetrics = ["pos_err@500", "pos_err_final", "rot_err_deg@500", "impact_dv_err", "impact_dw_err",
                  "max_penetration_pred", "final_ang_speed_err", "rest_creep_cm_s",
                  "pos_accr_resting", "pos_accr_first_impact", "pos_accr_contact", "pos_accr_rebound_repeat"]
    hdr = "cell".ljust(10) + "".join(k.rjust(22) for k in keymetrics)
    print("\n== median over episodes ==\n" + hdr)
    for cell in args.cells:
        row = summary[cell + args.tag]
        print((cell + args.tag).ljust(10) + "".join(f"{row[k]['median']:.4g}".rjust(22) if k in row else "-".rjust(22) for k in keymetrics))
    print(f"\nwrote {out}/summary.json")


if __name__ == "__main__":
    main()
