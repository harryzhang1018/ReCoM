"""Export a ReCoM dataset in the NeDM collector layout so that NeDM's preprocess/trainer run unchanged.

Layout (see /home/harry/NeDM/src/nedm/training/preprocess.py):
    <out>/dataset_index.json          {dataset_name, created_utc, config, episodes:[{episode_id, split, scenario_family, rows, csv_path}]}
    <out>/episodes/<episode_id>.csv   one row per recorded step (rows = n_steps + 1)
    <out>/episodes/<episode_id>.json  per-episode metadata

Column naming follows NeDM's HMMWV BASE_FIELDS conventions (quat_e0..e3 scalar-first, Z-up).
Suggested NeDM field selection:
    state_fields   = vel_world_{x,y,z}_mps, ang_vel_world_{x,y,z}_radps  (+ pos_z_m, quat_e0..e3 if the net should see pose)
    action_fields  = the contact block c{k}_* below (oracle contacts at train time; encoder output at inference)
    rollout_fields = pos_{x,y,z}_m, quat_e0..e3
The contact block is the K-slot canonical set (active, d, n, p_box_local) flattened per slot.
"""
from __future__ import annotations

import csv
import datetime as _dt
from pathlib import Path

import numpy as np

from ..config import K_SLOTS, dump_json
from ..data.splits import load_splits
from ..data.storage import load_episode, read_index

STATE_COLS = ["time_s", "pos_x_m", "pos_y_m", "pos_z_m", "quat_e0", "quat_e1", "quat_e2", "quat_e3",
              "vel_world_x_mps", "vel_world_y_mps", "vel_world_z_mps", "ang_vel_world_x_radps", "ang_vel_world_y_radps", "ang_vel_world_z_radps",
              "half_extent_x_m", "half_extent_y_m", "half_extent_z_m", "settled"]


def contact_cols() -> list[str]:
    cols = []
    for k in range(K_SLOTS):
        cols += [f"c{k}_active", f"c{k}_d_m", f"c{k}_nx", f"c{k}_ny", f"c{k}_nz", f"c{k}_pbx_m", f"c{k}_pby_m", f"c{k}_pbz_m"]
    cols += ["n_contacts", "contact_mode", "analytic_min_gap_m"]
    return cols


def export_nedm(root: str, out: str, dataset_name: str = "recom_boxdrop") -> None:
    idx = read_index(root)
    splits = load_splits(Path(root) / "splits.json")
    split_of = {}
    for name in ("train", "val", "test", "test_geometry", "test_height"):
        for eid in splits.get(name, []):
            split_of[eid] = "train" if name == "train" else "val"  # NeDM knows train/val only; keep the fine split in metadata
    out_p = Path(out)
    (out_p / "episodes").mkdir(parents=True, exist_ok=True)
    entries = []
    ccols = contact_cols()
    for e in idx["episodes"]:
        eid = e["episode_id"]
        rec = load_episode(root, eid)
        S, C = rec.states, rec.canon
        n = rec.n_steps
        he = np.asarray(rec.meta["half_extents"])
        rows = []
        for k in range(n + 1):
            kk = min(k, n - 1)  # last row repeats the last contact (consumed only as a target by NeDM)
            row = [S["t"][k], *S["pos"][k], *S["quat"][k], *S["lin_vel"][k], *S["ang_vel_world"][k], *he, int(S["settled"][k])]
            for j in range(K_SLOTS):
                row += [int(C["c_active"][kk, j]), C["c_d"][kk, j], *C["c_n"][kk, j], *C["c_p_box_local"][kk, j]]
            row += [int(C["n_contacts"][kk]), int(C["contact_mode"][kk]), C["analytic_min_gap"][kk]]
            rows.append(row)
        csv_path = out_p / "episodes" / f"{eid}.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(STATE_COLS + ccols)
            w.writerows(rows)
        dump_json({"meta": rec.meta, "events": rec.events, "recom_split": next((s for s in splits if eid in set(splits[s])), None)}, out_p / "episodes" / f"{eid}.json")
        entries.append({"episode_id": eid, "split": split_of.get(eid, "val"), "scenario_family": f"approach_{e['approach_mode']}", "rows": n + 1, "csv_path": f"episodes/{eid}.csv"})
    dump_json({
        "dataset_name": dataset_name,
        "created_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "config": {"control_dt_s": idx["gen_config"]["physics"]["dt"], "recom_gen_config": idx["gen_config"]},
        "collection": {"source": "recom.sim.chrono_box_drop", "schema_version": "recom.boxdrop.v1"},
        "episodes": entries,
    }, out_p / "dataset_index.json")
    print(f"exported {len(entries)} episodes to {out_p}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("out")
    a = ap.parse_args()
    export_nedm(a.root, a.out)
