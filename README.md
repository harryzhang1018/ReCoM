# ReCoM — Reduced Contact Model (ContactROM)

Learned contact-information encoder for neural reduced dynamics (NeDM/NRD).
This repository implements **Study 1: Chrono box drop** from `docs/`:

* `docs/ContactROM_contact_encoder_high_level_plan.md` — the high-level plan
* `docs/ContactROM_box_drop_study_plan.md` — the first study case (this code base)

The pipeline separates the two learned modules of the plan:

```
box/ground geometry + pose  --E_contact-->  K=4 contact tokens  --F_NRD-->  next box state
        (recom.models.contact_encoder)                       (recom.models.transition)
```

## Environment

```bash
conda create -n recom python=3.12 -y
conda activate recom
conda install projectchrono::pychrono -c conda-forge -y      # PyChrono 10.0.0
pip install torch numpy scipy pyarrow pyyaml tqdm matplotlib pytest tensorboard
# or: conda env create -f environment.yml
python -m pytest tests -q
```

## Layout

| Path | Content |
| --- | --- |
| `recom/config.py` | Frozen configuration schema (`PhysicsConfig`, `EpisodeConfig`, `DatasetGenConfig`), `K_SLOTS=4`, `RAW_MAX=8` |
| `recom/geometry/` | `transforms` (quaternion utils, numpy+torch), `box_plane_analytic` (exact box–plane contacts, modes), `box_mesh` (12-triangle mesh, cached surface-patch tokens, surface points) |
| `recom/sim/` | `sampling` (deterministic SO(3)/height/geometry sampling, orientation strata), `chrono_box_drop` (Chrono NSC scene, `ReportContactCallback` recorder, settle logic) |
| `recom/data/` | `schema` (versioned record + time alignment), `canonicalize` (raw → canonical K-slot view + events), `storage` (npz/json per episode), `splits`, `validate` (dataset/replay tests), `dataset` (torch windows + balanced contact-query view), `nedm_export` (NeDM CSV layout) |
| `recom/models/` | `contact_encoder` (surface-patch encoder, PointNet-style baseline, K-slot set decoder), `analytic_baseline`, `losses` (Hungarian set loss), `nedm_adapter` (contact → NeDM features + pooling), `transition` (NeDM-compatible causal transformer with exact pose integration) |
| `recom/train/` | `train_contact` (Experiment C), `train_transition` (Experiments A/B/D/E), `rollout` (closed loop with gt / analytic / learned contact sources) |
| `recom/eval/` | `metrics` (contact + dynamics metrics), `visualize` (failure-case plots) |
| `scripts/` | `generate_dataset.py`, `validate_dataset.py`, `train_contact.py`, `train_transition.py`, `summarize_runs.py` |
| `configs/` | `smoke.yaml` (50 ep, fixed box), `fixed1a.yaml` (600 ep, Phase 1A), `pilot1b.yaml` (2000 ep, 240 box instances, Phase 1B) |
| `tests/` | geometry/contact unit tests, Chrono sign/timing conventions, dataset tests, model invariance tests |

## Quick start

```bash
# 1. data (deterministic; ~40 s for the 2000-episode pilot on 28 cores)
python scripts/generate_dataset.py configs/smoke.yaml
python scripts/generate_dataset.py configs/fixed1a.yaml
python scripts/generate_dataset.py configs/pilot1b.yaml
python scripts/validate_dataset.py data/pilot1b --replay 3      # schema, free flight, timing, replay, leakage

# 2. Experiment C: contact encoders (balanced query view, geometry holdout)
python scripts/train_contact.py --data data/pilot1b --encoder analytic --out runs/expC_analytic
python scripts/train_contact.py --data data/pilot1b --encoder patch    --out runs/expC_patch --steps 8000
python scripts/train_contact.py --data data/pilot1b --encoder point    --out runs/expC_point --steps 8000

# 3. Experiments A/B: state-only vs oracle-contact transition model (fixed box)
python scripts/train_transition.py --data data/fixed1a --contact-mode none     --out runs/expA_state_only --steps 30000
python scripts/train_transition.py --data data/fixed1a --contact-mode explicit --out runs/expB_oracle_explicit --steps 30000 --eval-contact-sources gt,analytic

# 4. Experiment D: learned contacts + frozen NeDM (evaluate B's model with the learned encoder in the loop)
python scripts/train_transition.py --data data/pilot1b --contact-mode explicit --out runs/expD --encoder-ckpt runs/expC_patch/final.pt --eval-contact-sources gt,analytic,learned

# 5. Experiment E: joint fine-tuning with unrolled loss (explicit + latent)
python scripts/train_transition.py --data data/pilot1b --contact-mode explicit+latent --train-contact-source learned --encoder-ckpt runs/expC_patch/final.pt --finetune-encoder --rollout-horizon 8 --out runs/expE

python scripts/summarize_runs.py runs      # markdown tables of final metrics
```

## Frozen conventions (Study 1)

* World Z up; ground plane z = 0 with normal +Z. Quaternions `(w, x, y, z)` with canonical sign `w >= 0`.
* **Time alignment**: `states[k]` is sampled before `DoStepDynamics` of step k; `contacts[k]` are the contacts of
  the collision pass at the *start* of step k (computed from `states[k]`, reported after the step through
  `ReportAllContacts`); `states[k+1]` is the post-step state. Learning record `(s_k, c_k) -> s_{k+1}`.
  Verified: reported `distance` equals the pre-step analytic gap.
* **Canonical contact**: A = box, B = ground; normal points ground → box; `d > 0` separated, `d < 0` penetrating;
  box-local point clamped to the half-extents (Bullet reports box points on the envelope-inflated shape laterally,
  exact along the normal); ground point = projection of the corrected box point onto the plane. Raw Chrono
  records (`raw_*`, incl. contact frame, reaction force/torque in the contact frame, constraint offset) are always kept.
* Chrono integrator relations hold exactly in the data (`pos[k+1] = pos[k] + dt v[k+1]`, `q[k+1] = exp(dt w[k+1]) q[k]`),
  so the transition model predicts only `(dv, dw)` (residual over the known gravity step) and integrates pose exactly.
* Physics (pilot): NSC, Bullet box–box, APGD (100 it, 1e-6), Euler implicit linearized, dt = 1 ms, envelope 1 mm,
  margin 0.5 mm, friction 0.5, restitution 0.3, density 1000, gravity 9.81, max 2 s, pose-based settle detection.

## Things learned while building (see `tests/test_chrono_conventions.py`)

* `ChSystem.SetCollisionSystemType()` **resets** `ChCollisionModel` default envelope/margin to 30 mm / 10 mm. Set
  the defaults after it and pin every model explicitly (the recorder asserts the actual values and stores them).
* Bullet box–box (with 1 mm envelope) is the most accurate pairing: `d` equals the analytic gap (median 0, max
  ~2 mm), normals exact, ≤ 4 contacts per pair. Triangle-mesh grounds give wrong normals on edges; convex hulls give
  wrong points.
* A box resting on 4 NSC contacts shows a deterministic period-4 rocking limit cycle (|v| ≈ 3 mm/s, |w| ≈ 0.08 rad/s,
  ±7 µm penetration, zero drift). Settling is therefore detected from pose change over a window, not velocity.
* Uniform SO(3) yields (15° rule) ≈ 35 % corner-, 55 % edge-, 10 % face-dominant releases; `orientation_strata`
  can rebalance.

## NeDM interface

`recom.models.transition.BoxTransitionModel` mirrors NeDM's `HMMWVDynamicsModel` (causal continuous-token transformer,
`states (B,T,S)` + context → normalized deltas). To train inside the original NeDM repo instead, export with
`python -m recom.data.nedm_export data/pilot1b <out>` and declare the contact block as NeDM `action_fields`.

## Status / next steps

Implemented: M0 (recorder + schema + replay tests), analytic baseline, patch/point encoders + set decoder,
state-only / oracle-contact / learned-contact / joint fine-tuning training paths, metrics, splits with geometry holdout.
Not yet implemented: neural-SDF encoder baseline (Section 11.2 item 3), Phase 1C dynamic-condition sets (config
supports `lin_vel_range`/`ang_vel_range`), policy-driven data aggregation (Stage F).
