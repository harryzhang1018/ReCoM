# ReCoM Study 1 — Progress Log and Resume Guide

Last updated: 2026-08-26 (America/Chicago). Written so the next session can pick up without re-deriving anything.

## 0. TL;DR / where we stopped

* The Study-1 code base (box drop in Chrono → canonical contact schema → contact encoders → NeDM-compatible
  transition model → closed-loop rollouts) is implemented, tested (28/28 pytest), and three validated datasets exist.
* Experiments A/B (state-only vs oracle-contact transition, fixed box) ran to completion and show the H1 sign of life.
* Experiment C (contact encoders on the variable-geometry pilot) was **interrupted**: four trainers were launched
  concurrently, the machine wedged (load avg ~285, GPU/driver hang, `ps`/`nvidia-smi` unresponsive) and needs a
  reboot. Two bugs found during those runs are already fixed in the code (see §6).
* **Next action after reboot:** run §7 commands one at a time.

## 1. Environment

```bash
conda create -n recom python=3.12 -y && conda activate recom
conda install projectchrono::pychrono -c conda-forge -y       # pychrono 10.0.0
pip install torch numpy scipy pyarrow pyyaml tqdm matplotlib pytest tensorboard   # torch 2.13+cu130 was installed
# in scripts:  source /home/harry/anaconda3/etc/profile.d/conda.sh && conda activate recom
python -m pytest tests -q          # 28 passed
```

Machine: 32 cores, 62 GB RAM, RTX 4090. **Run only one training job at a time** with
`OMP_NUM_THREADS=4 MKL_NUM_THREADS=4` and `--workers 2` (see §6, item 3).

## 2. What is implemented (all under `/home/harry/ReCoM`)

| Area | Files | Status |
| --- | --- | --- |
| Config / schema | `recom/config.py` (`PhysicsConfig`, `EpisodeConfig`, `DatasetGenConfig`, `K_SLOTS=4`, `RAW_MAX=8`), `recom/data/schema.py` | done |
| Geometry | `recom/geometry/transforms.py` (quat utils np+torch), `box_plane_analytic.py` (exact box–plane contacts, modes, ballistic checks), `box_mesh.py` (12-tri mesh, surface-patch tokens 34-D, surface points) | done, unit-tested |
| Chrono recorder | `recom/sim/chrono_box_drop.py` (`BoxDropScene`, `ContactReporter(ReportContactCallback)`, `run_episode`), `recom/sim/sampling.py` (deterministic SO(3)/clearance/geometry sampling, orientation strata, coverage diagnostic) | done, conventions tested |
| Data | `recom/data/canonicalize.py` (raw→canonical K-slot view + events), `storage.py` (npz+json per episode, `index.json`, `splits.json`), `splits.py` (episode-level, geometry-group holdout), `validate.py` (schema, free flight, timing, replay), `dataset.py` (`TransitionWindowDataset`, balanced `ContactQueryDataset`, categories), `nedm_export.py` (NeDM CSV layout) | done |
| Models | `recom/models/contact_encoder.py` (`PatchContactEncoder`, `PointContactEncoder`, `SetDecoder` K=4, `GeometryCache`), `analytic_baseline.py`, `losses.py` (Hungarian set loss, focal/huber/cosine/cardinality/uncertainty), `nedm_adapter.py` (19-D per-slot features + DeepSets pooling), `transition.py` (`BoxTransitionModel`: causal transformer, gravity prior, contact gate, exact pose integration) | done |
| Training / eval | `recom/train/train_contact.py` (Exp C), `train_transition.py` (Exp A/B/D/E incl. unrolled loss, `--finetune-encoder`), `rollout.py` (contact sources `gt|analytic|learned`, regime metrics), `recom/eval/metrics.py`, `visualize.py` | done (E path only smoke-tested) |
| Scripts / configs | `scripts/generate_dataset.py`, `validate_dataset.py`, `train_contact.py`, `train_transition.py`, `summarize_runs.py`; `configs/smoke.yaml`, `fixed1a.yaml`, `pilot1b.yaml` | done |
| Docs | `README.md` (layout, conventions, quick start), this file | done |

Not implemented: neural-SDF encoder baseline (plan §11.2 item 3), Phase 1C dynamic-condition dataset (config fields
`lin_vel_range`/`ang_vel_range` exist), Stage F policy-driven aggregation, height/orientation-holdout *training* runs.

## 3. Frozen conventions (verified empirically; tests in `tests/test_chrono_conventions.py`)

* World Z up, ground plane z=0, normal +Z. Quaternion `(w,x,y,z)`, canonical `w>=0` (tie-break x>=0 when w≈0).
* Time alignment: `states[k]` pre-step; `contacts[k]` = collision pass at the start of `DoStepDynamics` k
  (from `states[k]`), read after the step via `ReportAllContacts`; `states[k+1]` post-step.
  Verified: reported `distance` == pre-step analytic gap. Record: `(s_k, c_k) -> s_{k+1}`.
* Canonical contact: A=box, B=ground, normal ground→box, `d>0` separated / `d<0` penetrating; box-local point clamped
  to half-extents (Bullet inflates box points laterally by the envelope); ground point = corrected box point
  projected onto the plane. Raw Chrono fields (`raw_*`: pA, pB, plane frame, distance, eff. radius, force/torque in
  contact frame, A/B identity, constraint offset) are always stored.
* Chrono integrator relations hold exactly in the data: `pos[k+1]=pos[k]+dt*v[k+1]`, `q[k+1]=exp(dt*w[k+1])⊗q[k]`
  → the transition model predicts only `(dv, dw)` (residual over the gravity step) and integrates pose exactly.
* Physics: NSC, Bullet box–box, APGD 100 it / 1e-6, Euler implicit linearized, dt=1 ms, envelope 1 mm, margin
  0.5 mm, μ=0.5, e=0.3, ρ=1000, g=9.81, max 2 s, pose-based settle detection (200-step window, 0.1 mm).

Important Chrono facts discovered (also in memory notes):
* `ChSystem` construction / `SetCollisionSystemType` resets default envelope/margin to 30 mm / 10 mm → the recorder
  sets defaults *after* creating the system and pins each model (`collision_envelope_actual` is stored in metadata).
* Bullet box–box (1 mm envelope) is the accurate pairing (d = analytic gap, exact +Z normals, ≤4 contacts).
  Mesh ground → bad edge normals; convex-hull box → bad points.
* Resting box on 4 NSC contacts has a deterministic period-4 rocking cycle (|v|≈3 mm/s, |w|≈0.08 rad/s, ±7 µm).
* Uniform SO(3) → 35 % corner / 55 % edge / 10 % face-dominant releases (15° rule).

## 4. Datasets (git-ignored, regenerable; all pass `scripts/validate_dataset.py`)

| Name | Config | Episodes | Frames | Notes |
| --- | --- | --- | --- | --- |
| `data/smoke` | `configs/smoke.yaml` | 50 | 59 k | fixed box 0.20×0.15×0.10, 45 settled |
| `data/fixed1a` | `configs/fixed1a.yaml` | 600 | 713 k | Phase 1A, splits 420/90/90 |
| `data/pilot1b` | `configs/pilot1b.yaml` | 2000 | 2.44 M | Phase 1B, 240 box instances (sides 0.05–0.30 m, aspect ≤4), splits train 1189 / val 255 / test 255 / test_geometry 301 (36 held-out groups), 710 MB, 37 s to generate |

Validation results (all three): schema pass; free flight matches ballistics (vel err 1e-13, pos offset = known
semi-implicit lag ≤2.7 mm); first contact within ≤1 step of the analytic proximity crossing (median 0.26);
replay bit-exact (0.0 diff); no split leakage. One pilot npz was corrupted on first write and was re-simulated.

## 5. Results so far (`runs/*/final_metrics.json`; `python scripts/summarize_runs.py runs`)

Analytic baseline vs Chrono (smoke, `runs/dbg_analytic`, deleted; re-run cheaply): slot recall 99.5 %, d MAE 0.16 mm,
point error 0, normal error 0°, first-impact timing median 0 / p99 ≤1 step. (The 0.5 % missed slots are Chrono
contacts with 2 mm < d < 3 mm, i.e. the persistent-manifold tail.)

Experiment A vs B, fixed box (`runs/expA_state_only`, `runs/expB_oracle_explicit`; 8000 steps, **without** gravity
prior/gate, block 32, 3 layers, 128-d):

| model | one-step dv err first impact | dw err first impact | rollout impact dv err (gt contacts) | pos err @100 steps |
| --- | --- | --- | --- | --- |
| state-only | 0.69 m/s | 4.7 rad/s | 3.7 m/s | 8.8 mm |
| oracle explicit contacts | 0.29 m/s | 2.7 rad/s | 1.4 m/s | 3.7 mm |

Re-runs with the gravity prior (30 k steps): A free-flight step error 0.0008 m/s, rollout pos@100 3.2 mm;
B rollout (analytic contacts) impact dv err 2.2 m/s, pos@100 1.8 mm. Long rollouts still diverge after the first
impact (rot err >100° at 2 s) — expected before gating and longer training. `runs/expA_pilot_state_only` finished
(variable geometry); `runs/expB_pilot_oracle_explicit` was interrupted.

Experiment C (interrupted at ~step 2000/8000): patch encoder val slot recall 93.6 %, frame recall 87.6 %,
d MAE 1.7 mm, matched point error median 0.03 % of min box side, before diverging (bug, fixed — §6).

## 6. Bugs found and fixed (code already updated, not yet re-run)

1. `SetDecoder` uncertainty head: log-variance was unbounded → NLL → −∞ → training blew up at step ~2000.
   Fixed: `log_var = 6*tanh(h/6)`.
2. Transition model: added `gravity_prior` (residual over `[0,0,-g dt,0,0,0]`) and `contact_gate` (residual ×
   any-contact-active; exact for Chrono since a step with no reported contact is pure free flight). Teacher-forced
   loss now goes through `predict_delta` so the gate is trained consistently. Flags: `--no-gravity-prior`,
   `--no-contact-gate`, `--loss huber`.
3. Machine hang: four concurrent trainers (4 loader workers each, default thread pools, CPU Hungarian) → load 285,
   GPU/driver wedged, unkillable. Run sequentially with thread limits. Do not `pkill -f <pattern>` with a pattern
   that appears in the calling shell's command line.
4. Minor: validator quaternion tie-break tolerance; ground-point projection; `SLOT_FEAT_DIM=19`.

## 7. Resume plan (run sequentially, in this order)

```bash
source /home/harry/anaconda3/etc/profile.d/conda.sh && conda activate recom
cd /home/harry/ReCoM && export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
python -m pytest tests -q                                   # sanity after reboot
ls data/pilot1b/index.json data/fixed1a/index.json          # regenerate with scripts/generate_dataset.py if missing

# Experiment C (contact encoders, geometry holdout) — ~25 min each on the 4090
python scripts/train_contact.py --data data/pilot1b --encoder patch    --out runs/expC_patch    --steps 8000 --eval-every 2000 --max-eval-episodes 120 --workers 2
python scripts/train_contact.py --data data/pilot1b --encoder point    --out runs/expC_point    --steps 8000 --eval-every 2000 --max-eval-episodes 120 --workers 2
python scripts/train_contact.py --data data/pilot1b --encoder analytic --out runs/expC_analytic --max-eval-episodes 120
#   gates (plan §17): near-contact recall ≥99.5 %, first-impact median ≤1 step / p99 ≤2, point ≤1 % min dim, normal ≤2°, no test_geometry collapse

# Experiments A/B with gravity prior + contact gate (fixed box, then pilot) — ~3 min each
python scripts/train_transition.py --data data/fixed1a --contact-mode none     --out runs/expA_state_only      --steps 30000 --eval-every 10000 --workers 2 --eval-contact-sources gt
python scripts/train_transition.py --data data/fixed1a --contact-mode explicit --out runs/expB_oracle_explicit --steps 30000 --eval-every 10000 --workers 2 --eval-contact-sources gt,analytic
python scripts/train_transition.py --data data/pilot1b --contact-mode none     --out runs/expA_pilot_state_only      --steps 30000 --eval-every 10000 --workers 2 --eval-contact-sources gt
python scripts/train_transition.py --data data/pilot1b --contact-mode explicit --out runs/expB_pilot_oracle_explicit --steps 30000 --eval-every 10000 --workers 2 --eval-contact-sources gt,analytic

# Experiment D: learned contacts + frozen NeDM (uses the B-pilot training recipe, evaluates with the learned encoder in the loop)
python scripts/train_transition.py --data data/pilot1b --contact-mode explicit --out runs/expD_learned_frozen --steps 30000 --eval-every 10000 --workers 2 \
    --encoder-ckpt runs/expC_patch/final.pt --eval-contact-sources gt,analytic,learned

# Experiment E: explicit+latent, unrolled loss, joint fine-tuning
python scripts/train_transition.py --data data/pilot1b --contact-mode explicit+latent --train-contact-source learned --encoder-ckpt runs/expC_patch/final.pt \
    --finetune-encoder --rollout-horizon 8 --out runs/expE_joint --steps 30000 --eval-every 10000 --workers 2 --eval-contact-sources learned,analytic

python scripts/summarize_runs.py runs > docs/RESULTS_$(date +%F).md
```

Optional follow-ups: failure-case figures (`recom.eval.visualize.save_worst_frames`), NeDM export
(`python -m recom.data.nedm_export data/pilot1b <out>`), orientation-stratified set (`orientation_strata` in a config),
neural-SDF baseline, Phase 1C.

## 7b. Cluster (Euler) workflow — added 2026-08-26 because the local box hung

Access: `ssh euler` (login `euler-login-1.engr.wisc.edu`, user `hzhang699`, home `/srv/home/hzhang699`, repo cloned at
`~/ReCoM`). Partitions available: `research` (RTX4000Ada ×8 nodes, RTXA4500 ×8, A100 ×4 on euler17/19, H100 ×4 on
euler29/30, 16-day limit) and `sbel` (euler16 2×2080Ti, euler19 4×A100). Conda on the cluster:

```bash
module load conda/miniforge && bootstrap-conda && conda activate recom     # cluster/env.sh does this
bash cluster/setup_env.sh                                                  # one-time env creation (done 2026-08-26)
```

Pipeline submission (from `~/ReCoM`, after `git pull`):

```bash
bash cluster/submit_all.sh            # gen_data -> C(patch,point,analytic) + A/B(fixed,pilot) -> D,E -> summary
SKIP_GEN=1 bash cluster/submit_all.sh # if data/ already exists on the cluster
squeue -u $USER; tail -f cluster/logs/<job>-<id>.out
```

`cluster/train.sbatch` is a generic single-GPU job (`sbatch --job-name=X cluster/train.sbatch <script> <args>`);
`cluster/gen_data.sbatch` regenerates and validates the datasets on a 32-core CPU allocation. Results land in
`runs/*/final_metrics.json` on the cluster and `docs/RESULTS_cluster.md` (summary job). Copy back with
`scp -r euler:ReCoM/runs/<name>/final_metrics.json ...` or commit `docs/RESULTS_cluster.md` from the cluster.

## 8. Open questions / decisions to revisit

* Contact-encoder threshold for "active" is 0.5 on the logit; the analytic baseline uses margin 2 mm while Chrono
  reports up to ~3 mm (envelope + 2×margin) — decide whether the encoder target should be Chrono's reported set
  (current) or the analytic proximity set.
* Gating with the learned encoder makes missed activations equal missed impulses (plan risk table): evaluate the
  soft-probability gate vs hard threshold in Experiment D.
* Long-horizon rollout divergence after impact (rot error) must be re-measured with the gate + 30 k steps before
  judging the "closed-loop degradation ≤10 %" gate.
