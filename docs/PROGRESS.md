# ReCoM Study 1 — Progress Log and Resume Guide

Last updated: 2026-08-28 (America/Chicago). Written so the next session can pick up without re-deriving anything.

## 0. TL;DR / where we stopped

* The Study-1 code base (box drop in Chrono → canonical contact schema → contact encoders → NeDM-compatible
  transition model → closed-loop rollouts) is implemented, tested (28/28 pytest), and three validated datasets exist.
* Experiments A/B (state-only vs oracle-contact transition, fixed box) ran to completion and show the H1 sign of life.
* The local box wedged on 2026-08-25 (load avg ~285, GPU/driver hang); all experiments were then run on the Euler
  cluster (§7b) and are **complete as of 2026-08-27** — results in §7c/§7d and `results/cluster_2026-08-26/`.
* Headline: oracle contacts make NeDM ~6× better at impacts (H1); a point-based contact encoder reproduces Chrono's
  contacts at 99 % recall / exact timing on unseen box sizes (H2, H4); feeding it to the *frozen* NeDM matches the
  oracle within noise (Stage D); joint fine-tuning helps only when the encoder is weak (H3 partially).
* **2026-08-28 — encoder–decoder milestone confirmed (§9.11, `docs/encoder-decoder/RESULTS_2026-08-28.md`):** with the
  impulse decoder (v6: exact Chrono gap + frictional single-contact solver prior + yaw augmentation) in the loop, JL-6-R
  beats the matched BASE-64 at 30 k steps × 3 seeds on all episodes: rot@0.5 s −33 % / −29 %, pos@0.5 s −16 % / −25 %
  (test / held-out geometry), impact Δω −41 % / −48 %, penetration −95 %, residual spin −59 %. Orientation at 2 s is
  chaotic-saturated for every model incl. the exact-contact oracle. **Next actions:** pass-2 items (JL-6-C momentum loss,
  POOL-DEC, ORACLE-JL, decoder fine-tuning), NeDM-repo integration of the wrench interface, broad-phase study.
* Older next actions: end-of-episode orientation drift analysis (per-regime), near-contact recall 99.2 → 99.5 %
  (threshold/calibration or more data), neural-SDF baseline, Phase 1C, NeDM-repo integration via `nedm_export`.

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

## 7c. Cluster round 1 results (2026-08-26; full tables in `results/cluster_2026-08-26/RESULTS.md`)

All 9 training jobs completed (H100/A100; patch encoder 10 min, point 23 min, transition runs 6–14 min, joint E 73 min).
Datasets regenerated on the cluster are statistically identical to the local ones (deterministic recipe).

**Experiment C — contact encoders (8000 steps, K=4 set loss, pilot1b, held-out geometry `test_geometry`)**

| encoder | frame recall (test / test_geo) | frame precision | d MAE | matched point err (median, % min side) | normal err (median) | first-impact timing median / p99 [steps] |
| --- | --- | --- | --- | --- | --- | --- |
| analytic | 0.990 / 0.990 | 1.00 | 0.15 mm | 0 | 0° | 0 / 1 |
| patch (selected) | 0.81 / 0.88 | 0.96 | 1.0 mm | 0.006 % | 0.39° | 3 / 417 |
| point | 0.78 / 0.83 | 0.96 | 0.8 mm | 0.005 % | 0.23° | 102 / 595 |

Per category (patch, test): far_free FPR 0.000, resting recall 1.00, contact recall 0.73, first_impact recall 0.48
(FPR 0.26), near_contact FPR 0.10. → Point/normal gates pass; the near-contact-recall and timing-p99 gates fail
because the ~3 mm Chrono reporting boundary is only partially resolved (val recall was still rising at 8 k steps:
0.77 → 0.84 → 0.88 → 0.86). No collapse on held-out geometry (H4 sign of life). Patch > point on recall/timing.

**Experiments A/B — H1 (30 k steps, gravity prior; B with contact gate)**

| | one-step dv err first impact | dw err first impact | free-flight step err | rollout impact dv err (median) | pos err @500 steps |
| --- | --- | --- | --- | --- | --- |
| A state-only (fixed / pilot) | 0.67 / 0.68 m/s | 4.8 / 7.0 rad/s | 7e-4 / 2.6e-3 m/s | 3.4 / 3.3 m/s | 0.10 / 0.11 m |
| B oracle explicit (fixed / pilot), analytic contacts in loop | 0.24 / 0.29 m/s | 2.3 / 4.0 rad/s | 4e-8 (exact) | 0.62 / 0.45 m/s | 0.055 / 0.046 m |

→ H1 confirmed: oracle contacts cut impact-step errors 2.5× (one-step) and 5–7× (closed loop) and make free flight
exact. Rollouts with *recorded* contacts replayed open-loop penetrate (0.3 m) because contacts do not follow the
predicted pose; recomputing contacts from the predicted pose (analytic) keeps penetration ≤ 1 cm. Orientation
error at 2 s is still ~90° for every model (long resting/rocking phase; to be examined per regime).

**Experiment D — learned patch contacts + frozen NeDM (trained with gt contacts)**: impact dv err 3.5–3.7 m/s,
pos err @100 = 11–14 mm → as bad as state-only. Cause: near-boundary false positives (10 %) open the contact gate in
free flight and first-impact misses (52 %) delay impulses. The gate design makes activation accuracy critical.

**Experiment E — explicit+latent, trained on learned contacts, 8-step unrolled loss**: learned-contact rollouts reach
the oracle level on all splits: impact dv err 0.45–0.59 m/s (B: 0.45–0.63), pos err @500 = 0.044–0.056 m (B analytic:
0.041–0.055), max penetration 2.5 mm (B: 10 mm), also on `test_geometry` (0.45 m/s, 0.056 m). → the closed-loop
≤10 % degradation gate is met on these metrics for held-out geometry. Caveat found afterwards: in round 1 the encoder
received **no gradients** (`batch_contacts` ran under `no_grad`) and no contact loss was retained, so round-1 E is
"transition trained on frozen learned contacts + unrolled loss", not joint fine-tuning. Fixed in code (true joint
fine-tuning + `--contact-loss-weight`, fine-tuned encoder saved as `encoder_finetuned.pt`); round 2 re-runs it.

Round 2 (`cluster/submit_round2.sh`): C_patch/C_point at 30 k steps → D_r2, E_r2 (joint + contact loss) → summary.

## 7d. Cluster round 2 results (2026-08-26, partial)

**Experiment C at 30 k steps** (same architecture; only longer cosine schedule):

| encoder | frame recall test / test_geo | precision | d MAE | matched point err | first-impact timing median / p99 | near-contact FPR | first-impact recall |
| --- | --- | --- | --- | --- | --- | --- | --- |
| patch 30k | 0.958 / 0.971 | 0.98 | 0.5 mm | 0.0002 % | 0 / 29–58 steps | 4–5 % | 0.87–0.90 |
| **point 30k** | **0.992 / 0.990** | **0.996** | 0.48 mm | 0.0001 % | **0 / 1 step** | 1 % | 0.91–0.95 |

→ Longer training resolves the reporting boundary. The point encoder now passes the timing gate (median 0, p99 1)
and is at the near-contact recall gate (99.2 % vs 99.5 % target) with no degradation on held-out geometry; the patch
encoder lags on the activation boundary (its 12 face tokens give the decoder less spatial resolution near corners
than the 152 surface points). **Bake-off selection for Study 1: point encoder** (patch kept as the mesh-general path).

**Experiment D_r2 (frozen NeDM trained with gt contacts, evaluated with the 30k patch encoder in the loop)**:
free flight exact (pos err @100 = 0), impact dv err 0.58–0.69 m/s vs 0.44–0.54 with analytic contacts, pos err
@500 = 0.047–0.050 m vs 0.041–0.049, penetration 2 cm vs 1 cm, same on held-out geometry → within ~15–30 % of the
oracle without any adaptation of the transition model (round-1 D had failed completely).

**Experiment E_r2 (patch 30k encoder, true joint fine-tuning, retained contact loss, 8-step unrolled loss, 8 k steps,
73 min on RTX4000Ada)** — learned-contact closed loop, median over episodes:

| split | impact dv err | pos err @500 | pos err final (2 s) | max penetration | rot err final |
| --- | --- | --- | --- | --- | --- |
| val | 0.42 m/s | 0.046 m | 0.17 m | 4.7 mm | 93° |
| test | 0.48 m/s | 0.051 m | 0.16 m | 7.8 mm | 92° |
| test_geometry | 0.43 m/s | 0.047 m | 0.15 m | 3.6 mm | 98° |
| oracle B (analytic contacts) for reference | 0.45–0.54 m/s | 0.041–0.049 m | 0.12–0.18 m | ~10 mm | 87–94° |

→ Joint fine-tuning with the explicit contact loss retained reaches the oracle-contact upper bound on impact and
position metrics, on held-out geometry as well (closed-loop degradation gate met). Evaluating this model with
analytic contacts is worse (impact 1.7–1.9 m/s) because the transition now relies on the fine-tuned encoder's latent.

**Gate bug found via D_point**: D with the (better) point encoder gave impact dv err 2.2–2.9 m/s. Cause: the frozen
NeDM was trained with a hard 0/1 gate but the rollout used the encoder's *soft* probability as the gate, scaling
impulses down at uncertain first-impact frames. Fixed (`soft_gate` only for jointly trained models) and both D
variants are being re-evaluated with `--eval-only` (jobs 16041/16042). `expE_r2_point` OOMed on a 40 GB A100
(152 tokens × 10 k frames with autograd) → resubmitted with batch 64 (job 16043); summary job 16044.

**Experiment D with the gate fixed (`--eval-only` re-evaluation of the frozen, gt-trained NeDM; jobs 16041/16042)** —
median closed-loop metrics, learned contacts vs analytic contacts recomputed from the predicted pose:

| encoder in the loop | split | impact dv err learned / analytic | pos err @500 learned / analytic | pos err final learned / analytic |
| --- | --- | --- | --- | --- |
| patch 30k | test | 0.60 / 0.55 m/s | 0.045 / 0.044 m | 0.16 / 0.13 m |
| patch 30k | test_geometry | 0.46 / 0.53 m/s | 0.039 / 0.047 m | 0.13 / 0.11 m |
| point 30k | test | 0.43 / 0.45 m/s | 0.050 / 0.048 m | 0.17 / 0.18 m |
| point 30k | test_geometry | 0.44 / 0.46 m/s | 0.045 / 0.047 m | 0.11 / 0.11 m |

→ With a hard activation gate, **learned contacts feeding a frozen NeDM match the exact-contact oracle within noise
(≤10 % on every metric, held-out geometry included)** — Stage D of the plan passes without joint fine-tuning. Free
flight stays exact (pos err @100 = 1e-7) because both encoders have ~0 false positives away from the ground.
The soft-gate variant (round 2b) is what had inflated D_point to 2–3 m/s.

**Experiment E_r2 with the point encoder (job 16357, euler19 A100, batch 32, 8 k steps, 66 min)** — the point
encoder's kNN block needs ~2.5 MB/frame, so joint training needs batch ≤ 32 on 40 GB and no-grad evaluation is
chunked (fix in `batch_contacts`). Learned-contact closed loop: impact dv err 0.64 / 0.71 m/s (test /
test_geometry), pos err @500 = 0.062 / 0.054 m, final pos err 0.13 / 0.16 m, final rot err 78° / 90°.
→ Slightly *worse* than the frozen-NeDM D_point (0.43 / 0.44 m/s) and than E_patch (0.48 / 0.43 m/s): with the
point encoder already at Chrono-level accuracy, joint fine-tuning at this budget adds nothing and the smaller batch
hurts. Joint fine-tuning helped only when the encoder was the bottleneck (round-1 8 k-step patch encoder).

### Study-1 closed-loop summary (median over test episodes; impact dv err / pos err @500 / max penetration)

| configuration | test | test_geometry (held-out box sizes) |
| --- | --- | --- |
| A  state-only NeDM | 3.4 m/s / 0.11 m / 7 cm | 3.3 m/s / 0.12 m / 5.5 cm |
| B  oracle contacts (analytic, recomputed from predicted pose) | 0.45 m/s / 0.046 m / 1.0 cm | 0.45 m/s / 0.052 m / 0.8 cm |
| D  learned **point** contacts + frozen NeDM (hard gate) | 0.43 m/s / 0.050 m / 1.4 cm | 0.44 m/s / 0.045 m / 1.7 cm |
| D  learned **patch** contacts + frozen NeDM (hard gate) | 0.60 m/s / 0.045 m / 2.0 cm | 0.46 m/s / 0.039 m / 2.0 cm |
| E  joint fine-tuning, patch | 0.48 m/s / 0.051 m / 0.8 cm | 0.43 m/s / 0.047 m / 0.4 cm |
| E  joint fine-tuning, point | 0.64 m/s / 0.062 m / — | 0.71 m/s / 0.054 m / — |

Pilot gates (plan §17): data replay ✓, sign conventions ✓, oracle benefit ✓ (H1), contact timing ✓ (point: median 0,
p99 1 step), near-contact recall ✗ marginal (99.2 % vs 99.5 %), contact point ✓ (0.0001 % of min side), normal ✓ (0°),
closed-loop degradation ✓ (D_point within noise of B), geometry holdout ✓ (no degradation on held-out sizes).
Open weakness: end-of-episode orientation error ~80–95° for every model including the oracle (settling/rocking phase).

## 8. Open questions / decisions to revisit

* Contact-encoder threshold for "active" is 0.5 on the logit; the analytic baseline uses margin 2 mm while Chrono
  reports up to ~3 mm (envelope + 2×margin) — decide whether the encoder target should be Chrono's reported set
  (current) or the analytic proximity set.
* Gating with the learned encoder makes missed activations equal missed impulses (plan risk table): evaluate the
  soft-probability gate vs hard threshold in Experiment D.
* Long-horizon rollout divergence after impact (rot error) must be re-measured with the gate + 30 k steps before
  judging the "closed-loop degradation ≤10 %" gate.

## 9. Encoder–decoder study (contact impulse decoder + wrench-conditioned NeDM), implemented 2026-08-27

Plan: `docs/encoder-decoder/ContactROM_contact_encoder_decoder_plan.md`; review with the corrections applied:
`docs/encoder-decoder/plan_review_2026-08-27.md`. Pipeline: geometry encoder (frozen point 30k) → K=4 slots incl. the new
post-attention `slot_embedding` → `ContactImpulseDecoder` (set attention over slots, friction-cone head) → per-slot
mass-normalized impulses `j_i/m` → deterministic wrench `(Σ j_i/m, Σ r_i × j_i/m)` → transition model conditioned on the
6-D contact-induced `(Δv, Δω)` (`--contact-mode wrench`, JL-6), its linear part (`wrench_lin`, J-3), or applied as a
Newton–Euler step + residual (`--physics-residual`, JL-6-R). Everything is recomputed from the predicted state at every
rollout step (`ContactSource(decoder=...)`).

### 9.1 Frozen label formulas (`recom/data/impulse_targets.py`, `impulse_labels.v1`) and ED0 audit

Verified on pilot1b (60 episodes) and smoke (`results/audit_impulse_labels/smoke.md`; `scripts/audit_impulse_labels.py`):

* `c_force_world` is a true force (N) on the box and the recorded contact set is complete:
  `m(v[k+1]−v[k]) = (ΣF + m g) dt` to 3.6e‑15 → `J = dt ΣF`.
* Chrono's discrete angular update **includes the gyroscopic torque**: `L_b = I_b(ω_b[k+1]−ω_b[k]) + dt (ω_b × I_b ω_b)`
  is zero to 7e‑13 on 7,920 tumbling free-flight frames; the plan's `H[k+1]−H[k]` is off by 0.5 % median / 4.3 % max.
* Force-derived torque `Σ r_i × F_i dt` equals `R L_b` to 3e‑14 with Chrono's **raw** contact points; with the canonical
  clamped points (what the encoder predicts) the median error is 1.4–2 % (p90 7–8 %) → small inherent floor for the
  decoder's `L`.
* 25 % of active slots carry exactly zero force (envelope contacts; 48–67 % at first impact) → activation ≠ impulse.
* Mass spans 0.125–27 kg (200×) → all labels/inputs are mass-normalized: `dv_c = J/m`, `dL = R L_b/m`, `dw_c = R I_b⁻¹ L_b`.

**Gyroscopic prior (new, affects every existing model):** the contact gate forces `Δω = 0` when no slot is active, but a
tumbling box changes `ω_world` in free flight: median 0.57 rad/s (p90 5 rad/s) net per post-impact free-flight run in
pilot1b. `Δω_w = −dt R I_b⁻¹(ω_b × I_b ω_b)` (only inertia ratios → from half-extents) reproduces Chrono to 1e‑12 and is now
a state-dependent prior (`BoxTransitionModel(gyro_prior=True)`, default in `train_transition.py`, `--no-gyro-prior` to
disable; old checkpoints rebuild with `gyro_prior=False` under `--eval-only`).

### 9.2 Code added / changed

| Area | Files |
| --- | --- |
| Labels, data | `recom/data/impulse_targets.py` (new), `recom/data/dataset.py` (`EpisodeArrays.impulse/phys`, `target_*` + `PHYS_KEYS` per window item, `ImpulseFrameDataset`, `compute_wrench_normalization`, `compute_state_normalization(..., gyro)`) |
| Encoder | `SetDecoder` returns `slot_embedding (B,K,d_model)`; excluded from the set loss / metrics / adapter (no behaviour change: old checkpoint evaluates bit-identically) |
| Decoder | `recom/models/impulse_decoder.py` (new): `tangent_basis`, `aggregate_wrench`, `wrench_to_delta`, `body_angular_impulse_torch`, `gyro_delta_omega_world`, `cone_violation`, `ContactImpulseDecoder` (cone/free head, `pooled_only`, `use_slot_embedding`, null embedding for gt/analytic geometry) |
| Transition | `recom/models/transition.py`: `gyro_prior`, `prior_delta(states, he)`, contact modes `wrench`/`wrench_lin`, `physics_residual`; explicit/latent paths untouched |
| Training | `recom/train/train_impulse.py` + `scripts/train_impulse.py` (ED2 decoder pretraining, frozen encoder in the loop, Huber on the normalized net wrench over encoder-active frames, `missed_impulse_rate`, zero-impulse baseline, gt-geometry ablation); `recom/train/train_transition.py`: `--decoder-ckpt`, `--physics-residual`, `--no-gyro-prior`, `--wrench-ablation zero|shuffle`, impulse-binned one-step errors, `final["impulse"]` |
| Rollout / metrics | `recom/train/rollout.py`: `ContactSource(decoder=, wrench_ablation=)`, `attach_wrench`, per-episode `phys`; `recom/eval/metrics.py`: `impulse_frame_metrics`/`aggregate_impulse_metrics`, `settled_face_np`, `symmetry_rot_err_deg_np` (D2 primary, octahedral bound) → new rollout keys `sym_rot_err_deg_final`, `settled_face_match`, `final_speed_err`, ... |
| Scripts, cluster, tests | `scripts/audit_impulse_labels.py`, `scripts/summarize_runs.py` (decoder table, seed groups mean ± std), `cluster/submit_ed.sh`; tests `test_impulse_labels.py`, `test_impulse_decoder.py`, `test_transition_wrench.py`, + extensions (42 passed) |

### 9.3 Local verification (2026-08-27, data/smoke, 4090)

* `python -m pytest tests -q` → 42 passed.
* Guard: `--eval-only runs/expB_pilot_oracle_explicit/final.pt` on smoke gives bit-identical one-step and rollout medians
  with the pre-change code (git worktree at `b7e0526`) and the new code.
* `scripts/train_impulse.py --data data/smoke --encoder-ckpt runs/expC_point/final.pt --steps 300` (`runs/smoke_ed2`):
  decoder already below the zero-impulse baseline on first impact (dv MAE 1.78 vs 2.06 m/s), cone violations 0.
* `scripts/train_transition.py` 200-step smoke runs for `explicit` (BASE-64 with gyro prior), `wrench`, `wrench_lin`,
  `wrench --physics-residual`, `--rollout-horizon 4`, `--wrench-ablation zero` and `--eval-only`: all run, free flight
  exact (pos err @100 = 1e‑7), eval-only reproduces the run's numbers exactly, zeroing the wrench changes the rollouts.

### 9.4 Cluster recipe (`cluster/submit_ed.sh`)

`ed0_audit` → `ed2_dec_cone` (20 k steps, batch 512) → for seeds 0–2: `ed3_base64_s*` (explicit, learned contacts, gyro
prior), `ed3_j3_s*` (`wrench_lin`), `ed3_jl6_s*` (`wrench`), `ed3_jl6r_s*` (`wrench --physics-residual`); reference rows
`ed3_base64_gt_s0` (explicit, gt-trained, gyro prior: isolates the prior) and `ed3_ref_expD_point` (old frozen NeDM
re-evaluated with the new metrics); `ed3_jl6_s0_wrench_zero` (wrench-usage gate #7); summary. All ED3 runs use the same
30 k-step recipe as `expD_r2_learned_frozen_point` but train on the *learned* contacts (chunked no-grad point-encoder
forward per step, est. +2 h per run; add an encoder-output cache if the 6 h limit bites). Gate ED2 (decoder ≫ zero
baseline on first_impact / rebound, no test_geometry collapse) before reading ED3.

Pass 2 (not yet implemented): JL-6-C momentum-consistency loss, POOL-DEC (`--pooled-only` decoder exists, transition
recipe not queued), ORACLE-JL, decoder fine-tuning + per-slot Hungarian-matched loss, free vs cone head.

### 9.5 Decoder diagnosis and fixes (2026-08-27/28, local 4090, `runs/local_*`)

The first decoder recipe (v1: 128-D slot embedding + 19-D features, fixed output scale, Huber δ=1, balanced sampling)
produced JL-6 / JL-6-R rollouts *worse* than BASE-64 at 8 k steps.  Root causes found, in order:

1. **Memorization** — train first-impact dv MAE 0.09 m/s vs 0.73–0.88 on val/test.  pilot1b has only ~1,200 distinct
   first impacts; the balanced sampler repeats each ~300×.  Dropping the slot embedding alone did not help.
2. **Yaw augmentation** (`--yaw-aug`: rotate quat/velocities/labels about z; exact symmetry of the box–plane problem,
   verified to 1e‑6) removed the memorization and fixed the angular channel (first-impact dL MAE 0.027 vs 0.060 zero
   baseline; was worse than zero before), but the model then could not fit even the training set (~50 % of the impulse).
3. **The remaining ambiguity is Chrono's NSC gap stabilization**: at the first reported contact the impulse only removes
   the excess approach `|v_n| − d/dt` (traces: `v_n → −d/dt`, restitution the next step), so it depends on the reported
   distance at sub-mm precision — the regressed `d` (≈0.5 mm error) is useless for it.  The predicted contact point is
   exact (0.000 mm), but Chrono's `d` is *not* the point height: `d = h − (Σ_a|R[2,a]| − 1)·envelope` (Bullet envelope
   inflation; verified: median residual 0.0000 mm, p99 0.02 mm over 149 k active slots, `test_chrono_gap_formula...`).
   `impulse_decoder.chrono_gap` gives the decoder Chrono's exact gap; timing features are `d/dt`, `u = −v_n − d/dt`, `relu(−v_n)`.
4. **Dynamic range**: the head outputs a multiplier of a per-slot physical scale (`--head-scale delassus`: the exact
   single-contact normal impulse `−(1+e)v_n / (1 + (r×n)ᵀ(I_b/m)⁻¹(r×n))` + g dt, restitution only above
   `min_bounce_speed`); Huber δ=0.1 (near-L1) so resting frames (0.2 σ) and impacts (50 σ) get equal weight.

Decoder v5 (= `--no-slot-embedding --yaw-aug --head-scale delassus`, 20 k steps, 11 min on the 4090): no train/test gap
(1.07 / 1.05 first-impact dv MAE vs 2.1 zero baseline — the residual per-step ambiguity is left to NeDM), first-impact
dL 0.027 vs 0.060, rebound dv 0.21 vs 0.42, resting dv 0.0024 vs 0.0102 (zero baselines), 0 cone violations,
encoder-miss rate 0.2 %.

**Local 8 k-step comparison (same recipe as the cluster runs but 8 k steps, 120 test episodes, learned point contacts,
paired per-episode medians; `scripts/compare_rollouts.py`)** — JL-6-R (physics residual) vs BASE-64:

| metric (test) | BASE-64 | JL-6-R | change | 95 % CI of paired median diff |
| --- | ---: | ---: | ---: | --- |
| one-step first-impact dv / dw | 0.272 m/s / 3.46 rad/s | 0.163 / 2.06 | −40 % / −40 % | (teacher-forced means) |
| one-step rebound / contact / resting dv | 0.055 / 0.0100 / 0.0037 | 0.031 / 0.0057 / 0.0026 | −44 / −43 / −30 % | |
| rollout impact dv err | 0.605 m/s | 0.453 | −25 % | [−0.149, −0.009] * |
| rollout impact dw err | 6.11 rad/s | 3.95 | −35 % | [−2.18, −0.23] * |
| rot err @500 steps | 27.4° | 21.9° | −20 % | [−4.9, 0] |
| pos err @500 / @1000 | 0.055 / 0.211 m | 0.039 / 0.159 | −28 % / −25 % | [−0.011, 0] / [−0.062, +0.008] |
| max penetration | 7.0 mm | 0.5 mm | −93 % | [−7.9, −4.8] mm * |
| final angular speed err | 0.79 rad/s | 0.26 | −67 % | [−0.54, −0.26] * |
| rot err final (2 s) | 91° | 91° | 0 % | saturated for every model incl. the oracle (chaotic settling face) |

Held-out geometry (`test_geometry`): impact dw −19 % *, penetration −89 % *, final angular speed −70 % *, rot@500 −21 %,
pos@500 −13 %.  JL-6 (NRD full delta) shows the same one-step gains but smaller closed-loop gains than JL-6-R.
Cluster wrench arm (30 k steps × 3 seeds, `cluster/submit_ed_wrench.sh`, jobs 17564–17575) submitted 2026-08-28.

**Wrench-usage gate (#7, `--wrench-ablation zero|shuffle` on the 8 k JL-6-R, test, 120 episodes):** zeroing the decoder
wrench at evaluation raises impact dv err 0.45 → 4.07 m/s, rot err @500 22° → 60°, max penetration 0.5 mm → 5.6 m (the
box falls through the ground); shuffling the wrench across episodes: 0.45 → 3.82 m/s, penetration 1.6 m.  All paired
differences significant (`runs/local_ed3_jl6r_8k_v5_wrench_{zero,shuffle}`).  The transition model relies on the decoder.

### 9.6 Closed-loop training (plan stage ED4) — interim, 2026-08-28

*Why:* at 30 k steps the one-step advantage of JL-6-R (−40–50 % in every regime vs all three BASE-64 seeds) did not
translate into long-horizon pose gains: every model — including the exact-contact oracle (gt-trained NeDM + analytic
contacts) — has the same saturated error-vs-horizon curve (pos ≈ 0.03–0.05 m @500, ≈ 0.2 m @1000, rot ≈ 95–120° @1000);
the 2 s pose of a bouncing box is chaotic.  Only per-step accuracy in the closed loop can move the pre-saturation part.

*Result (local, 8 k steps, 120 test episodes, paired medians; `runs/local_ed3_jl6r_8k_v5_h8` = JL-6-R + `--rollout-horizon 8`
vs `runs/local_ed3_base64_8k` without unrolled loss — the matched BASE-64+H8 run is queued):*

| metric | test | held-out geometry |
| --- | --- | --- |
| pos err @500 / @1000 / final | −33 % / −25 % / −27 % (all *) | −17 % / −8 % / −15 % (all *) |
| rot err @500 | −43 % * | −30 % |
| rot err final (2 s) / symmetry-aware | −1 % / −38 % | −3 % / +12 % (saturated) |
| impact dv / dw err | −30 % / −31 % (*) | −28 % / −33 % (*) |
| max penetration | −96 % * | −92 % * |
| final angular-speed err | −82 % * | −77 % * |

Cluster: 30 k × 3 seeds of BASE-64+H8 and JL-6-R+H8 submitted (`cluster/submit_ed_h8.sh`, jobs 17702–17708).
Decoder v6 (`--head-scale solver`: closed-form frictional single-contact impulse with Chrono's gap-stabilization /
restitution regimes as prior features and normal scale; `single_contact_solver`) is being evaluated locally.

### 9.7 Decoder v6: analytic frictional single-contact prior (2026-08-28)

The remaining first-impact error of v5 had a clear structure: on single-contact frames the true normal impulse was
0.66–1.9× the *frictionless* single-contact value (median 1.34; friction–normal coupling through the full contact
Delassus matrix `G = I − [r]ₓ(I_b/m)_w⁻¹[r]ₓ`) and the decoder hedged at 0.73×.  `single_contact_solver` computes the
closed-form impulse of one frictional contact acting alone: normal target `v'_n = −d/dt` for a positive Chrono gap
(stabilization), `e·(−v_n)` when penetrating above `min_bounce_speed`, penetration recovery otherwise (all three verified
in recorded traces); stick if the cone allows, else Coulomb slip against the pre-impact sliding direction.  Its impulse
(in the (n,t1,t2) basis), stick flag and normal target are decoder features and its normal impulse is the head scale
(`--head-scale solver`).  Decoder v6 = v5 + solver prior, 20 k steps.

| decoder (test split) | first-impact dv MAE (zero 2.43) | first-impact dL MAE (zero 0.066) | rebound dv (zero 0.42) |
| --- | ---: | ---: | ---: |
| v1 (slot embedding, fixed scale) | 0.88 | 0.073 (worse than zero) | 0.27 |
| v5 (yaw-aug, exact Chrono gap, Delassus scale) | 1.28 | 0.026 | 0.21 |
| **v6 (+ frictional solver prior)** | **0.60** | **0.014** | — |

JL-6-R with decoder v6, 8 k steps, no unrolled loss, vs BASE-64 8 k (120 episodes, paired medians, * = 95 % CI excludes 0):

| metric | test | held-out geometry |
| --- | --- | --- |
| pos err @500 / final | −47 % * / −27 % * | −33 % * / −16 % |
| rot err @500 | −40 % * | −39 % |
| impact dv / dw err | −34 % * / −52 % * | −36 % * / −47 % * |
| max penetration / final angular-speed err | −93 % * / −65 % * | −88 % * / −60 % * |

Cluster: decoder v6 + JL-6-R / JL-6 / JL-6-R+H8 × 3 seeds submitted (`cluster/submit_ed_v6.sh`).

### 9.8 Matched closed-loop comparison — MILESTONE (local, 8 k steps, 2026-08-28)

Both models trained with the 8-step unrolled loss (`--rollout-horizon 8`), same encoder, data, steps, batch, gate and
priors; 120 test / 120 held-out-geometry episodes, learned point contacts recomputed from the predicted state, paired
per-episode medians (`scripts/compare_rollouts.py --base runs/local_ed3_base64_8k_h8`):

| metric | BASE-64+H8 | **JL-6-R v6+H8** | test | held-out geometry |
| --- | ---: | ---: | ---: | ---: |
| pos err @500 steps | 0.042 m | 0.026 m | **−39 %** * | **−28 %** * |
| pos err final (2 s) | 0.132 m | 0.102 m | **−23 %** * | **−34 %** * |
| rot err @500 steps | 23.6° | 12.7° | **−46 %** * | **−36 %** * |
| rot err final (2 s) | 87.7° | 81.8° | −7 % | −5 % (paired −21° *) |
| impact dv / dw err | 0.43 m/s / 6.2 rad/s | 0.31 / 2.7 | −28 % * / −57 % * | −28 % * / −46 % * |
| max penetration | 29 mm | 0.3 mm | −99 % * | −98 % * |
| final angular-speed err | 1.54 rad/s | 0.15 rad/s | −90 % * | −90 % * |

(* = 95 % bootstrap CI of the paired median difference excludes 0.)  The unrolled loss alone improves BASE-64's pose
(pos final 0.184 → 0.132 m) but degrades its physics (penetration 7 → 29 mm, residual spin 0.79 → 1.54 rad/s); the
wrench bottleneck improves both.  JL-6-R v5+H8 (previous decoder) gave −13 % / −33 % on pos@500 / rot@500 — the
solver prior in v6 is what makes the pose gains large.  Runs: `runs/local_ed3_{base64_8k_h8,jl6r_8k_v6_h8,jl6r_8k_v5_h8}`.
Pending: cluster confirmation at 30 k steps × 3 seeds (`ed4_base64_h8_s*` vs `ed5_jl6r_v6_h8_s*`).

### 9.9 Cluster confirmation, v5 decoder arm (30 k steps × 3 seeds, no unrolled loss; `results/cluster_ed/ed3_*`)

Per-seed medians averaged over seeds (64 episodes/split) and pooled paired difference vs the same-seed BASE-64
(189 pairs; * = bootstrap 95 % CI excludes 0):

| test | pos@500 | pos final | rot@500 | rot final | impact dv | impact dw | penetration | final spin |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BASE-64 | 0.047 m | 0.166 m | 25.1° | 95.7° | 0.49 m/s | 6.9 rad/s | 16.6 mm | 0.45 rad/s |
| J-3 (linear only) | +64 % * | +98 % * | +42 % * | +17 % * | +105 % * | +62 % * | −49 % * | +501 % * |
| JL-6 (2 seeds) | −7 % | −7 % | −3 % | +1 % | −5 % | −23 % * | −84 % * | −27 % * |
| JL-6-R | −2 % | +3 % | −3 % | −2 % | −2 % | −17 % * | −82 % * | −37 % * |

Held-out geometry: the same pattern (J-3 +73…+114 % *, JL-6/JL-6-R pose −2…+19 % n.s., impact dw −22/−24 % *,
penetration −88/−90 % *, spin −35/−31 % *).  Conclusions: (i) gate #6 — the 3-D linear impulse alone is far worse than
the 64-D context; the angular impulse channel is essential; (ii) with decoder v5 and one-step training the wrench
bottleneck reaches pose parity with BASE-64 while fixing the physics (penetration, residual spin, impact rotation);
(iii) the pose gains (§9.8) need the v6 solver prior + unrolled training — cluster confirmation `ed5_*` pending.

**Ablations at the milestone setting (decoder v6, 8 k steps, both with H8, vs BASE-64+H8; `runs/local_ed3_{jl6,j3}_8k_v6_h8`):**

| model | pos@500 | pos final | rot@500 | impact dv / dw | penetration | final spin |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| JL-6-R (physics residual) | −39 % * | −23 % * | −46 % * | −28 % * / −57 % * | −99 % * | −90 % * |
| JL-6 (NRD full delta, NeDM-transferable) | −40 % * | −21 % * | −43 % * | −32 % * / −47 % * | −99 % * | −90 % * |
| J-3 (linear impulse only) | 0 % | +110 % * | +29 % | +5 % / +24 % | −6 % | +146 % * |

Held-out geometry: JL-6-R −28 % * / −34 % * / −36 % *; JL-6 −24 % * / −28 % / −19 % *; J-3 +14 % / +124 % * / +17 %.
→ The 6-D wrench works equally well when NRD predicts the full transition (JL-6) — the form that carries over to reduced
NeDM states — and the angular-impulse channel is indispensable (J-3).

**Ablations at the matched setting (decoder v6 + H8, 8 k steps, vs BASE-64+H8; test / held-out geometry):**
JL-6 (NRD predicts the full delta from the 6-D wrench — the NeDM-transferable form): pos@500 −40 % * / −24 % *,
rot@500 −43 % * / −19 % *, impact dv −32 % * / −29 % *, dw −47 % * / −47 % *, penetration −99 % * / −98 % *,
final spin −90 % * / −90 % * — essentially as good as JL-6-R.  J-3 (linear impulse only, gate #6): pos final +110 % * /
+124 % *, rot final +23 % * / +25 %, final spin +146 % * / +327 % *, penetration +529 % * on held-out geometry — the
angular-impulse channel is essential.  Runs: `runs/local_ed3_{jl6,j3}_8k_v6_h8`.

### 9.9 Cluster, 30 k steps, no unrolled loss (64 test episodes; relative change of paired medians vs the 3 BASE-64 seeds, mean over pairings)

| variant | pos@500 | pos final | rot@500 | impact dv | impact dw | penetration | final spin |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| JL-6-R, decoder v5 (3 seeds), test | −13 % | 0 % | −13 % | +5 % | −28 % | −95 % | −60 % |
| JL-6-R, decoder v5, held-out geometry | +13 % | +16 % | −1 % | +3 % | −30 % | −96 % | −49 % |
| JL-6-R, decoder v6 (seed 0), test | −30 % | −14 % | −27 % | −17 % | −37 % | −95 % | −45 % |
| JL-6-R, decoder v6 (seed 0), held-out geometry | −14 % | −32 % | −26 % | −30 % | −53 % | −96 % | −29 % |
| JL-6, decoder v6 (seed 0), test / held-out | −26 % / +8 % | −22 % / +2 % | −8 % / −8 % | −15 % / −14 % | −37 % / −52 % | −96 % | −43 % / −30 % |
| J-3, decoder v5 (3 seeds), test | +115 % | +97 % | +98 % | +112 % | +105 % | −86 % | +532 % |

Absolute medians (test, mean over seeds): BASE-64 pos@500 0.047 m, rot@500 25.1°, impact dw 6.9 rad/s, penetration 16.6 mm;
JL-6-R v6: 0.033 m, 18.2°, 4.3 rad/s, 0.7 mm.  Runs in `results/cluster_ed/`.

### 9.10 Cluster, 30 k steps, no unrolled loss, ALL episodes (255 test / 301 held-out geometry), decoder v6 seed 0 vs each BASE-64 seed

Paired medians (`*_ev255` re-evaluations, `cluster/submit_ed_eval255.sh`); range over the three BASE-64 seeds, * = every pairing significant:

| metric | JL-6-R v6, test | JL-6-R v6, held-out geometry | JL-6 v6, test | JL-6 v6, held-out |
| --- | --- | --- | --- | --- |
| pos err @500 | −19 … −34 % * | −22 … −29 % * | −5 … −22 % | −16 … −23 % |
| pos err final | −15 … −22 % (2/3 *) | −14 … −22 % (1/3 *) | −11 … −17 % | −6 … −14 % |
| rot err @500 | −34 … −42 % * | −20 … −25 % | −17 … −27 % | −21 … −26 % |
| rot err final | −0.5 … −3.5 % (2/3 *) | −9 … −10 % | ≈ 0 | −9 % |
| impact dv / dw | −23 … −28 % * / −37 … −43 % * | −34 … −38 % * / −45 … −48 % * | −20 … −24 % * / −25 … −32 % * | −20 … −26 % * / −40 … −43 % * |
| max penetration | −97 … −98 % * | −96 … −97 % * | −97 … −98 % * | −96 … −97 % * |
| final spin err | −31 … −53 % (2/3 *) | −22 … −43 % * | −26 … −50 % * | −26 … −47 % (2/3 *) |

Absolute (test): BASE-64 pos@500 0.043–0.053 m, rot@500 24.8–28.3°, impact dw 5.4–6.0 rad/s, penetration 15–22 mm;
JL-6-R v6: 0.035 m, 16.4°, 3.4 rad/s, 0.4 mm.

**Ablation 7 (predicted vs Chrono geometry into decoder v6, `final_metrics.json` of `runs/local_ed2_dec_v6`):** with the
recorded Chrono contact set instead of the point encoder's, first-impact dv MAE 0.596 vs 0.604 m/s (test), dL 0.016 vs
0.014, rebound dv 0.143 vs 0.149, resting identical; the learned geometry costs nothing measurable — the residual decoder
error is the per-step contact-response ambiguity, not geometry.  Encoder-miss rate 0.17 % (test), 0.28 % (held-out).

**Ablation 3 (unconstrained Cartesian head vs friction-cone head, decoder v6 recipe, `runs/local_ed2_dec_v6_free`):** the
free head fits Chrono better at the decoder level (first-impact dv MAE 0.41 / 0.34 / 0.31 on val / test / held-out vs
0.63 / 0.60 / 0.62 for the cone head; dL equal) and gives equally strong rollouts with JL-6-R+H8 vs BASE-64+H8
(test: pos@500 −29 % *, rot@500 −38 % *, impact dv/dw −36 % * / −56 % *; held-out: pos@500 −35 % *, pos final −39 % *,
impact −44 % * / −53 % *; cone head: −39 / −46 / −28 / −57 and −28 / −34 / −28 / −46).  The cone head stays the primary
(interpretable, 0 violations by construction); the free head is the better Chrono imitator.

### 9.11 FINAL — cluster, 30 k steps, 3 seeds each, all episodes (255 test / 301 held-out geometry), 2026-08-28

Relative change of the paired per-episode median vs each of the three BASE-64 seeds (9 pairings; mean ± std; [fraction of
pairings whose 95 % bootstrap CI excludes 0]).  `*_ev255` re-evaluations in `results/cluster_ed/`,
`results/cluster_ed/summary_matched_30k.json`.

**A) Both models trained with the 8-step unrolled loss (JL-6-R v6+H8 vs BASE-64+H8):**

| metric | test | held-out geometry |
| --- | --- | --- |
| pos err @500 / @1000 / final | −16±5 % [0.8] / −21±8 % [0.8] / −15±11 % [0.4] | −25±2 % [1.0] / −15±2 % [0.8] / −17±4 % [0.7] |
| rot err @500 / @1000 / final | −33±5 % [1.0] / −16±2 % [0.9] / −3±1 % [1.0] | −29±4 % [0.9] / −9±5 % [0.3] / −3±1 % |
| symmetry-aware rot err final | −31±6 % | −3±15 % |
| impact dv / dw err | −26±6 % [1.0] / −41±3 % [1.0] | −25±4 % [1.0] / −48±3 % [1.0] |
| max penetration | −95±2 % [1.0] | −93±3 % [1.0] |
| final angular-speed err | −59±8 % [1.0] | −55±9 % [1.0] |

Absolute medians (test, mean over seeds): BASE-64+H8 pos@500 0.039 m, rot@500 22.6°, impact dw 5.6 rad/s, penetration
10.8 mm, spin 0.40 rad/s; JL-6-R v6+H8: 0.032 m, 15.0°, 3.3 rad/s, 0.4 mm, 0.16 rad/s.  The previous decoder (v5) with the
same training gives only −10 % rot@500 and no position gain: the solver prior is what carries the pose improvement.

**B) No unrolled loss (JL-6-R v6 and JL-6 v6 vs BASE-64):**

| metric | JL-6-R v6, test | JL-6-R v6, held-out | JL-6 v6, test | JL-6 v6, held-out |
| --- | --- | --- | --- | --- |
| pos err @500 / @1000 | −26±6 % [0.8] / −23±3 % [0.9] | −28±3 % [1.0] / −19±5 % [0.7] | −19±9 % [0.8] / −19±4 % [0.8] | −21±4 % [0.7] / −16±6 % [0.6] |
| rot err @500 | −32±6 % [1.0] | −30±6 % [0.6] | −22±5 % [0.3] | −22±4 % [0.1] |
| impact dv / dw | −29 % [1.0] / −39 % [1.0] | −34 % [1.0] / −46 % [1.0] | −23 % [1.0] / −29 % [1.0] | −29 % [1.0] / −41 % [1.0] |
| penetration / spin | −97 % [1.0] / −37 % [0.8] | −96 % [1.0] / −32 % [0.9] | −97 % [1.0] / −30 % [0.7] | −96 % [1.0] / −31 % [0.6] |

Conclusion: with the impulse decoder in the loop, orientation error at 0.5 s drops by about a third and position error
by 15–28 % on both splits, first-impact angular-velocity error by 40–48 %, penetration by >90 % and residual spin at rest
by >50 %, at matched training; the 2 s orientation is saturated (chaotic settling) for every model including the
exact-contact oracle.  Gates #1–#7 of the plan pass (§9.5–9.10).
