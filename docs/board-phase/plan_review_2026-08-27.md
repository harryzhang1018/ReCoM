# Review: Broad-Phase Integration Plan

Reviewed 2026-08-27 against the code at HEAD (`b7e0526`, "docs: project webpage") and `docs/PROGRESS.md` (Study 1 complete: D_point matches the oracle within noise on held-out geometry; near-contact recall 99.2% vs 99.5% gate is the one open miss).

## Verdict

The plan is well-constructed and unusually careful about the things that actually bit you in Study 1 (time alignment, gate semantics, no replay of stale contacts, attribution metrics). Two overall themes in this review:

1. **BP0/BP1 can be much lighter than planned.** Given how the code actually works, parity is an evaluation-only exercise and most of the proposed infrastructure (schema v2, sidecars, Quadrants backend, overflow machinery) can be deferred.
2. **BP2 hides the real work, and one part of its design is currently impossible.** The near-miss hard-negative scheme cannot work with the existing encoder inputs — not "hard to learn," but information-theoretically impossible — so an encoder input extension must be added to the plan as a first-class item.

## What the plan gets right (checked against code)

- Recompute-from-predicted-state (§9.3) is already how the analytic source works in `rollout.py`; the principle is proven there.
- The candidate-set contract's batched `(B, N_geom, 3)` shape matches the existing batched rollout (`rollout_batch` is a single time loop vectorized over B episodes).
- "No candidate ⇒ exact free flight" (§5.3.4) is guaranteed by existing mechanics: inactive slots zero all 19 slot features (`nedm_adapter.py:36`), pooling gives zero context, and the hard gate zeroes the residual over the gravity prior (`transition.py:130-136`). No new model machinery needed.
- Keeping broad phase non-learned and non-differentiable (§9.4), retaining candidate-but-separated hard negatives (§9.1), and the three-way FN decomposition (§9.2) are all the right calls.

## Main suggestions

### 1. BP1 is evaluation-only — and the parity gate can be far stronger than "within noise"

BP-A vs BP-C needs **no retraining**: reuse the frozen `expC_point_30k` + frozen NeDM checkpoints with `--eval-only`. More importantly, with the hard gate the plan's statistical parity gate (§12.4, ≤2% median change) undersells what you can assert:

- No candidate ⇒ zero contact context ⇒ *identical* compute path to "encoder called, predicts inactive" (both produce all-zero features and a closed gate).
- Therefore BP-C's trajectory can differ from BP-A's **only** at steps where the encoder would fire on a non-candidate — and far-field FPR measured in Study 1 is 0.000.

So assert per-episode, step-level equivalence: gate-open steps of BP-C ⊆ gate-open steps of BP-A, and trajectories bit-identical except on episodes containing a suppressed far-field FP (expect ~none). This is a sharper, cheaper gate than re-deriving noise bands, and any violation is immediately diagnosable. Keep the statistical comparison only as a fallback report.

### 2. Don't store candidates at all in BP0/BP1 — derive on the fly, kill schema v2 until BP2

§8.3's own observation ("candidates are a deterministic function of the recorded pre-step state and frozen scene metadata") argues against storing them. The training cache is already an in-memory float32 rebuild of the episodes (`dataset.py`), and candidate generation for 1–128 pairs is a few fused tensor ops. Recommend:

- Compute candidates inside the dataset/collate and inside `ContactSource`, parameterized by a versioned `BroadPhaseConfig` (margin, bound mode, formula version) recorded in run metadata.
- Store nothing per-step. The "sidecar" becomes a *validation report* produced by `scripts/validate_dataset.py` (recall vs Chrono contacts, lead-time distribution), not a data product.
- Margin ablations then require zero regeneration, replay-staleness (§15 last risk) becomes structurally impossible, and the multiprocessing/GPU contention risk (§8.4) disappears entirely.
- Schema v2 is then needed only in BP2, where it has real content: body/geometry tables, platform extents, valid-pair table, and a per-slot `c_geom_b` pair ID — which `schema.py` currently lacks entirely (canonicalization hard-codes A=box, B=ground).

### 3. Make the production backend pure batched Torch; defer Quadrants entirely

For this study the workload is ≤128 valid pairs × B environments — a `(B, P)` broadcast AABB-overlap test is microseconds of tensor work. A plain Torch implementation:

- is GPU-resident and feeds the encoder with zero copies *by construction* (no Quadrants→Torch bridge to build or test);
- runs the same code on CPU and GPU, shrinking the §13.2 parity matrix;
- gets stable, deterministic candidate ordering for free (`torch.nonzero` is row-major; no atomics, no compaction nondeterminism — §6.6 and two rows of the risk table dissolve);
- needs no fixed-capacity buffers, so the overflow machinery (§5.1.3, §12.1) reduces to an assertion until a kernel backend exists.

Keep the backend protocol exactly as specified so SAP/LBVH/Quadrants can drop in later, but move all kernel work behind the "hundreds of bodies" trigger in §6.5. This deletes most of M-BP2. The research narrative is unaffected: a masked parallel all-vs-all over a statically filtered pair list *is* the Genesis-style traversal at this scale.

### 4. BP2's near-miss hard negatives are impossible for the current encoder — plan the input extension explicitly

Both encoders consume only `(half_extents, pos_z, R)`; the plane is hard-coded at z=0 with a +Z-biased normal decoder (`contact_encoder.py:112-121`, `:104`), and the encoders are *tested* for translation invariance in x,y. Consequence for §10.3: a box falling laterally 2 cm past a platform edge, at top-surface height, presents **exactly the same encoder input** as a box squarely landing on it — same `pos_z` relative to the top, same R — but opposite labels. That's not a hard example; it's label noise that will degrade the activation head, and at rollout time it opens the hard gate with a phantom impulse — precisely the round-1 Experiment D failure mode (near-boundary FPs at 10% were enough to wreck rollouts).

So §10.3's "existing encoder in the platform-relative frame" framing needs amending:

- Extend the pair features with the in-plane offset of the box relative to the platform and the platform's top half-extents (≈5 scalars in the platform frame), and retrain — cheap at ~25 min/encoder on the cluster.
- The analytic baseline needs the same treatment (finite-support box-plane contact) or BP-B/BP-C oracle comparisons are only valid on fully-supported episodes. Extending it means clipping the contact set to the platform footprint; if you'd rather not, constrain BP2 generation so any contacting box is fully supported, and say so in the plan.
- The alternative — spacing/height-separating platforms so near-misses never share the landing platform's top height — preserves the current encoder but guts the hard-negative story (§10.3's whole point). I'd take the input extension.

Related, worth one explicit line in §10.3: keep platform tops horizontal (the plan implies it). The 19 slot features mix frames (normal and velocities in world, contact point in box-local), which is only platform-frame-consistent while tops are horizontal; tilted platforms are a later, deliberate step.

### 5. Pin the margins with numbers you already have

The plan leaves margins symbolic; the dataset bounds them concretely:

- Max fall speed: clearance ≤1.5 m from rest ⇒ |v|max ≈ √(2·9.81·1.5) ≈ 5.4 m/s ⇒ ≤5.5 mm travel per 1 ms step (rebounds are slower, e=0.3; Phase 1C is deferred, so no initial velocities).
- Chrono reporting boundary: ~3 mm (envelope + margins, per PROGRESS §3/§8).
- AABB gap underestimates true gap, so an expanded margin m catches every pair with true gap ≤ m — the conservative direction.

So a **10 mm expanded margin** (3 reporting + 5.5 travel + safety) guarantees current-contact coverage plus ≥1 step of lead time, satisfying §5.1.1–2 by construction; sweep {5, 10, 50} mm for the §10.2 ablation (50 mm aligns candidates with the dataset's 5 cm `near_contact`/`far_free` boundary, making the BP-gated training distribution match Study 1's exactly). Swept/velocity-aware bounds add nothing while everything falls from rest — implement current+expanded now and move swept bounds to the 1-BP3/Phase-1C work item where they're actually stressed.

### 6. Gate and metric details

- State in §6.8/§9 that BP-C runs the **hard** gate. The soft-gate/hard-gate mismatch already cost you a round of D-experiment confusion; don't let it recur via the new code path.
- Add one metric to §12.3: encoder FPR *on candidate near-misses* (BP2), reported separately from far-field FPR — it's the number that predicts gate-opening failures, per round-1 D.
- §12.2's "identical unordered candidate sets" CPU/GPU gate conflicts with §13.2's float32 tolerance policy. With a Torch backend you can make both devices run identical float32 arithmetic; otherwise define the gate as: exact agreement for pairs with |gap − margin| > ε, tolerance band inside. Episodes stored float64, training cache float32 — decide which one the reference backend reads, and write it down.

### 7. Milestones, resequenced

M-BP0 (contract, filtering, AABB, reference + Torch backend, unit tests) stands. M-BP1 and M-BP2 mostly evaporate per suggestions 2–3: fold the validation report into BP0 and the `--eval-only` parity run into M-BP3. That makes the true shape of the project visible: **BP0+BP1 ≈ days; BP2 is the milestone with real scope** — Chrono multi-platform scene generator, schema v2 scene tables + per-slot pair IDs, encoder input extension + retraining, per-(frame, pair) category labeling in `ContactQueryDataset`, and the DeepSets pooling denominator (`cnt = active/K_SLOTS`) once slots come from multiple pairs. Budget accordingly.

A concrete de-risk step available today: `rollout.py` already has `min_gap_np`, and `ContactSource.__call__` (rollout.py:36-48) is the exact seam where the candidate check belongs — a 5-line vertical-gap gate there, evaluated with `--eval-only`, gives you the BP1 parity answer for the single-pair scene before any broadphase module exists.

## Minor

- Rename `docs/board-phase/` → `docs/broad-phase/` (the plan itself needs a terminology disclaimer for it; cheap to fix now, confusing forever in a public repo).
- BP-E: agree it's optional — Study 1 round 2 showed joint fine-tuning helps only when the encoder is the bottleneck, and the point encoder isn't.
- §3's Genesis paths under `/home/harry/Genesis` weren't verifiable from this review (outside the repo); the plan correctly avoids depending on them at runtime.
- Ground box is 6×6×0.2 m with top at z=0 and the box is released near the center, so BP1's single pair is safely inside the finite-support regime; the §15 infinite-plane risk row is already moot for Study 1.

## Suggested execution order (revised §16)

1. Freeze Study-1 checkpoints as reference (unchanged).
2. M-BP0: types, static filtering, current+expanded AABB, reference + Torch backend, unit tests, dataset validation report (recall + lead time, on the fly).
3. Interim: 5-line gap gate in `ContactSource`, `--eval-only` parity sanity check.
4. M-BP3: real backend behind `ContactSource` + dataset-side candidate conditioning; step-level BP-A/BP-C equivalence gate; margin sweep {5, 10, 50} mm.
5. M-BP4 (the big one): schema v2 scene tables + pair IDs, multi-platform generator, encoder input extension + retrain, finite-support analytic baseline (or constrained generation), near-miss FPR gate, scaling benchmark 1/8/32/128.
6. Swept bounds + Phase 1C velocities as 1-BP3.
7. Kernel backend (Quadrants/SAP) only if a follow-up study's scale demands it.
