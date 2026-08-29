# ContactROM Broad-Phase Collision Pipeline Integration Plan

**Status:** proposed implementation plan  
**Scope:** extend the current ReCoM box-drop study with an explicit, GPU-friendly broad-phase collision pipeline  
**Terminology:** this document uses the standard term **broad phase**. The directory name `board-phase` is retained to match the requested project layout.

## 1. Decision summary

ReCoM should adopt the **architecture of the Genesis rigid-body broad-phase pipeline**, without making any particular Genesis traversal algorithm part of the research contribution.

The target recurrent simulation loop is:

```text
scene geometry and state
    -> static collision-pair filtering
    -> batched GPU world-AABB update
    -> conservative broad-phase traversal
    -> compact candidate-pair buffer
    -> pair-relative feature gathering
    -> neural contact-information encoder
    -> per-environment contact-token aggregation
    -> NeDM/NRD neural dynamics and contact solver
    -> physics-grounded state integration
    -> repeat from the predicted state
```

The broad phase is an algorithmic, conservative routing stage. It should answer:

> Which body or geometry pairs could be in contact, near contact, or enter contact during the next simulation step?

It should **not** answer whether contact actually exists. That remains the responsibility of the neural contact encoder, which predicts an empty contact set or explicit contact tokens for every candidate pair.

The first implementation may use parallel all-vs-all traversal over a statically filtered pair list because the current study contains very few bodies. Sweep-and-prune (SAP), spatial hashing, LBVH, or another GPU-friendly traversal can be introduced behind the same interface when benchmarks justify it.

## 2. Motivation

The completed Study 1 validates the learned narrow-phase/contact interface for one predetermined box-ground pair:

```text
box geometry and pose -> learned contact tokens -> frozen NeDM -> next box state
```

The learned point encoder matches the analytic box-plane oracle within experimental noise on the existing distribution. However, the current runtime already knows the only possible pair, so it does not represent the scene-level collision pipeline required by a general rigid-body simulator.

Adding broad phase makes the system decomposition explicit and scalable:

1. Broad phase removes geometrically irrelevant pairs cheaply.
2. The neural contact encoder is evaluated only on conservative candidates.
3. Candidate pairs and neural contact tokens stay on the GPU.
4. The same pipeline is used during data preparation, supervised training, and recurrent rollout.
5. The contact encoder can later generalize from box-plane pairs to arbitrary body-body pairs without changing the broad-phase contract.

The intended research claim becomes:

> A conventional GPU-friendly broad phase performs conservative scene routing, while a learned geometry-conditioned contact encoder replaces narrow-phase contact generation and a neural reduced dynamics model replaces contact solving and state evolution.

The specific broad-phase algorithm is a systems choice and an ablation, not the central learning contribution.

## 3. Existing Genesis reference paths

The local Genesis checkout examined for this plan is:

```text
/home/harry/Genesis
```

The main Genesis broad-phase implementation is:

```text
/home/harry/Genesis/genesis/engine/solvers/rigid/collider/broadphase.py
```

Related reference paths are:

| Purpose | Genesis path |
| --- | --- |
| Broad-phase traversal constants and automatic-selection documentation | `/home/harry/Genesis/genesis/constants.py` |
| Rigid-solver broad-phase option | `/home/harry/Genesis/genesis/options/solvers.py` |
| Runtime traversal selection and collision entry point | `/home/harry/Genesis/genesis/engine/solvers/rigid/rigid_solver.py` |
| Collider build, valid-pair initialization, and broad-to-narrow dispatch | `/home/harry/Genesis/genesis/engine/solvers/rigid/collider/collider.py` |
| SAP and parallel all-vs-all traversal kernels | `/home/harry/Genesis/genesis/engine/solvers/rigid/collider/broadphase.py` |
| Current narrow-phase consumer of the candidate-pair buffer | `/home/harry/Genesis/genesis/engine/solvers/rigid/collider/narrowphase.py` |
| World-AABB update kernel | `/home/harry/Genesis/genesis/engine/solvers/rigid/abd/forward_kinematics.py` |
| `ColliderState`, candidate counts, candidate-pair buffers, and valid-pair data | `/home/harry/Genesis/genesis/utils/array_class.py` |
| Quadrants-to-Torch zero-copy helpers | `/home/harry/Genesis/genesis/utils/misc.py` |
| Quadrants dependency/version | `/home/harry/Genesis/pyproject.toml` |

At the time of inspection, Genesis automatically uses:

- SAP on CPU.
- Parallel all-vs-all on a homogeneous GPU scene.
- SAP on GPU when hibernation or heterogeneous entities require it.

Genesis's SAP implementation uses a warm-started insertion sort along one axis and then checks full 3D AABB overlap. Its all-vs-all implementation traverses a prefiltered valid-pair list in parallel and compacts passing pairs with atomic operations.

ReCoM should adopt the **pipeline structure and data movement**, not depend on private Genesis collider objects. The preferred implementation is a small ReCoM-owned broad-phase interface with a Quadrants backend informed by the Apache-2.0 Genesis implementation and carrying appropriate attribution.

## 4. Scope and non-goals

### 4.1 In scope

- Scene/body/geometry registration for broad phase.
- Static valid-pair filtering.
- Current, expanded, and swept AABB construction.
- Batched GPU candidate traversal and compaction.
- Candidate-pair records in the ReCoM dataset.
- Candidate-conditioned contact-encoder training.
- Live broad-phase recomputation from predicted poses during rollout.
- A parity re-run of the original box-drop study.
- A nontrivial box-drop extension containing multiple potential collision pairs.
- Accuracy, recall, memory, and runtime attribution by pipeline stage.

### 4.2 Not in scope for this study

- Learning the broad phase itself.
- Backpropagating through discrete candidate selection.
- Replacing every spatial data structure with a neural network.
- Arbitrary nonconvex body-body contact.
- Stable multi-body stacking.
- Articulated robots or policy learning.
- Claiming that box-plane neural contact is faster than the analytic formula.

Those are later studies built on the interface established here.

## 5. Functional requirements

### 5.1 Correctness

1. Every Chrono-reported contact pair must be present in the broad-phase candidate set at the same pre-step state.
2. Every first-impact pair should be present at least one step before the reported impact when the configured look-ahead policy requests it.
3. Broad-phase overflow must be detected and reported. Candidate pairs must never be silently dropped.
4. CPU and GPU implementations must agree as unordered sets for the same bounds, filters, and configuration.
5. Candidate generation must be reproducible from stored state and metadata.

### 5.2 Performance

1. World bounds, pair lists, candidate counts, and candidate IDs remain GPU resident during rollout.
2. Candidate IDs feed the Torch contact encoder without a CPU round trip.
3. Work scales with the statically valid pair set rather than an unfiltered global body cross-product.
4. Candidate output is sparse and compact, not a dense `N x N` tensor passed through the rest of the model.
5. The traversal implementation is replaceable without changing dataset or model contracts.

### 5.3 Contact-learning behavior

1. A candidate pair does not imply contact.
2. Candidate-but-separated pairs are retained as hard negatives.
3. The contact encoder predicts activation, signed separation, paired surface anchors, normal, cardinality, latent contact features, and uncertainty.
4. A scene with no candidates produces an empty/zero contact context and follows the exact free-flight prior.

## 6. Proposed ReCoM architecture

### 6.1 Scene and geometry registry

Introduce a minimal scene representation that is independent of Chrono and the neural model:

```text
BodyRecord
    body_id
    dynamic/fixed flag
    geometry range
    material identifier
    mass/inertia reference

GeometryRecord
    geometry_id
    owning body_id
    local AABB
    collision group/mask
    geometry encoder asset key
    optional plane/top-surface metadata for Study 1
```

Rigid geometry features and local bounds are computed once. Runtime broad phase receives only current poses, twists, local bounds, the valid-pair table, and configuration.

### 6.2 Static valid-pair filtering

At scene build time, generate a compact list of geometry pairs that are allowed to collide. Eliminate:

- Same-geometry pairs.
- Same-body/self pairs unless self-collision is explicitly enabled.
- Fixed-fixed pairs.
- Collision-group or collision-mask mismatches.
- Explicitly excluded pairs.
- Adjacency exclusions when articulation is added later.

Store the valid-pair list in a stable lexicographic order. This list is shared by all homogeneous batched environments. Heterogeneous scenes may use per-environment lists or a padded list plus an environment mask later.

### 6.3 World AABB construction

For a local AABB with center `c_local` and half extent `h_local`, body/world transform `(R, p)` gives:

```text
c_world = R c_local + p
h_world = abs(R) h_local
aabb_min = c_world - h_world
aabb_max = c_world + h_world
```

For the current boxes, this formula avoids transforming all eight corners. A general mesh can use its cached local bounds with the same formula.

Three bound modes should be supported:

1. **Current:** exact world AABB at `t`.
2. **Expanded:** current AABB enlarged by a configured proximity/query margin.
3. **Swept:** union of the current and predicted endpoint bounds plus a conservative rotational pad.

The pairwise candidate threshold must have unambiguous semantics. If `m_query` is the desired maximum AABB separation for a pair, either expand one member by `m_query` or expand both by `m_query / 2`; do not accidentally double the intended margin.

A first conservative rotational bound for body radius `r` is:

```text
rotation_pad = min(r * ||omega|| * dt, 2 * r)
```

The endpoint center uses the known/predicted linear motion. For data preparation, the recorded next pose can be used only for diagnostics; the production candidate rule must depend on information available at state `t`, such as current pose, twist, gravity, `dt`, and a documented safety margin.

The broad-phase query margin should cover:

```text
contact reporting/proximity margin
+ relative linear travel during dt
+ conservative rotational surface travel during dt
+ numerical safety margin
```

### 6.4 Traversal backend interface

Define a backend-independent contract similar to:

```python
BroadPhaseOutput = broadphase.query(
    aabb_min,          # (B, N_geom, 3), GPU
    aabb_max,          # (B, N_geom, 3), GPU
    valid_pairs,       # (P, 2), GPU
    pair_mask=None,    # optional heterogeneous/filter mask
)
```

Output:

```text
candidate_pairs       (capacity, B, 2) or compact flat pairs plus environment IDs
candidate_counts      (B,)
candidate_pair_keys   optional stable pair IDs
coarse_pair_features  optional AABB gap/overlap features
overflow/status       (B,)
```

The contact encoder must depend only on this contract.

### 6.5 Initial backends

Implement in this order:

1. **CPU reference backend**
   - NumPy or Torch CPU.
   - Simple valid-pair iteration.
   - Used for tests, validation, and small dataset tools.

2. **Quadrants parallel-pair GPU backend**
   - Traverse the statically valid pair list over `(pair, environment)`.
   - Test three-axis expanded/swept AABB overlap.
   - Compact candidates into fixed-capacity buffers.
   - Expose zero-copy Torch views.

3. **Optional Quadrants SAP backend**
   - Add only after the first performance sweep identifies a useful regime.
   - Preserve the same input/output contract.

4. **Future spatial hash or LBVH backend**
   - Consider for hundreds or thousands of bodies per environment.

The automatic dispatcher should eventually use measured crossover curves based on backend, number of environments, valid-pair count, body count, and sparsity. The present box study should not wait for the optimal large-scene traversal.

### 6.6 Candidate compaction and ordering

GPU atomics can produce nondeterministic candidate order even when the candidate set is identical. The downstream network should be pair-order invariant, but dataset reproducibility and debugging benefit from stable ordering.

Use one of:

- Stable compaction in valid-pair-table order.
- Lexicographic sort by `(environment_id, geom_a, geom_b)` after compaction.
- Treat candidate ordering as explicitly unordered and canonicalize only when serializing records.

The first implementation should prefer stable valid-pair order if it does not materially harm performance.

### 6.7 Pair feature gathering

For each candidate `(i, j)`, gather on GPU:

- Geometry asset identifiers and cached geometry tokens.
- Body identifiers.
- Relative pose `T_ij`.
- Relative linear and angular velocity for future swept/TOI models.
- Material parameters.
- Timestep.
- Optional coarse AABB separation features for diagnostics or conditioning.

For the minimal box-plane adapter, transform the box into the candidate plane/platform frame so that the existing encoder can retain its pair-relative `+Z` plane convention. Do not treat world `+Z` as a permanent interface assumption.

### 6.8 Contact encoding and scene aggregation

Run the contact encoder over the compact candidate batch. For each pair it produces up to `K` contact slots. Attach environment and body-pair IDs to the slots, then aggregate/scatter them into the per-environment contact context consumed by NeDM.

The aggregation must distinguish:

- No broad-phase candidate: encoder not called.
- Candidate but encoder predicts no contact: empty pair contact set.
- Candidate with one or more predicted contact tokens.

The transition model may initially reuse its existing DeepSets contact pooling. Later multi-body work will require body-indexed or graph-based aggregation so that impulses are assigned to the correct bodies.

## 7. Proposed ReCoM file layout

Suggested new files:

```text
recom/broadphase/__init__.py
recom/broadphase/types.py
recom/broadphase/aabb.py
recom/broadphase/filtering.py
recom/broadphase/base.py
recom/broadphase/reference.py
recom/broadphase/quadrants_backend.py
recom/broadphase/pipeline.py
```

Suggested responsibilities:

| File | Responsibility |
| --- | --- |
| `types.py` | Body/geometry records, valid pair table, broad-phase configuration and output types |
| `aabb.py` | Local-to-world, expanded, and swept AABB construction |
| `filtering.py` | Static valid-pair construction and collision masks |
| `base.py` | Backend protocol and validation helpers |
| `reference.py` | Deterministic CPU implementation |
| `quadrants_backend.py` | GPU traversal, compaction, capacity/status, zero-copy Torch bridge |
| `pipeline.py` | Scene build, runtime query, pair gathering, and encoder dispatch orchestration |

Likely existing-file changes:

```text
recom/config.py
recom/data/schema.py
recom/data/storage.py
recom/data/dataset.py
recom/data/validate.py
recom/sim/chrono_box_drop.py
recom/train/rollout.py
recom/train/train_contact.py
recom/train/train_transition.py
recom/models/contact_encoder.py
scripts/generate_dataset.py
scripts/validate_dataset.py
scripts/train_contact.py
scripts/train_transition.py
```

The implementation should preserve a compatibility path for the existing `recom.boxdrop.v1` datasets and checkpoints.

## 8. Dataset and schema changes

### 8.1 Schema version

Introduce a new schema version, for example:

```text
recom.boxdrop.bp.v2
```

Do not rewrite existing Study-1 data in place. Either derive a v2 broad-phase sidecar from existing episodes or regenerate a new named dataset.

### 8.2 Scene-level metadata

Store:

- Body table and geometry table.
- Local geometry bounds.
- Collision masks and explicit exclusions.
- Stable valid-pair table.
- Broad-phase backend identifier.
- Bound mode: current, expanded, or swept.
- Query/contact margins.
- Look-ahead horizon and swept-bound formula version.
- Candidate capacity.
- Numeric precision.
- Quadrants and backend versions when applicable.

### 8.3 Per-step candidate data

For state `states[k]`, store or reproducibly derive:

- Candidate count.
- Candidate geometry/body pair IDs.
- Optional AABB separation vector or scalar gap.
- Candidate reason flags: current overlap, expanded overlap, swept overlap.
- Whether the pair has a Chrono contact at `k`.
- Whether the pair first contacts at `k+1` for look-ahead diagnostics.
- Overflow/status flags.

Candidate records align with the existing convention:

```text
(state[k], broadphase[k], contacts[k]) -> state[k+1]
```

### 8.4 Data preparation strategy

Avoid launching a GPU kernel separately from every Chrono multiprocessing worker at every millisecond.

Recommended workflow:

1. Chrono records states and authoritative contacts as it does today.
2. A deterministic postprocessing stage batches many frames on the GPU.
3. The same production broad-phase rule is evaluated from `state[k]` information.
4. Candidate records are written as a sidecar or merged into the v2 episode.
5. Validation compares candidates with Chrono contact events.

An inline CPU reference mode can remain available for smoke tests. Postprocessing does not change the logical time alignment because candidates are a deterministic function of the recorded pre-step state and frozen scene metadata.

## 9. Training changes

### 9.1 Candidate-conditioned contact-query dataset

Replace global frame sampling with pair-aware candidate sampling. Maintain balanced categories within candidates:

1. Candidate but safely separated: hard negative.
2. Candidate and within the configured near-contact margin.
3. First impact.
4. Rebound/repeated contact.
5. Persistent/resting contact.

Keep a small diagnostic sample of noncandidate pairs to verify that the broad phase is not excluding positives, but do not waste contact-encoder training capacity on arbitrarily distant pairs.

### 9.2 Losses

The existing set loss remains valid conditional on a candidate pair. Add reporting that separates:

- Broad-phase false negatives.
- Conditional contact-encoder false negatives.
- End-to-end pipeline false negatives.

Do not optimize broad-phase margin using contact-encoder gradients. Select it using validation recall, candidate count, and runtime.

### 9.3 Transition training

For teacher-forced transition training, use stored or freshly recomputed candidates from ground-truth states. For unrolled training and evaluation, recompute broad phase from every predicted state.

Do not replay recorded candidate pairs during closed-loop rollout; they become invalid when the predicted pose diverges from the recorded trajectory, just as recorded contact tokens do.

### 9.4 Gradient boundary

The first pipeline treats candidate selection as a nondifferentiable routing decision. Gradients flow through:

```text
candidate pair features -> contact encoder -> contact aggregation -> NeDM -> predicted state
```

but not through the discrete broad-phase selection. Conservative coverage is enforced algorithmically and evaluated explicitly.

## 10. Reworked box-drop study

The current Study-1 result should be preserved and extended in three controlled phases.

### 10.1 Study 1-BP0: candidate-pipeline smoke test

**Purpose:** verify schema, time alignment, CPU/GPU agreement, candidate serialization, and rollout plumbing.

Scene:

- One dynamic box.
- One fixed finite ground box, matching the present Chrono scene.
- Existing fixed geometry and physics settings.
- 50 deterministic smoke episodes.

Tests:

- Hand-built far, near, touching, and penetrating poses.
- Face-, edge-, and corner-dominant orientations.
- Candidate behavior under current, expanded, and swept bounds.
- One-step look-ahead coverage at maximum expected fall speed.
- CPU versus Quadrants candidate-set equality.
- Candidate-buffer overflow test with intentionally tiny capacity.

Expected result:

- Far-away box-ground pair may be rejected.
- The pair appears before the configured contact-query boundary.
- Every Chrono contact lies inside the candidate set.

### 10.2 Study 1-BP1: parity re-run of the original pilot

**Purpose:** prove that inserting broad phase does not change the successful learned-contact result.

Data:

- Existing `pilot1b` geometry and release distribution.
- Prefer deriving broad-phase records from the existing episodes first so every model is compared on identical trajectories.
- Optionally regenerate a named v2 pilot after smoke validation to exercise the complete collection workflow.

Configurations:

| ID | Pair routing | Contact source | Transition model | Purpose |
| --- | --- | --- | --- | --- |
| BP-A | Always evaluate the known pair | Point encoder 30k | Frozen oracle-trained NeDM | Existing learned-contact reference |
| BP-B | Broad phase | Analytic box-plane contact | Frozen oracle-trained NeDM | Broad-phase-only oracle |
| BP-C | Broad phase | Point encoder 30k | Frozen oracle-trained NeDM | Full proposed modular pipeline |
| BP-D | Broad phase | Patch encoder 30k | Frozen oracle-trained NeDM | Representation/runtime comparison |
| BP-E | Broad phase | Point encoder | Jointly fine-tuned NeDM | Optional; not required if frozen model already matches oracle |

Primary comparison:

```text
BP-A versus BP-C
```

If the candidate rule has no false negatives, these should match within numerical and episode-sampling noise.

Margin ablation:

- Exact current AABB overlap.
- Contact-reporting margin.
- Fixed near-contact expansion.
- Swept/velocity-aware expansion.

The exact-current bound is expected to activate too late and serves as a negative-control ablation.

### 10.3 Study 1-BP2: nontrivial multi-pair box drop

**Purpose:** make broad-phase rejection and GPU pair compaction measurable without yet changing the learned box-plane contact problem.

Scene family:

- One dynamic box.
- A finite landing platform or ground tile.
- Multiple fixed distractor platforms/bodies.
- Fixed-fixed pairs filtered at build time.
- All dynamic-static pairs initially valid.
- Randomized horizontal release location so the landing platform is not encoded by a fixed pair index.

Recommended scene sizes:

```text
1, 8, 32, and 128 static candidate bodies/platforms
```

Design constraints:

- Preserve box-to-planar-top contact semantics for the first multi-pair experiment.
- Place the box far enough from platform edges that side-face contact cannot occur unless explicitly included.
- Include some near-miss platforms that enter broad phase but do not contact, creating realistic hard negatives.
- Include distant distractors that should be rejected immediately.
- Ensure the correct landing body varies across episodes.
- Record and verify the contacted body ID.

The contact encoder receives each candidate in the platform-relative frame. Its output must be associated with the correct body pair before scene aggregation.

This phase validates:

- Scene-level pair routing rather than temporal gating of one known pair.
- Static valid-pair filtering.
- Dynamic candidate reduction.
- Pair-ID propagation through the neural contact and dynamics pipeline.
- Runtime scaling with potential pair count.

### 10.4 Study 1-BP3: optional dynamic-condition extension

After BP1 and BP2 pass, add Phase 1C initial linear/angular velocity randomization. This stresses swept candidate generation, grazing approaches, and next-step contact coverage. It should not block the initial broad-phase integration.

## 11. Hypotheses

### H-BP1: conservative coverage

Expanded/swept broad phase includes every Chrono contact pair and every configured imminent-contact pair in the held-out data.

### H-BP2: pipeline parity

Broad-phase-gated learned-contact rollouts reproduce the always-evaluate learned-contact baseline within experimental noise.

### H-BP3: useful candidate reduction

In multi-platform scenes, broad phase eliminates most statically valid but spatially irrelevant pairs before neural contact inference.

### H-BP4: scalable GPU routing

Broad-phase cost plus compact pair gathering grows substantially more slowly than evaluating the contact encoder on every statically valid pair as scene body count increases.

### H-BP5: modular attribution

Failures can be assigned separately to candidate routing, contact prediction, and neural dynamics rather than being reported only as final rollout error.

## 12. Metrics and provisional gates

### 12.1 Static and dynamic pair metrics

- Total possible geometry pairs.
- Statically valid pair count.
- Candidate pairs per environment and timestep.
- Candidate reduction ratio.
- Candidate-count distribution and maximum.
- Candidate-buffer utilization and overflow count.

### 12.2 Broad-phase correctness

- Current-contact pair recall.
- Near-contact pair recall within the configured query margin.
- Next-step first-impact pair recall.
- Candidate lead time before first impact.
- CPU/GPU candidate-set disagreement.

Provisional gates:

| Gate | Criterion |
| --- | --- |
| Current-contact coverage | Zero Chrono contact pairs omitted on validation/test |
| Imminent-contact coverage | Zero next-step first-impact pairs omitted for the selected swept policy |
| CPU/GPU parity | Identical unordered candidate sets |
| Overflow safety | Zero unreported overflows; deliberate overflow test fails loudly |
| Reproducibility | Candidate records regenerate exactly as sets from state and metadata |

If zero misses are observed, also report a statistical upper bound on the miss probability rather than interpreting the empirical zero as a universal guarantee.

### 12.3 Conditional contact metrics

Report existing activation, timing, distance, point, normal, cardinality, and calibration metrics **conditional on candidate pairs**.

Also report end-to-end metrics in which a broad-phase miss counts as an inactive prediction. This prevents the contact encoder from receiving credit for examples it never evaluated.

### 12.4 Dynamics metrics

Retain the Study-1 metrics:

- One-step `dv`/`dw` error by regime.
- First-impact velocity-change error.
- Position/orientation error at rollout horizons.
- Symmetry-aware final box orientation/corner error.
- Rebound apex/time.
- Penetration and artificial-energy growth.

Broad-phase parity gate:

> BP-C should show no statistically meaningful degradation from BP-A; target <=2% change in selected medians, while remaining inside the existing <=10% oracle degradation gate.

### 12.5 Runtime metrics

Measure on CPU and GPU:

- Local-to-world AABB update time.
- Traversal and compaction time.
- Pair-feature gathering time.
- Contact-encoder time.
- NeDM time.
- Full simulation-step time.
- Peak memory.
- Candidate pairs and contact-encoder calls per second.

Sweep over:

- Body count per environment.
- Statically valid pair count.
- Candidate sparsity.
- Number of batched environments.
- Backend/traversal choice.

Synchronize GPU timing correctly and separate one-time compilation/warmup from steady-state performance.

## 13. Tests

### 13.1 Unit tests

- Local AABB to world AABB under translations and rotations.
- Expanded-margin semantics do not double the requested pair distance.
- Swept translation covers both endpoints.
- Rotational pad covers sampled intermediate orientations.
- Strict/non-strict boundary behavior at touching AABBs.
- Static pair filtering.
- Stable pair IDs and canonical pair order.
- Empty scene, no valid pairs, and one valid pair.
- Capacity boundary and overflow status.

### 13.2 Cross-backend tests

- CPU versus Quadrants candidate-set equality.
- Single versus batched environments.
- Float32 boundary cases with a documented tolerance policy.
- Stable behavior under candidate ordering differences.

### 13.3 Chrono integration tests

- Every recorded box-ground contact is a candidate.
- First-impact look-ahead coverage.
- Dataset time alignment remains `(s_k, bp_k, c_k) -> s_{k+1}`.
- Broad-phase records regenerate from stored state and metadata.

### 13.4 Model integration tests

- No candidates produces exact free flight.
- Candidate/no-contact produces no contact residual.
- Candidate contacts retain the correct environment/body pair.
- Always-evaluate and broad-phase-gated paths agree when the pair is present.
- Closed-loop rollout recomputes candidates from predicted poses.

## 14. Implementation milestones

### M-BP0: contract and CPU reference

Deliverables:

- Broad-phase types and backend protocol.
- Static valid-pair builder.
- Current/expanded/swept AABB implementation.
- Deterministic CPU reference.
- Unit tests.

Exit criterion: hand-built candidate sets and margin semantics pass.

### M-BP1: dataset/schema integration

Deliverables:

- v2 schema or sidecar format.
- Candidate postprocessor.
- Validator and diagnostics.
- BP0 smoke dataset.

Exit criterion: zero missed current/next-step Chrono contact pairs in the smoke set.

### M-BP2: Quadrants GPU backend

Deliverables:

- GPU AABB/traversal/compaction path.
- Zero-copy Torch candidate view.
- CPU/GPU parity tests.
- Overflow handling.

Exit criterion: exact candidate-set parity and no host round trip during rollout.

### M-BP3: model and rollout integration

Deliverables:

- Candidate-conditioned contact dataset.
- Pair gather/scatter adapter.
- Broad-phase-aware contact source.
- Live candidate recomputation in rollouts.

Exit criterion: original Study-1 learned-contact result reproduced in BP1.

### M-BP4: nontrivial multi-pair study

Deliverables:

- Multi-platform/distractor scene generator.
- Pair-ID-aware dataset and evaluation.
- Scaling benchmarks.

Exit criterion: conservative coverage plus meaningful candidate reduction without rollout degradation.

### M-BP5: report and architecture freeze

Deliverables:

- Final tables and failure cases.
- Backend crossover plot.
- Frozen broad-phase/contact interface for the next body-body study.

Exit criterion: the pair-routing interface is sufficiently general to replace box-plane geometry with arbitrary pair encoders without another schema redesign.

## 15. Risks and mitigations

| Risk | Consequence | Mitigation |
| --- | --- | --- |
| Exact AABB overlap activates too late | Missed near-contact tokens and impulses | Expanded/swept bounds; next-step recall gate |
| Query margin is too large | Too many encoder calls | Validate recall/candidate-count Pareto curve |
| Query margin is too small | Catastrophic false negatives | Conservative formula and zero-miss gate |
| Plane/infinite geometry has awkward bounds | Always-on or invalid candidates | Use finite ground/platform collision boxes in Study 1; define special infinite-plane handling explicitly later |
| GPU atomic compaction changes order | Nondeterministic serialized records | Stable valid-pair compaction or canonicalize before storage |
| Candidate buffer overflows | Silent loss of contacts | Status buffer, hard failure, capacity telemetry |
| Full Genesis dependency is heavy/private | Fragile integration | ReCoM-owned interface and minimal Quadrants backend |
| CPU Chrono workers contend for GPU | Slow or unreliable generation | Batched postprocessing rather than per-step GPU calls |
| Broad phase appears useful only because of trivial distractors | Weak systems result | Report both controlled scaling and later body-body follow-up; avoid overclaiming |
| Contact encoder is hard-coded to world ground frame | Cannot process multiple platforms | Use a candidate-pair/platform-relative frame |
| Stored candidates are replayed during rollout | Incorrect contact routing after divergence | Always recompute from predicted state |

## 16. Recommended execution order

1. Freeze the current Study-1 results and checkpoints as the no-broad-phase reference.
2. Implement the backend-independent contract and CPU reference.
3. Add expanded/swept bounds and validate next-step contact coverage.
4. Derive broad-phase sidecars for the existing smoke and `pilot1b` datasets.
5. Add the Quadrants parallel-pair backend and zero-copy Torch bridge.
6. Re-run the frozen point-encoder/frozen-NeDM pipeline with broad-phase gating.
7. Confirm BP-A/BP-C parity before changing scene geometry.
8. Add the multi-platform/distractor dataset and pair-ID-aware contact adapter.
9. Benchmark candidate reduction and stage-level runtime across scene/batch sizes.
10. Add or select SAP/grid/LBVH only if the measured crossover requires it.
11. Freeze the scene-to-candidate and candidate-to-contact contracts.
12. Advance to arbitrary two-body contact as the next research study.

## 17. Definition of done

The broad-phase extension is complete when:

- ReCoM has a documented, backend-independent scene-to-candidate interface.
- CPU and GPU implementations agree as candidate sets.
- The selected expanded/swept rule omits no current or next-step Chrono contacts in the test corpus.
- Candidate pairs remain GPU resident and feed the contact encoder without a CPU copy.
- Broad-phase-gated learned-contact rollouts reproduce the existing always-evaluate Study-1 result.
- A multi-pair box-drop variant demonstrates meaningful candidate rejection and favorable end-to-end scaling.
- Runtime and error are attributed separately to broad phase, contact encoding, and neural dynamics.
- The interface can accept a future arbitrary body-body contact encoder without redesigning the dataset or recurrent simulation loop.

## 18. Expected research outcome

The revised box-drop study will no longer claim only that a neural contact encoder can imitate box-plane contact information. It will demonstrate a complete modular neural simulation pipeline:

```text
GPU scene routing
    + learned narrow-phase/contact representation
    + learned reduced contact dynamics
    + physics-grounded integration
```

The study will also make clear which parts remain classical and why: the broad phase is retained because it is conservative, inexpensive, hardware-friendly, and does not require solving detailed surface geometry. The learned contribution begins at candidate-pair contact inference and continues through recurrent dynamics.
