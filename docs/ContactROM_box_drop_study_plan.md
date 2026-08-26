# ContactROM Study 1: Chrono Box-Drop Contact Encoder

Last updated: 2026-08-25 (America/Chicago)

## 1. Study purpose

The first ContactROM sign of life is a rigid box dropped from randomized heights and orientations onto a flat rigid ground plane in Project Chrono.

The experiment deliberately separates two learned modules:

1. **Contact-information encoder**

   \[
   \text{box/ground geometry and current poses}
   \rightarrow
   \text{contact information}
   \]

2. **Existing NeDM transition model**

   \[
   \text{box state history and contact information}
   \rightarrow
   \text{next box state}
   \]

The contact-information encoder is the new central contribution. The box-drop study validates its interface, representation, event timing, and effect on recurrent prediction before moving to multi-object manipulation or locomotion.

## 2. What this study can and cannot prove

### It should prove

- Chrono contact records can be collected with an unambiguous time and sign convention.
- A geometry-conditioned network can distinguish free flight, near contact, first impact, rebound, repeated impacts, and resting contact.
- Predicted contacts can replace oracle Chrono contacts at the NeDM input with limited rollout degradation.
- The geometry representation handles unseen box dimensions and orientations.

### It should not claim yet

- Generalization to arbitrary non-convex objects.
- Superiority to state-of-the-art GPU collision detection.
- Manipulation-policy or locomotion-policy performance.
- Multi-solver invariance.

An analytic box-plane calculation will almost certainly be faster and more accurate than a neural network. Its role is a strict correctness baseline. The box study validates the learned interface, not the final speed claim.

## 3. Hypotheses

### H1: oracle contact information helps NeDM

NeDM supplied with Chrono contact information will predict impact and post-impact velocities more accurately than state-only NeDM.

### H2: contact information can be learned from geometry and pose

A contact encoder operating on a canonical surface representation and relative pose will accurately predict contact activity, paired points, normal, and signed distance.

### H3: explicit plus latent contact representation is best

Explicit contact quantities plus a learned local contact latent will support better NeDM rollouts than either representation alone.

### H4: the interface generalizes across box geometry

A model trained on a range of box dimensions will retain contact accuracy on held-out dimensions and aspect ratios.

## 4. Chrono system definition

### 4.1 Bodies

- One dynamic rigid box.
- One fixed rigid ground plane.
- Gravity: \([0,0,-9.81]\ \mathrm{m/s^2}\).
- No actuator or control action in the primary experiment.

### 4.2 Contact method

Use one contact method and one collision pipeline for the primary dataset. Recommended starting choice:

- `ChSystemNSC` for a hard-contact, nonsmooth first study.
- A single documented Chrono collision system.
- Fixed solver, iteration limits, tolerances, collision envelope/margin, and timestep.

Do not mix NSC and SMC samples in the initial training dataset. A later solver-domain experiment can collect matched NSC and SMC trajectories with explicit provenance.

### 4.3 Physical parameters

Hold material parameters fixed initially to isolate geometry and event timing:

- One friction coefficient.
- One restitution coefficient.
- Zero rolling/spinning friction unless explicitly studied.
- Uniform density; recompute mass and inertia consistently when dimensions change.

Material randomization is a later extension because it changes response dynamics but not instantaneous geometric contact generation.

### 4.4 Time discretization

Initial recommendation:

- Physics timestep: \(\Delta t=10^{-3}\ \mathrm{s}\).
- Maximum episode duration: 2.0 s.
- Terminate early only after a documented settling condition persists for a fixed window.
- Record every physics step in the pilot.

The timestep must remain fixed within the first dataset. A separate timestep-holdout test can be introduced later.

## 5. Controlled experiment phases

### Phase 1A: fixed-box smoke test

Use one box geometry to debug:

- Logging and replay.
- Coordinate and normal conventions.
- Contact-set canonicalization.
- State-only versus oracle-contact NeDM.
- End-to-end recurrent execution.

This phase cannot validate the geometry encoder because the network can memorize the single box.

### Phase 1B: variable-box geometry test

Randomize box dimensions and reserve held-out dimensions/aspect ratios. This is the minimum experiment capable of testing the proposed unified geometry representation.

### Phase 1C: optional dynamic-condition extension

After the geometry encoder passes Phase 1B, add small randomized initial linear and angular velocities, followed later by material variation.

## 6. Randomization design

Use deterministic episode seeds and sample each episode independently.

| Variable | Primary distribution | Notes |
| --- | --- | --- |
| Ground clearance at release | Uniform, 0.10-1.50 m | Define height as minimum box-to-ground clearance, not center-of-mass height |
| Initial orientation | Uniform on \(SO(3)\) | Do not sample Euler angles independently |
| Box dimensions, Phase 1A | Fixed, e.g. 0.20 × 0.15 × 0.10 m | Debugging only |
| Box dimensions, Phase 1B | Each side 0.05-0.30 m with bounded aspect ratio | Use stratification so cubes and elongated boxes are represented |
| Initial linear velocity | Zero | Add small random values only in Phase 1C |
| Initial angular velocity | Zero | Add small random values only in Phase 1C |
| Horizontal position | Small bounded range or fixed origin | Physics is translation-invariant; variation mainly tests implementation |
| Material parameters | Fixed | Randomize only after geometry/event milestones pass |
| Density | Fixed | Mass and inertia change consistently with dimensions |

Reject and resample any initialization that intersects the ground or violates the specified clearance.

### Orientation coverage

Maintain orientation coverage diagnostics rather than trusting the random sampler alone. Report the distribution of the initially lowest feature:

- Face-dominant.
- Edge-dominant.
- Corner-dominant.

These categories do not need perfectly equal probability, but the dataset must contain enough examples of each impact mode.

## 7. Dataset scale and generation gates

### Smoke set

- 50 deterministic episodes.
- Manually inspect representative face, edge, and corner impacts.
- Verify record/replay and sign conventions.

### Pilot set

- 2,000 episodes.
- Approximately four million raw frames at 2 s and 1 ms, fewer with early settling.
- Use the pilot to estimate contact frequency, storage, generation cost, and learning curves.

### Main set

- Scale to approximately 10,000-20,000 episodes only after the pilot passes data and model gates.
- Preserve a fixed, never-trained-on test set before large-scale generation.
- Let pilot learning curves determine the final size rather than selecting the largest dataset by default.

## 8. Data organization

Use four logically separate tables or record groups.

### 8.1 Episode metadata

- `episode_id`
- Scenario seed.
- Chrono version/commit.
- Contact method, collision system, solver, integrator, and timestep.
- Solver iterations and tolerances.
- Collision envelope/margin.
- Box geometry identifier and exact dimensions.
- Collision mesh/hash.
- Density, mass, inertia tensor.
- Friction, restitution, and any other material parameters.
- Initial pose and twist.
- Planned termination time and actual termination reason.

### 8.2 Per-step box state

Record both the state needed by NeDM and the pose required to reconstruct scene geometry:

- `episode_id`, `step_id`, and timestamp.
- Box position \(x_t\).
- Box orientation quaternion \(R_t\), using one documented component order and canonical sign rule.
- Linear velocity \(v_t\).
- Angular velocity \(\omega_t\).
- Optional derived \(\Delta v_t\), \(\Delta\omega_t\), acceleration, and kinetic/potential energy.
- State-validity and settled flags.

Linear and angular velocity are the core box dynamics state requested for NeDM. Position and orientation must also be recorded because the contact encoder needs the current geometry pose and because the next scene must be reconstructed during recurrent rollout.

### 8.3 Raw Chrono contacts

For every contact at every step, preserve:

- Contact index within the frame.
- Body/contactable IDs for A and B.
- Contact point `pA`.
- Contact point `pB`.
- Contact frame `plane_coord`.
- Contact normal extracted from the X axis of that frame.
- Signed contact distance.
- Effective radius.
- Reaction force and torque when valid.
- NSC constraint offset when available.

Chrono's current `ReportContactCallback::OnReportContact` exposes these quantities directly ([Chrono API](https://api.projectchrono.org/classchrono_1_1_ch_contact_container_1_1_report_contact_callback.html)). Note that Chrono Multicore documentation warns that reported reaction forces and torques can be zero, so force labels must be validated before use ([Chrono Multicore contact container](https://api.projectchrono.org/classchrono_1_1_ch_contact_container_multicore.html)). Contact force is not required for the first contact-encoder target.

### 8.4 Derived canonical contacts and events

In addition to raw Chrono records, store:

- Fixed A/B convention: box and ground order.
- Contact points in world, box-local, and ground-local frames.
- Normal with one verified direction convention.
- `contact_active` and `near_contact` masks.
- Minimum analytic box-corner-to-plane gap.
- Padded \(K\)-slot contact set and mask.
- Contact mode: none, corner-like, edge-like, face-like.
- First-impact time.
- Rebound intervals.
- Resting-contact interval and settle time.

Never discard the raw Chrono values when creating these derived views.

## 9. Time-alignment specification

Each learning record must have the meaning

\[
(s_t,c_t) \rightarrow s_{t+1},
\]

where \(c_t\) is the contact manifold associated with the state and collision configuration used to advance from \(s_t\) to \(s_{t+1}\).

The recorder must document whether each state/contact value is sampled before collision detection, after collision detection but before solving, or after the complete Chrono step. Do not infer this later from timestamps.

Required validation:

1. A free-flight episode produces no active contacts and matches the analytic ballistic trajectory within integration tolerance.
2. The first recorded contact lies within one timestep of the analytic box-plane crossing bound.
3. Contact normal and signed-distance signs are verified using hand-constructed configurations above, touching, and penetrating the plane.
4. Replaying the stored initial condition and settings reproduces the recorded event sequence and state trajectory within tolerance.

## 10. Two complementary training datasets

### 10.1 Rollout-transition dataset

Use all chronological samples from box-drop episodes:

\[
(s_t,c_t,s_{t+1}).
\]

This preserves the true state distribution required for NeDM training and rollout evaluation.

### 10.2 Balanced contact-query dataset

Contact events occupy only a small portion of a drop. Build a second index/view containing balanced samples from:

- Far free space.
- Positive-clearance near contact.
- First impact.
- Penetrating/contact configurations.
- Rebound and repeated impacts.
- Resting contact.

Oversample windows around each contact transition and use hard-negative mining near the ground. This prevents a misleading high-accuracy model that predicts “no contact” everywhere.

## 11. Geometry representation for Study 1

### 11.1 Primary representation: cached surface-patch tokens

Represent the box using its canonical triangular collision surface rather than a box-specific width/length/height vector alone.

For each triangle/patch, store:

- Three vertices relative to face centroid.
- Face centroid in box-local coordinates.
- Face normal.
- Face area.
- Adjacent-face normals or dihedral-angle features.
- Sharp-edge/boundary flags.
- Physical scale.

For the flat ground, use an analytic plane token containing its origin, normal, and optional material identifier. The ground does not need a large artificial mesh.

The box patch embeddings are cached. At runtime, face centroids, normals, and vertices are transformed into the ground frame using the current box pose.

This design follows the contact-specific lesson from FIGNet: sparse face interactions preserve face-interior contacts that node-only representations can miss ([FIGNet](https://arxiv.org/abs/2212.03574)).

### 11.2 Required representation baselines

1. **Analytic primitive baseline**
   - Input: box half-extents and relative pose.
   - Exact analytic box-plane gaps/contact features.
   - Used for correctness, not as the general representation.

2. **Surface-point baseline**
   - Uniform surface samples plus guaranteed corner/edge samples.
   - Features: local position, normal, sampling weight, and scale.
   - PointNet++ encoder.

3. **Neural-SDF baseline**
   - Shape latent plus query locations in the object frame.
   - Predict signed distance and gradient/normal.

4. **Surface-patch model**
   - Recommended primary model.
   - Face adjacency plus local pair interaction with the plane.

The study should select the representation based on held-out geometry accuracy, near-contact recall, and cached runtime—not on training loss alone.

## 12. Contact-encoder architecture

### 12.1 Input

- Cached box object token.
- Cached box surface-patch tokens.
- Box-to-ground relative transform.
- Ground plane token.

Do **not** give linear/angular velocity to the geometry encoder in the primary experiment. Geometry and pose should determine the instantaneous contact information. Velocity remains part of the NeDM state and is joined later.

### 12.2 Patch encoder

- Shared 2-3 layer MLP for raw face features.
- Two to four lightweight message-passing layers over face adjacency.
- Residual connections and normalization.
- Cached canonical patch embeddings for each geometry.

For the 12-triangle box mesh this network is deliberately small. The same interface can later operate on coarsened arbitrary meshes.

### 12.3 Runtime pair-interaction block

For each box patch:

1. Transform its geometric features into the ground frame.
2. Compute relative-to-plane features.
3. Combine cached patch features with pose-dependent features using a shared MLP.
4. Use lightweight attention or pooling to share information across nearby box patches.

### 12.4 Contact-set decoder

Use \(K=4\) contact slots for box-plane contact. Each slot predicts:

- Active/near-contact logit.
- Box-local contact point.
- Ground-local contact point.
- Normal.
- Signed distance.
- Learned local latent.
- Confidence.

The four slots cover face contact at four corners while naturally masking unused slots for edge or corner contact. Treat slots as unordered and match them to Chrono contacts with a Hungarian/set loss.

### 12.5 Contact-to-NeDM adapter

For every predicted contact, compute from the box state:

- Relative contact-point velocity.
- Normal relative velocity.
- Tangential relative velocity.
- Lever arm from box center of mass.
- Fixed material parameters and timestep.

Pool or attend over the \(K\) tokens, then append the result to the existing NeDM token at the corresponding timestep.

## 13. Training losses

Use a weighted multi-task loss:

\[
\mathcal L_{\mathrm{contact}} =
\lambda_a\mathcal L_{\mathrm{active}}
+\lambda_d\mathcal L_{\mathrm{distance}}
+\lambda_p\mathcal L_{\mathrm{points}}
+\lambda_n\mathcal L_{\mathrm{normal}}
+\lambda_K\mathcal L_{\mathrm{cardinality}}
+\lambda_u\mathcal L_{\mathrm{uncertainty}}.
\]

Recommended components:

- Focal loss for active/near-contact classification.
- Huber loss for signed distance.
- Hungarian-matched L1/Huber loss for contact points.
- Cosine loss for normals.
- Cross-entropy for contact mode/cardinality.
- Calibration loss or ensemble variance for uncertainty.

After modular training, add NeDM losses:

\[
\mathcal L =
\mathcal L_{\mathrm{contact}}
+\lambda_1\mathcal L_{\mathrm{state,1-step}}
+\lambda_R\mathcal L_{\mathrm{state,rollout}}.
\]

Retain the explicit contact loss during joint fine-tuning so that the latent cannot silently stop representing contact geometry.

## 14. Training sequence

### Experiment A: state-only NeDM

Input:

- Pose/twist history.
- No contact information.

Purpose: establish the existing recurrent baseline.

### Experiment B: oracle-contact NeDM

Input:

- Same state history.
- Canonical contact information from Chrono.

Purpose: establish the maximum benefit of the proposed contact interface.

### Experiment C: contact encoder only

Train geometry/pose to contact information using the balanced contact-query view.

Purpose: select the geometry representation and measure contact accuracy independently of dynamics.

### Experiment D: learned contacts + frozen NeDM

Replace oracle contacts with predicted contacts while keeping NeDM fixed.

Purpose: attribute rollout degradation to the contact encoder.

### Experiment E: joint rollout fine-tuning

Fine-tune the contact encoder and NeDM using scheduled multi-step rollouts.

Purpose: improve event timing and policy-relevant recurrent behavior without losing the explicit contact contract.

## 15. Dataset splits

Split by whole episode and geometry group, never by random frames.

### In-distribution split

- 70% training episodes.
- 15% validation episodes.
- 15% test episodes.
- Disjoint seeds and trajectories.

### Geometry holdout

Reserve complete box-dimension/aspect-ratio groups for test. Do not let nearly identical dimensions appear in training and test merely under different seeds.

### Height/impact-speed holdout

Train on the central height interval and reserve low and high release bands for evaluation. This primarily tests NeDM rather than the static contact encoder.

### Orientation holdout

Reserve stratified regions of \(SO(3)\) or held-out initial contact-mode bins. Report both random-orientation and structured holdout results.

## 16. Evaluation metrics

### Contact encoder

- Contact and near-contact precision/recall.
- False-negative rate near first impact.
- First-impact time error in timesteps.
- Signed-distance MAE and percentile errors.
- Contact-point matching error, normalized by the smallest box dimension.
- Normal angular error.
- Contact cardinality and mode accuracy.
- Confidence calibration and error-versus-uncertainty curve.

### NeDM dynamics

- One-step \(v\) and \(\omega\) error.
- Change in velocity across first impact.
- Rebound apex/time error.
- Position and orientation rollout error.
- Settling time and final-orientation error.
- Maximum penetration and artificial-energy growth.
- Error broken down by no contact, near contact, first impact, rebound, repeated impact, and rest.

### Compute

- One-time geometry preprocessing cost.
- Cached contact-encoder latency.
- Contact queries per second over increasing batch size.
- Complete contact-encoder + NeDM step time.
- GPU memory versus batch size.

Compare contact runtime with Chrono and the analytic box-plane calculation, but do not use this simple case to make the final scalability claim.

## 17. Provisional pilot gates

These are initial engineering gates and should be revised after the 2,000-episode pilot establishes noise floors.

| Gate | Provisional criterion |
| --- | --- |
| Data replay | State and contact sequence reproduced within documented deterministic tolerance |
| Normal/sign convention | All hand-built above/touching/penetrating unit tests pass |
| Oracle benefit | Oracle-contact NeDM clearly improves impact/post-impact error over state-only NeDM |
| Contact timing | Median first-impact error ≤ 1 timestep; 99th percentile ≤ 2 timesteps |
| Near-contact recall | ≥ 99.5% within the configured proximity margin |
| Contact point | Median matched error ≤ 1% of smallest box dimension |
| Normal | Median angular error ≤ 2 degrees |
| Closed-loop degradation | Learned-contact NeDM stays within 10% of oracle-contact NeDM on selected rollout metrics |
| Geometry holdout | No catastrophic failure on held-out box dimensions/aspect ratios |

Failure of a gate should trigger root-cause analysis rather than immediate dataset scaling.

## 18. Required tests

### Geometry/contact unit tests

- Axis-aligned box above the plane: no contact, positive gap.
- Axis-aligned box exactly touching: face contact.
- Axis-aligned controlled penetration: correct sign and normal.
- Rotated corner contact.
- Rotated edge contact.
- Translation in the ground plane: invariant contact result.
- Joint rotation of both box and ground: equivariant result.
- Box mesh remeshing/tessellation change: stable canonical output.

### Dataset tests

- No NaN/Inf values.
- Unit quaternions and consistent order.
- Monotonic timestamps.
- Valid body IDs and geometry hashes.
- Contact points transform consistently among world, box, and ground frames.
- Raw and canonical contact counts remain traceable.
- No episode leakage across splits.

### Model tests

- Permuting surface-token order does not change the result.
- Global translation does not change local contact predictions.
- Confidence increases near OOD geometry or corrupted input.
- Fixed input produces deterministic inference in evaluation mode.

## 19. Deliverables

1. Chrono box-drop generator with deterministic configuration files.
2. Versioned raw/canonical dataset schema.
3. Dataset validation and replay tests.
4. Analytic box-plane baseline.
5. Point-set, neural-SDF, and surface-patch encoder baselines.
6. State-only and oracle-contact NeDM results.
7. Learned-contact and jointly fine-tuned NeDM results.
8. Contact, rollout, and runtime benchmark report.
9. Failure-case visualizer showing predicted versus Chrono contact points and normals.

## 20. Immediate sprint

1. Freeze the state/contact/time-alignment schema.
2. Implement 50 deterministic Chrono smoke episodes.
3. Validate face, edge, corner, free-flight, and resting-contact cases.
4. Train state-only and oracle-contact NeDM baselines.
5. Generate the 2,000-episode pilot with randomized height, orientation, and box geometry.
6. Implement analytic-primitive and surface-point baselines.
7. Implement the cached surface-patch encoder and \(K=4\) set decoder.
8. Run the geometry-held-out bake-off.
9. Connect the selected contact encoder to frozen NeDM.
10. Jointly fine-tune only after the modular attribution results are recorded.

The study advances to manipulation only after the learned-contact recurrent model approaches the oracle-contact upper bound on held-out box geometries.
