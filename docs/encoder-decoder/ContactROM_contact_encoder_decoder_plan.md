# ContactROM Contact Encoder–Impulse Decoder Plan

**Status:** proposed implementation and experiment plan  
**Primary study:** redo the existing Chrono box-drop experiment with a physics-grounded contact bottleneck  
**Baseline:** the current contact-geometry encoder, 64-D pooled contact context, and direct NeDM/NRD transition model  
**Primary proposal:** contact-geometry encoder, contact-impulse decoder, 6-D net contact wrench, and NRD state transition

## 1. Decision summary

ReCoM should test a modular contact **encoder–decoder** architecture:

```text
body geometry and pose
    -> contact geometry encoder
    -> K contact slots and unpooled slot embeddings
    -> contact impulse decoder conditioned on state and physical parameters
    -> per-contact impulses
    -> deterministic aggregation into net linear and angular impulse (J, L)
    -> NRD conditioned on state and (J, L)
    -> next state
```

The two learned modules have deliberately different responsibilities:

1. The **contact geometry encoder** answers a kinematic question:

   > Where could contact occur now, what are the contact normals, and what are the signed distances?

2. The **contact impulse decoder** answers a dynamics question:

   > Given the contact manifold, relative motion, bodies, materials, and timestep, what impulse acts at each contact?

This separation supports the intended transfer strategy:

- Train the encoder across many geometric shapes and contact configurations.
- Condition or fine-tune the decoder for particular entity classes, materials, and dynamics systems.
- Keep the interface between them explicit and testable.

The primary proposed model will pass the **6-D net contact wrench** `(J, L)` to NRD and let NRD predict the full velocity transition. A Newton–Euler update followed by an NRD residual is retained as a physics-strong ablation, not imposed as the only architecture.

## 2. Why the contact bottleneck should be `(J, L)`, not only `J`

For contact slot `i`, let:

- `j_i` be the predicted 3-D impulse in world coordinates.
- `r_i = p_i - x_COM` be the lever arm from the body's center of mass.

Aggregate the contact manifold using known rigid-body algebra:

```text
J = sum_i j_i
L = sum_i r_i x j_i
```

Here:

- `J` is the net linear contact impulse, with units N·s.
- `L` is the net angular contact impulse about the center of mass, with units N·m·s.

This aggregation is called **deterministic** or **analytic** in this plan. It means ordinary sums and cross products with no collision solver and no additional neural network.

A 3-D `J` alone is not sufficient for rigid-body motion. The same upward impulse applied at the center of a box and at a corner produces the same linear velocity change but different angular velocity changes. The smallest body-level contact representation that retains both effects is therefore:

```text
contact_wrench = concat(J, L)  # 6-D
```

The 3-D `J`-only model remains a required ablation because it directly tests whether angular impulse is necessary in the box-drop study.

## 3. What the current ReCoM design does

### 3.1 Current contact encoder

The current point and patch encoders take:

```text
half_extents, position, quaternion
```

For the existing infinite horizontal plane, the learned prediction effectively depends on box geometry, height, and orientation. It is invariant to horizontal translation.

The encoder returns up to `K=4` unordered contact slots. Each slot contains:

- Activation logit.
- Box-local contact point.
- Ground-relative contact point.
- Contact normal.
- Signed distance.
- A 16-D learned latent.
- Point-prediction uncertainty.

It also returns auxiliary contact-cardinality logits for 0–4 contacts.

### 3.2 Current contact-to-NRD path

At runtime, ReCoM keeps the active mask, signed distance, normal, box-local point, and optional latent. The adapter derives:

- World-space lever arm.
- Contact-point relative velocity.
- Normal velocity.
- Tangential velocity and speed.

These form 19 explicit features per contact, optionally plus the learned latent. `ContactPooling` applies learned per-slot processing followed by sum, max, and count pooling to produce a 64-D context. NRD then receives:

```text
13-D body state + 3-D half-extents + 64-D contact context
```

and directly predicts:

```text
delta linear velocity (3) + delta angular velocity (3)
```

### 3.3 Physics already present

The current implementation is not completely unstructured. It includes:

- A fixed gravity prior.
- A contact-activation gate that makes free flight exact.
- Explicit contact lever arms and relative velocities.
- Exact Chrono-style semi-implicit position and quaternion integration.

### 3.4 Missing physics bottleneck

The current implementation does **not** explicitly predict:

- Per-contact impulses.
- Net linear impulse.
- Net angular impulse.
- A contact wrench.
- Momentum consistency.
- Equal-and-opposite pair impulses.
- Friction-cone compliance.

The generic 64-D context is free to encode any information, and NRD must implicitly learn both contact response and rigid-body momentum transfer.

The dataset records Chrono reaction forces, but the present contact loss, adapter, and transition model do not use them.

### 3.5 Important latent-space limitation

The existing 16-D output named `latent` is not directly supervised by the standalone contact-set loss. The explicit contact heads are supervised, but the final latent projection receives dynamics-related training only when transition gradients are allowed to reach the encoder.

Consequently, the current latent must not be assumed to represent force or impulse. The proposed decoder should consume an explicitly returned **unpooled slot embedding** from the set decoder, together with the supervised geometric quantities.

## 4. Proposed architecture

### 4.1 End-to-end data flow

```text
candidate body pair
    -> geometry and relative pose
    -> geometry encoder
         outputs: active_i, p_i, n_i, d_i, slot_embedding_i
    -> per-slot dynamics feature construction
         relative contact velocity, lever arms, inverse mass/inertia,
         friction, restitution, timestep, body flags
    -> impulse decoder with cross-slot interaction
         outputs: j_i for every active contact slot
    -> deterministic aggregation
         J = sum_i j_i
         L = sum_i r_i x j_i
    -> NRD input token
         state, body parameters, J, L
    -> predicted delta velocity and delta angular velocity
    -> existing semi-implicit pose integration
    -> repeat from predicted state
```

During recurrent rollout, the geometry encoder and impulse decoder must be reevaluated from the predicted state at every step. Recorded contacts or impulses must never be replayed after the trajectory diverges.

### 4.2 Geometry encoder contract

The encoder remains responsible for kinematic contact information. For every candidate pair and slot `i`, return:

```text
active probability
p_A_local
p_B_local or pair-relative point
normal from B to A
signed distance
slot embedding z_i
uncertainty
```

For the first box-plane experiment, the existing point encoder is the default because it achieved the best contact timing and held-out-geometry result.

The current `SetDecoder` should expose its post-attention query representation before the output head, for example:

```text
slot_embedding: (B, K, D_slot)
```

The existing 16-D `latent` can be retained for compatibility, but the new impulse decoder should not depend exclusively on an unsupervised projection.

### 4.3 Why the pooled 64-D vector cannot be the only decoder input

The existing 64-D contact context is permutation invariant and intentionally compresses the set using sum, max, and count statistics. This is useful as a global description of the manifold, but it loses the direct association between a particular slot, its contact point, and its impulse.

Decoding `K` individual impulses from only this pooled vector makes the per-contact decomposition ambiguous. Therefore:

- Preserve every unpooled slot embedding.
- Compute a global 64-D manifold context using the existing pooling concept.
- Broadcast the global context back to each slot.
- Predict `j_i` from the local slot, global manifold context, and dynamics features.

The 64-D vector remains useful; it is global conditioning rather than the sole carrier of contact information.

### 4.4 Contact impulse decoder inputs

For contact slot `i`, construct:

```text
slot embedding z_i
active probability or hard mask
signed distance d_i
normal n_i
contact anchors / lever arms r_A_i and r_B_i
relative contact velocity v_rel_i
normal relative velocity v_n_i
tangential relative velocity v_t_i
global 64-D contact-manifold context
inverse masses of A and B
world or body-frame inverse inertias of A and B
friction and restitution
timestep dt
dynamic/fixed body flags
```

For Study 1, body B is fixed ground, so its inverse mass and inverse inertia are zero. Although density, friction, restitution, and timestep are fixed in the current dataset, they should still be explicit decoder inputs or recorded conditioning constants. Otherwise, the decoder will silently memorize one dynamics system and will be difficult to transfer.

### 4.5 Decoder structure

Use a small set transformer over the `K` active/candidate contact slots:

1. Concatenate local geometry, motion, physical parameters, and broadcast global context.
2. Apply a shared slot MLP.
3. Apply one or two self-attention blocks across slots so simultaneous contacts can coordinate.
4. Decode impulse components for each slot.

Cross-slot coupling is important because contact impulses are not independent. Face contact, for example, is a coupled manifold rather than four unrelated point impacts.

The first implementation should be substantially smaller than NRD, such as:

```text
slot width: 128
self-attention blocks: 2
heads: 4
output per slot: 3 impulse parameters + optional uncertainty
```

### 4.6 Physically structured impulse output

Construct a deterministic tangent basis `(t1_i, t2_i)` from `n_i`. Decode:

```text
raw normal impulse a_n
raw tangent coordinates a_t1, a_t2
```

Then parameterize:

```text
j_n >= 0
norm(j_t) <= mu * j_n
j_i = j_n * n_i + j_t1 * t1_i + j_t2 * t2_i
```

Recommended implementation:

- Use `softplus(a_n)` for a nonnegative normal impulse.
- Map the two tangential outputs into the friction disk of radius `mu * j_n`.
- Multiply the final impulse by the slot activation mask.
- Produce exactly zero impulse when no contact is active.

An unconstrained 3-D Cartesian impulse head should be retained as an ablation. It may fit Chrono more easily, while the structured head should generalize and remain physically interpretable.

### 4.7 Deterministic wrench aggregation

For the box in Study 1:

```text
J_box = sum_i j_i
L_box = sum_i r_box_i x j_i
```

For a future dynamic body pair `(A, B)`, a single predicted pair impulse is scattered with equal and opposite signs:

```text
J_A +=  j_i
J_B += -j_i
L_A += r_A_i x  j_i
L_B += r_B_i x -j_i
```

This scatter operation provides pairwise linear-momentum conservation by construction.

### 4.8 Primary NRD interface

The primary experiment replaces the current 64-D contact-context input to NRD with:

```text
J_world (3)
L_world (3)
```

NRD receives:

```text
state history
body geometry and dynamics parameters
6-D predicted contact wrench history
known external inputs / controls
```

and predicts the full:

```text
delta v, delta omega
```

The known gravity prior and exact pose integrator remain unchanged. Raw contact points, normals, and distances do not go directly to NRD in the primary model; their effect must pass through the impulse decoder.

### 4.9 Physics-strong residual ablation

Also implement or evaluate a stronger inductive-bias variant:

```text
state_phys_next = NewtonEuler(state, J, L, mass, inertia, gravity, dt)
state_next = state_phys_next + NRD_residual(...)
```

This tests whether NRD benefits from receiving the wrench while retaining freedom to predict the whole transition, or whether explicitly applying momentum transfer produces better stability and transfer.

The residual model is an ablation, not a prerequisite for accepting the encoder–decoder proposal.

## 5. Supervision targets

### 5.1 Contact geometry targets

Retain the current Hungarian-matched contact-set loss:

- Activation.
- Signed distance.
- Box and ground contact points.
- Normal.
- Cardinality.
- Point uncertainty.

This preserves the encoder's geometric interpretation and prevents end-to-end dynamics training from destroying contact accuracy.

### 5.2 State-derived net linear impulse

For the fixed-ground box-drop data, derive the net contact impulse from consecutive recorded states:

```text
J_state[k] = m * (v[k+1] - v[k] - g_vec * dt)
```

Here `g_vec = (0, 0, -g)`. This removes the known gravitational impulse and leaves the total impulse attributable to contact for the step.

This label is preferable to reported force as the initial source of truth because it is guaranteed to be consistent with the recorded transition.

### 5.3 State-derived net angular impulse

Compute world angular momentum about the center of mass:

```text
I_world[k] = R[k] * I_body * R[k]^T
H[k] = I_world[k] * omega_world[k]
L_state[k] = H[k+1] - H[k]
```

Gravity has no torque about the box center of mass. Any Chrono integrator convention or numerical discrepancy must be quantified during label validation before training.

### 5.4 Chrono per-contact force audit

The dataset records `c_force_world` for every canonical contact. Before using it as a per-contact impulse label, validate:

```text
j_force_i = c_force_world_i * dt
J_force = sum_i j_force_i
L_force = sum_i r_i x j_force_i
```

Compare `(J_force, L_force)` with the state-derived `(J_state, L_state)` by regime:

- First impact.
- Continued impact/contact.
- Rebound contact.
- Resting/rocking.

If the agreement is sufficiently accurate and nonzero for Chrono NSC, use `j_force_i` for per-contact supervision. If it is not, supervise only the aggregated wrench and treat the individual `j_i` decomposition as latent. In that case, do not claim that each predicted slot impulse matches a unique ground-truth solver impulse.

### 5.5 Physics-consistency losses

For the primary NRD model, add consistency between its predicted state transition and the decoder wrench:

```text
J_from_predicted_state = m * (delta_v_pred - g_vec * dt)
L_from_predicted_state = H_predicted_next - H_current
```

Penalize:

```text
loss_momentum = error(J_from_predicted_state, J_pred)
              + error(L_from_predicted_state, L_pred)
```

This permits NRD to predict the full transition while discouraging it from ignoring the physically meaningful bottleneck.

### 5.6 Total loss

Use a staged weighted objective:

```text
loss = lambda_contact  * loss_contact_geometry
     + lambda_impulse  * loss_net_wrench
     + lambda_per_slot * loss_per_contact_impulse   # only if labels pass audit
     + lambda_state    * loss_one_step_transition
     + lambda_rollout  * loss_multistep_rollout
     + lambda_momentum * loss_momentum_consistency
     + lambda_physics  * loss_constraint_violations
```

Normalize impulse and angular-impulse targets separately using robust training-split statistics. Impact distributions are heavy-tailed, so Huber loss is preferred initially.

## 6. Data and interface changes

### 6.1 No immediate dataset regeneration

The existing `pilot1b` episodes already contain:

- Consecutive position, orientation, linear velocity, and angular velocity.
- Box mass and body-frame inertia in metadata.
- Timestep and gravity.
- Canonical contact points and normals.
- Chrono reaction forces.

Therefore, the initial impulse-target audit and box-drop experiment should derive labels from the existing data without rewriting episodes.

### 6.2 Derived training fields

Add on-the-fly or cached derived fields:

```text
target_J_contact       (3)
target_L_contact       (3)
target_wrench          (6)
optional target_j_slot (K, 3)
mass                   (1)
inertia_body           (3, 3) or diagonal (3)
friction               (1)
restitution            (1)
dt                     (1)
```

Keep derived-label formula versions in run metadata. Do not create a new dataset schema until an experiment needs persistent multi-body pair tables.

### 6.3 Time alignment

Preserve the frozen convention:

```text
(state[k], contact_geometry[k], solved_contact_response[k]) -> state[k+1]
```

The impulse decoder may use only information available at `state[k]`: current geometry, pose, velocities, physical parameters, and history. It must not see `state[k+1]` except through the training loss.

## 7. Proposed ReCoM code structure

Add:

```text
recom/models/contact_impulse_decoder.py
recom/physics/contact_wrench.py
recom/data/impulse_targets.py
recom/eval/impulse_metrics.py
scripts/audit_impulse_labels.py
scripts/train_impulse_decoder.py
configs/encoder_decoder_boxdrop.yaml
```

Modify carefully:

```text
recom/models/contact_encoder.py
    expose unpooled post-attention slot embeddings

recom/models/nedm_adapter.py
    retain current pooling for the baseline
    add global manifold context for decoder conditioning

recom/models/transition.py
    add contact_input_mode = pooled_context | wrench | physics_residual

recom/data/dataset.py
    derive impulse targets and physical parameter tensors

recom/train/train_transition.py
    support decoder pretraining, frozen stages, and joint training

recom/train/rollout.py
    recompute geometry and impulse from every predicted state
```

Do not delete or silently alter the existing explicit-contact path. It is the frozen baseline required for a valid comparison.

## 8. Training sequence

### Stage ED0: impulse-label audit

1. Derive `(J_state, L_state)` from all training transitions.
2. Derive force-integrated per-contact and net impulses from Chrono records.
3. Compare the two label sources by contact regime and box geometry.
4. Inspect signs, units, outliers, inactive-frame residuals, and angular-momentum conventions.
5. Freeze the accepted target formulas and normalization statistics.

**Exit criterion:** inactive frames have approximately zero contact wrench, impact labels have physically plausible signs and scales, and the per-contact force labels are either validated or explicitly rejected.

### Stage ED1: expose and freeze geometry features

1. Load the selected 30k point contact encoder.
2. Expose post-attention slot embeddings.
3. Confirm that adding the returned embeddings does not change existing explicit outputs or metrics.
4. Freeze the encoder initially.

**Exit criterion:** contact precision, recall, timing, point, normal, and distance metrics reproduce the existing checkpoint.

### Stage ED2: decoder pretraining

Train the impulse decoder on teacher-forced states with the frozen learned geometry encoder in the loop. This matches inference-time geometry noise better than training only on perfect Chrono contact slots.

Required evaluations:

- Net `J` and `L` error.
- Impact direction and magnitude error.
- Friction and nonnegative-normal violations.
- Error by contact regime and geometry split.
- Optional per-contact impulse error if labels passed ED0.

**Exit criterion:** predicted net wrench substantially improves over a zero-impulse baseline and generalizes to held-out box sizes.

### Stage ED3: wrench-conditioned NRD

Train a transition model with the same transformer size, context length, event sampling, data splits, and optimization budget as the current baseline. Its only contact input is the predicted `(J, L)`.

Train in two steps:

1. Freeze geometry encoder and impulse decoder; train NRD.
2. Optionally fine-tune decoder and NRD jointly while retaining geometry and wrench supervision.

The primary model predicts full `(delta_v, delta_omega)` with momentum-consistency loss.

### Stage ED4: closed-loop fine-tuning

If one-step results pass but rollouts drift, add the existing short unrolled loss. At every unrolled step:

1. Integrate the predicted state.
2. Recompute contact geometry from that predicted state.
3. Recompute contact impulses.
4. Feed the new wrench to NRD.

Do not replay recorded contacts or impulses.

### Stage ED5: physics-strong residual

Train the Newton–Euler-plus-NRD-residual variant using the same decoder checkpoint and data budget. This isolates the value of applying the wrench explicitly instead of only conditioning NRD on it.

## 9. Box-drop comparison study

### 9.1 Controlled model variants

| ID | Contact representation passed to dynamics | Transition rule | Purpose |
| --- | --- | --- | --- |
| `BASE-64` | Current learned 64-D pooled contact context | NRD predicts full delta | Frozen existing baseline |
| `J-3` | Predicted net linear impulse `J` | NRD predicts full delta | Test the proposed 3-D reduction and expose missing torque |
| `JL-6` | Predicted net wrench `(J,L)` | NRD predicts full delta | Primary encoder–decoder model |
| `JL-6-C` | Predicted `(J,L)` | Full-delta NRD + momentum consistency | Test physics regularization |
| `JL-6-R` | Predicted `(J,L)` | Newton–Euler base + NRD residual | Strongest physics inductive bias |
| `POOL-DEC` | Only pooled 64-D context decoded back to impulses | NRD predicts full delta | Test whether lost slot identity matters |
| `ORACLE-JL` | State-derived ground-truth `(J,L)` in teacher-forced evaluation | NRD predicts full delta | One-step representation ceiling only |

`ORACLE-JL` must not be reported as a normal closed-loop result because its target uses the future recorded state. A closed-loop oracle requires a live Chrono or validated analytic impulse solver.

### 9.2 Fairness controls

Use identical:

- `pilot1b` train/validation/test/test-geometry splits.
- Priming history and rollout horizons.
- NRD transformer width, depth, and context length.
- Gravity prior and pose integrator.
- Event oversampling.
- Training-step budget and number of random seeds.
- Learned point-encoder checkpoint unless a variant explicitly fine-tunes it.

Match the impulse decoder parameter count when comparing constrained and unconstrained heads. Report total parameters and runtime separately; do not hide decoder cost inside NRD.

### 9.3 Contact-geometry metrics

Confirm that the new training path does not degrade:

- Frame and slot precision/recall.
- First-impact timing.
- Signed-distance error.
- Point and normal error.
- Held-out-geometry performance.

### 9.4 Impulse metrics

Report:

- Net linear impulse MAE and relative error.
- Net angular impulse MAE and relative error.
- Impulse direction error.
- Normal and tangential impulse error.
- Momentum-consistency residual.
- Friction-cone violation rate.
- Non-contact impulse false-positive rate.
- Optional per-slot impulse metrics.

Report impact and resting frames separately. A global average is dominated by zero-impulse free flight and is not meaningful.

### 9.5 One-step dynamics metrics

Retain the current metrics:

- Linear velocity-delta error.
- Angular velocity-delta error.
- First-impact and repeated-impact errors.
- Free-flight exactness.

Add error conditional on net impulse magnitude and contact cardinality.

### 9.6 Closed-loop metrics

Report paired per-episode comparisons for:

- Position error at 100, 500, 1000 steps and final time.
- Linear and angular velocity error.
- Impact `delta_v` and `delta_omega` error.
- Maximum penetration.
- Artificial energy gain.
- Settling time.
- Final speed and angular speed.
- Final center-of-mass height.
- Settled-face classification.
- Symmetry-aware orientation error.

The ordinary final orientation error should remain in the table, but it must not be the sole stability conclusion. Box impacts can branch into different physically plausible resting faces.

### 9.7 Runtime metrics

Measure:

- Geometry encoder time.
- Impulse decoder time.
- Wrench aggregation time.
- NRD time.
- Total step time.
- Peak GPU memory.

## 10. Existing baseline reference

The completed Study-1 point-encoder/frozen-NRD result currently reports approximately:

| Split | Impact `delta_v` error | Position error at 500 steps | Final position error | Maximum penetration |
| --- | ---: | ---: | ---: | ---: |
| Test | 0.43 m/s | 0.050 m | 0.17 m | 0.014 m |
| Held-out geometry | 0.44 m/s | 0.045 m | 0.11 m | 0.017 m |

The exact checkpoint and evaluation settings must be frozen before new experiments. These values are references from the current progress report, not substitutes for a paired rerun.

## 11. Success criteria

The new architecture is considered useful only if `JL-6` or `JL-6-C` demonstrates a repeatable benefit over `BASE-64` rather than merely introducing an interpretable intermediate variable.

Primary gates:

1. Geometry-contact metrics remain within the existing acceptance tolerance.
2. The decoder predicts nonzero impact wrench and near-zero free-flight wrench with no sign or timing failure.
3. `JL-6` improves both impact angular-velocity error and at least one long-horizon stability metric on test and held-out geometry.
4. Position, penetration, and artificial-energy metrics do not regress materially.
5. Improvements hold over at least three seeds or a paired episode-level confidence interval.
6. `JL-6` outperforms `J-3` on rotational metrics, validating the angular-impulse channel.
7. NRD uses the wrench: shuffling or zeroing `(J,L)` must materially degrade impact prediction.

A provisional quantitative target is a 15% or larger reduction in median impact angular-velocity error or position error at 500 steps, with no greater than 5% regression in median impact linear-velocity error. Final gates should be fixed after ED0 establishes label quality and baseline variance, not tuned after seeing test results.

## 12. Essential ablations

Run these before making architectural claims:

1. `J` versus `(J,L)`.
2. Pooled-only decoder versus unpooled slot decoder with global context.
3. Unconstrained Cartesian impulse versus normal/tangent friction-constrained impulse.
4. Frozen geometry encoder versus joint fine-tuning.
5. NRD full transition versus Newton–Euler plus NRD residual.
6. With versus without momentum-consistency loss.
7. Predicted geometry versus Chrono geometry during teacher-forced evaluation.
8. State-derived wrench targets versus force-derived targets if the force audit passes.
9. Decoder conditioned on physical parameters versus implicit fixed-system constants.

## 13. Risks and mitigations

| Risk | Consequence | Mitigation |
| --- | --- | --- |
| Decode per-contact impulses from pooled context alone | Slot identity and torque allocation are ambiguous | Preserve unpooled slot embeddings; use pooling only as global context |
| Chrono force records do not match state transitions | Incorrect impulse supervision | Audit against momentum differences; prefer state-derived net wrench |
| Net supervision cannot identify individual slot impulses | Plausible total wrench but uninterpretable slot decomposition | Use per-contact labels only if validated; otherwise report net metrics only |
| Decoder memorizes one box density or material | Poor transfer to new entities | Condition explicitly on mass, inertia, friction, restitution, and timestep |
| NRD ignores `(J,L)` and reconstructs contact response from state | Physics bottleneck provides no benefit | Momentum loss, wrench shuffling test, restrict raw contact input to NRD |
| Hard activation miss produces zero impulse | Missed collision destabilizes rollout | Retain geometry loss, conservative activation calibration, future broad-phase margin |
| Impulse error compounds recurrently | Long-horizon drift remains | Train with predicted-state unrolls and recompute contacts/impulses every step |
| Angular-momentum target uses inconsistent frames | Incorrect `L` labels | Unit tests in world and body frames; verify against state transitions and Chrono forces |
| Physics constraints hurt Chrono imitation | Worse benchmark despite better interpretation | Keep unconstrained head as an ablation and compare stability/generalization |
| Final orientation error reflects different valid landing faces | Misleading drift conclusion | Add settled-face and symmetry-aware metrics |

## 14. Relationship to broad phase

The encoder–decoder study and broad-phase study solve different problems:

- Broad phase selects candidate body pairs efficiently.
- The geometry encoder predicts a contact manifold for a candidate pair.
- The impulse decoder predicts the contact response.
- NRD predicts the state transition from the resulting body wrench.

The proposed final pipeline is:

```text
scene state
    -> broad-phase candidate pairs
    -> geometry encoder per pair
    -> impulse decoder per contact manifold
    -> pairwise impulse scatter and per-body wrench aggregation
    -> NRD state transition
```

The encoder–decoder box-drop experiment should precede the expensive multi-body broad-phase study because it directly targets the current oracle-contact dynamics drift. The broad-phase interface should nevertheless preserve body and geometry pair identities required by the future impulse scatter.

## 15. Recommended execution order

1. Freeze the current point-encoder and `BASE-64` checkpoints and paired rollout rows.
2. Implement and validate state-derived `(J,L)` targets.
3. Audit Chrono per-contact force labels against the state-derived wrench.
4. Expose unpooled slot embeddings without changing current encoder predictions.
5. Implement deterministic tangent bases, constrained impulse parameterization, and wrench aggregation.
6. Pretrain the impulse decoder with the geometry encoder frozen.
7. Train `J-3`, `JL-6`, and `JL-6-C` with matched NRD settings.
8. Evaluate one-step, closed-loop, geometry-holdout, momentum, energy, and penetration metrics.
9. Run the pooled-only and physics-residual ablations.
10. Jointly fine-tune only if the modular frozen experiment identifies a decoder bottleneck.
11. Select the architecture using validation and freeze it before final test reporting.
12. Carry the selected encoder–decoder interface into the multi-pair broad-phase study.

## 16. Definition of done

The encoder–decoder study is complete when:

- ReCoM exposes a stable geometry-encoder slot contract.
- A contact impulse decoder predicts per-slot impulses from current information only.
- Deterministic aggregation produces a 6-D per-body contact wrench.
- NRD can consume `(J,L)` without receiving raw contact geometry in the primary variant.
- State-derived impulse targets and Chrono force labels have a documented validation result.
- Current and proposed architectures are evaluated on identical splits and paired episodes.
- The `J`-only, `(J,L)`, pooled-only, and physics-residual ablations are reported.
- Geometry accuracy, impulse accuracy, one-step dynamics, long rollouts, penetration, energy, and runtime are all reported.
- The result establishes whether the physics-grounded bottleneck improves the existing box-drop baseline.

## 17. Intended research conclusion

The experiment should test a stronger claim than contact-geometry imitation:

> A shape-general contact encoder can represent the kinematic contact manifold, while a dynamics-conditioned contact decoder converts that manifold into physically meaningful per-contact impulses. Aggregating those impulses into a net contact wrench gives NRD a compact and interpretable interaction signal that may improve rigid-body transition accuracy and recurrent stability.

A positive result would justify transferring the geometry encoder across shapes while adapting the impulse decoder to new entities and dynamics systems. A negative result would still be informative: it would show whether the existing high-dimensional pooled context contains transition-relevant information that a 6-D wrench bottleneck discards, or whether the limitation lies mainly in the NRD rollout model rather than the contact representation.
