# ContactROM: High-Level Plan for a Neural Contact-Information Encoder

Last updated: 2026-08-25 (America/Chicago)

## 1. Purpose

ContactROM extends the existing NeDM/Neural Reduced Dynamics (NRD) workflow with a separate learned contact-information module.

The intended decomposition mirrors a conventional multibody simulator:

| Conventional simulator | ContactROM |
| --- | --- |
| Scene geometry and body poses | Scene geometry and body poses |
| Collision detection and contact generation | Neural contact-information encoder |
| Contact manifold | Explicit contact tokens plus learned local contact features |
| Multibody/contact solver and time integration | Recurrent NeDM/NRD transition model |
| Next body state | Next reduced dynamic state |

The project should **not** begin with one monolithic network that maps an entire scene directly to the next state. The central research contribution is the interface

\[
\text{scene geometry and poses}
\xrightarrow{E_{\mathrm{contact}}}
\text{contact information}
\xrightarrow{F_{\mathrm{NRD}}}
\text{next reduced state}.
\]

The existing NeDM pipeline is assumed to be capable of learning the recurrent state transition once it receives a Markov-sufficient state, action, and useful contact context. This plan therefore focuses on designing, training, and validating the contact-information encoder.

## 2. Central research question

> Can a learned, geometry-conditioned contact encoder replace the conventional collision-detection/contact-generation interface and provide sufficiently accurate contact information for stable, policy-valid recurrent neural simulation?

The first claim is **not** that the learned encoder will immediately beat every GPU collision detector. The ordered goals are:

1. Learn a geometry representation that transfers across poses and object instances.
2. Predict contact events, locations, normals, and separation accurately near the contact boundary.
3. Demonstrate that the predicted contact information supports stable NeDM rollouts.
4. Demonstrate policy training in the resulting neural simulator and transfer the unchanged policy back to Chrono.
5. Only then establish where the learned contact pipeline has a throughput or scaling advantage.

## 3. Proposed recurrent simulation loop

Let \(z_t^1\) denote the reduced dynamic state propagated by NeDM. For rigid-body manipulation it must contain, at minimum, all dynamic poses and twists needed to reconstruct the next scene:

\[
z_t^1 =
\left[
q_t,\dot q_t,
\{x_t^i,R_t^i,v_t^i,\omega_t^i\}_{i\in\text{dynamic bodies}}
\right].
\]

Static object geometry is cached and does not need to be propagated in \(z_t^1\).

At every step:

1. Decode or integrate the poses contained in \(z_t^1\).
2. Evaluate the contact encoder:

   \[
   c_t = E_{\mathrm{contact}}
   \left(\{\mathcal G_i,T_i(z_t^1)\}_{i=1}^{N}\right).
   \]

3. Combine the contact information with state-derived quantities such as relative contact-point velocity and material parameters.
4. Predict the next reduced state:

   \[
   z_{t+1}^1 = F_{\mathrm{NRD}}
   \left(z_{t-h+1:t}^1,u_{t-h+1:t},c_{t-h+1:t}\right).
   \]

5. Reconstruct the next scene from \(z_{t+1}^1\), then recompute \(c_{t+1}\).

The contact representation is therefore **computed context**, not a free-running latent that NeDM must propagate indefinitely. A short contact history can still be included to represent impact, sticking, sliding, and hysteresis.

## 4. Contact encoder output contract

The encoder should produce a variable-size set of contact tokens. Each token should contain an explicit, auditable geometric component and an optional learned local feature:

\[
c_t^k =
\left[
i,j,
p_A^{\mathrm{local}},p_B^{\mathrm{local}},
n^{\mathrm{pair}},
d,
a,
\ell_k,
\sigma_k
\right].
\]

Where:

- \(i,j\): body or link identifiers.
- \(p_A,p_B\): paired surface/contact locations.
- \(n\): consistently oriented contact normal.
- \(d\): signed distance or penetration measure.
- \(a\): contact or near-contact activation probability.
- \(\ell_k\): learned local contact feature describing patch geometry.
- \(\sigma_k\): calibrated confidence or uncertainty.

The encoder should report contacts within a small positive proximity margin, not only currently penetrating contacts. This gives NeDM advance information about an impact that may occur during the next integration interval.

A deterministic feature builder can append quantities that should not be learned by the geometry encoder, for example

\[
v_{\mathrm{rel}}^k =
\left(v_i+\omega_i\times r_i\right)
-
\left(v_j+\omega_j\times r_j\right),
\]

along with friction, restitution, compliance, and timestep. The resulting contact token is supplied to NeDM.

## 5. Geometry representation research decision

### 5.1 Requirements

A useful representation must:

- Support rigid objects with different shapes and mesh resolutions.
- Preserve face interiors, sharp edges, corners, normals, and local scale.
- Be invariant to vertex ordering and robust to remeshing.
- Separate canonical geometry from runtime pose.
- Permit cached per-object computation.
- Support sparse local interaction instead of all-to-all scene attention.
- Extend later to partial point clouds or reconstructed surfaces.
- Be efficient enough for thousands of batched neural environments.

### 5.2 Candidate representations

| Representation | Strength | Main limitation | Role in this project |
| --- | --- | --- | --- |
| Analytic primitive parameters | Exact and extremely cheap for boxes, spheres, and capsules | Does not extend to arbitrary geometry | Required box baseline |
| Dense voxel/SDF grid | Regular GPU memory and direct distance queries | Cubic memory and resolution-dependent contact error | Optional baseline |
| Global neural SDF latent | Continuous distance queries and compact shape code | Repeated MLP queries can be expensive; one global code can hide local contact structure | Important ablation |
| Surface point cloud | Simple, permutation-invariant, compatible with scanned geometry | May miss sharp features or face interiors unless sampling is carefully designed | Strong baseline |
| Triangle/face graph | Preserves topology, face interiors, sharp features, and normals | Variable graph size; requires pruning and mesh-quality controls | Recommended primary representation |

DeepSDF demonstrates that an entire shape class can be represented by a latent-conditioned continuous signed-distance field, but its runtime value must be tested rather than assumed for high-frequency contact queries ([DeepSDF](https://arxiv.org/abs/1901.05103)). PointNet and PointNet++ establish efficient permutation-invariant and hierarchical local processing of point sets ([PointNet](https://arxiv.org/abs/1612.00593), [PointNet++](https://arxiv.org/abs/1706.02413)).

Contact-specific evidence favors retaining local surface structure. LOCC reports that learning local colliding shape crops is easier and more GPU-friendly than learning entire global shapes ([LOCC](https://arxiv.org/abs/2304.09439)). FIGNet shows that node-only collision representations can miss collisions on face interiors, while face-to-face interactions improve accuracy and efficiency on sparse rigid meshes ([FIGNet](https://arxiv.org/abs/2212.03574)). Neural Collision Fields further show that triangle-pair neural fields can generalize to new meshes without retraining ([Neural Collision Fields](https://research.nvidia.com/labs/prl/publication/zesch2023ncf/)).

### 5.3 Recommended unified representation

Use a **two-level, multiresolution surface-patch representation**:

1. **Object token**
   - Geometry identifier.
   - Canonical scale and bounding dimensions.
   - Coarse global shape feature.
   - Optional material class and collision margin.

2. **Local surface-patch tokens**
   - Triangle vertices relative to the face centroid, or an oriented local point patch.
   - Face centroid in the object frame.
   - Face/patch normal.
   - Area or patch radius.
   - Curvature, sharp-edge, and boundary indicators.
   - Mesh-adjacency neighborhood feature.
   - Learned local latent cached for the asset.

All static geometry features should be computed once in the object frame. At runtime, only pose-dependent transformations and pairwise interaction processing are performed.

This representation is more general than box parameters, retains the contact-relevant information lost by one global latent, and can later accept scanned geometry by converting an oriented point cloud into local surface patches.

The representation remains a hypothesis. It must be compared against a point-set encoder and a neural-SDF query model before it is adopted as the final architecture.

## 6. Proposed neural architecture

### 6.1 Offline asset encoder

For each object:

1. Normalize the mesh into a canonical object frame while preserving physical scale separately.
2. Construct triangle/patch tokens and adjacency.
3. Apply shared per-patch MLPs followed by a small mesh GNN or local PointNet++ block.
4. Cache the resulting object and patch latents.

Do not recompute the complete geometry encoder at every physics step for rigid objects.

### 6.2 Runtime candidate-pair construction

For a small proof-of-concept scene, every object pair can be evaluated. For scalable manipulation or locomotion scenes, construct a conservative proximity graph using AABBs, a spatial hash, or a learned coarse occupancy head with conservative fallback.

The learned contribution begins at the local pair-interaction stage; retaining an inexpensive conservative broad phase avoids an \(O(N^2)\) global network.

### 6.3 Pair-interaction encoder

For each candidate body pair:

1. Express both surface-token sets in a common pair-relative frame.
2. Select spatially nearby patch pairs.
3. Use a small cross-attention block or face-interaction message-passing block.
4. Produce local interaction features that are invariant to global translation and structurally equivariant to rotation through the chosen relative frame.

The first implementation should prefer a small MLP/GNN architecture over a large Transformer. Complexity should be added only when the geometry bake-off demonstrates a need.

### 6.4 Contact-set decoder

Decode a fixed maximum of \(K\) unordered contact slots per body pair:

- Contact/no-contact logit.
- Paired local contact locations.
- Normal.
- Signed distance.
- Patch extent or contact-mode feature.
- Learned contact latent.
- Uncertainty.

Use a no-contact mask and Hungarian/set matching during training. A fixed \(K\) keeps batching simple; sparse variable-length tokens can be introduced later.

## 7. Data contract

Every recorded transition should preserve:

- Episode/scenario identifier and random seed.
- Geometry identifiers and the exact collision meshes.
- Object scale, mass, inertia, and material parameters.
- Position, orientation, linear velocity, and angular velocity for every dynamic body.
- Action or actuator command.
- Raw Chrono contact records, including object pair, paired points, contact frame/normal, signed distance, effective radius, reaction force/torque when valid, and solver-specific fields.
- A canonicalized contact-set view for supervised learning.
- State before the step and state after the step.
- Timestep, integrator, collision system, contact method, solver tolerances, Chrono version, and all physics settings.

Chrono's contact-report callback exposes paired points, a contact frame whose X axis is the normal, contact distance, effective radius, reaction force/torque, object identities, and an NSC constraint offset ([Chrono contact callback](https://api.projectchrono.org/classchrono_1_1_ch_contact_container_1_1_report_contact_callback.html)). Raw values must be preserved before any canonicalization.

Do not mix NSC, SMC, Chrono Multicore, or other simulator labels as if they were one exact ground truth. Preserve solver provenance and use solver conditioning or explicit solver-holdout experiments.

## 8. Training program

### Stage A: establish an oracle-contact NRD upper bound

Train/evaluate NeDM with contact information taken directly from Chrono. This establishes whether the chosen contact interface is sufficient for recurrent dynamics.

If oracle contact information does not improve impact and post-contact rollouts over state-only NeDM, redesign the state/contact interface before training a contact encoder.

### Stage B: geometry pretraining

Pretrain the geometry encoder using dense geometric objectives:

- Signed distance or closest-surface distance.
- Surface normal.
- Patch correspondence or relative geometry.
- Near-contact classification under randomized relative poses.

This stage should use balanced free-space, near-contact, and penetrating queries.

### Stage C: supervised contact-set training

Train against canonicalized Chrono contacts using:

- Focal/BCE loss for contact activation.
- Huber/L1 loss for signed distance.
- Matched contact-point loss.
- Cosine normal loss.
- Contact-cardinality/mode loss.
- Calibration loss for uncertainty.

### Stage D: connect pretrained modules

Feed predicted contacts to a frozen NRD model. The difference from the oracle-contact result isolates contact-encoder error.

### Stage E: dynamics-aware joint fine-tuning

Fine-tune the contact encoder and NRD together with multi-step rollout loss while retaining explicit geometric supervision:

\[
\mathcal L =
\lambda_c\mathcal L_{\mathrm{contact}}
+\lambda_1\mathcal L_{\mathrm{one-step}}
+\lambda_R\mathcal L_{\mathrm{rollout}}.
\]

This lets the learned contact latent capture contact details useful to dynamics without allowing it to abandon the interpretable contact contract.

### Stage F: policy-driven dataset aggregation

1. Freeze the neural simulator.
2. Train a manipulation or locomotion policy inside it.
3. Replay policy-visited states and actions in Chrono.
4. Add high-disagreement and high-uncertainty cases to the dataset.
5. Retrain and repeat.

## 9. Required ablations

### Contact source

- State-only NeDM.
- Chrono contacts + NeDM: oracle upper bound.
- Learned contacts + frozen NeDM.
- Jointly fine-tuned contact encoder + NeDM.

### Contact representation

- Explicit contact quantities only.
- Learned contact latent only.
- Explicit quantities plus learned local latent.

### Geometry representation

- Analytic primitive parameters.
- PointNet++ surface-point encoder.
- Recommended surface-patch/face encoder.
- Neural-SDF query encoder.

### Generalization

- Unseen poses of seen objects.
- Unseen dimensions/instances within a shape family.
- Unseen shape families.
- Unseen mesh resolutions and remeshings.
- Increasing object count and contact density.

## 10. Evaluation hierarchy

### Geometry/contact accuracy

- Near-contact recall and false-negative rate.
- Contact-event precision/recall.
- Signed-distance error.
- Contact-point set distance after matching.
- Normal angular error.
- Contact-cardinality and contact-mode accuracy.
- First-contact timing error.
- Calibration of confidence/uncertainty.

### Dynamics accuracy

- One-step velocity and pose error.
- Impact velocity change.
- Multi-step position/orientation and twist error.
- Penetration and artificial-energy diagnostics.
- Mode-specific performance: free flight, impact, sliding, sticking, rebound, and resting.

### Policy validity

- Policy reward/success inside the neural simulator.
- Reward/success after transferring the unchanged policy to Chrono.
- Failure rate caused specifically by missed or mistimed contacts.

### Compute performance

- Geometry preprocessing cost, reported separately.
- Cached contact-encoder latency and throughput.
- Memory versus object count and mesh resolution.
- Full neural-simulator step time.
- Scaling with number of environments, bodies, candidate pairs, and active contacts.

## 11. Milestones and signs of life

| Milestone | Deliverable | Sign of life |
| --- | --- | --- |
| M0: data contract | Deterministic Chrono recorder and replay test | Replayed episodes reproduce states and contacts within tolerance |
| M1: oracle interface | Chrono contacts fed into NeDM | Contact-aware NeDM improves impact/post-impact rollout over state-only NeDM |
| M2: geometry bake-off | Primitive, point, patch, and SDF baselines | One representation generalizes to held-out geometry without unacceptable runtime |
| M3: learned contact encoder | Geometry/pose to explicit contact tokens | First-contact time, location, normal, and signed distance meet pilot gates |
| M4: closed-loop neural simulation | Learned contact encoder + NeDM | Stable long rollouts with small degradation from the oracle-contact upper bound |
| M5: control demonstration | Policy trained in neural simulation and replayed in Chrono | Unchanged policy retains task performance in Chrono |
| M6: scaling result | Matched collision/contact benchmark | Identified regime where the learned pipeline has a measurable end-to-end advantage |

## 12. Main risks and controls

| Risk | Control |
| --- | --- |
| One fixed object lets the network memorize geometry | Hold out box dimensions first, then hold out shape families |
| Contact positives are rare | Near-contact sampling, event windows, and hard-negative mining |
| Chrono manifold ordering changes | Treat contacts as an unordered set and preserve raw records |
| Geometry discretization changes the label | Use remeshing tests and canonical local coordinates |
| A missed contact destabilizes the rollout | Conservative proximity margin, calibrated uncertainty, and exact fallback during development |
| NRD compensates for a poor contact encoder | Freeze modules during attribution experiments before joint training |
| Policy exploits model errors | Policy-visited Chrono relabeling and held-out Chrono evaluation |
| Contact encoder is fast alone but slow in the full loop | Report cached preprocessing, pair construction, inference, transfer, and full-step runtime separately |

## 13. Immediate execution order

1. Implement the Chrono box-drop recorder and canonical contact schema.
2. Establish the state-only and oracle-contact NeDM baselines.
3. Implement analytic box-plane contact as a correctness baseline.
4. Train the surface-point and surface-patch contact encoders.
5. Select the representation using geometry-held-out accuracy and cached inference cost.
6. Connect predicted contact tokens to the frozen NeDM model.
7. Jointly fine-tune with rollout supervision only after modular attribution is complete.

The accompanying box-drop study plan defines this first sign-of-life experiment in detail.
