# Review: Contact Encoder–Impulse Decoder Plan

Reviewed 2026-08-27 against the code at HEAD (`b7e0526`) and `docs/PROGRESS.md` (Study 1 complete: learned point
contacts + frozen NeDM match the exact-contact oracle within noise; every model, oracle included, ends with ~90°
orientation error). Implementation of pass 1 (ED0–ED3) landed the same day; see `docs/PROGRESS.md` §9 for the run recipes.

## Verdict

Agree with the direction. Every claim the plan makes about the current code (§3) checks out: `(half_extents, pos, quat)`
encoder inputs, K=4 slots with a 16-D unsupervised `latent`, 19-D adapter features + DeepSets 64-D pooling, 13+3+64 NRD
input predicting `(dv, dw)`, gravity prior, hard contact gate, exact pose integration, encoder re-evaluated from the
predicted state in rollouts, an existing `--rollout-horizon` unrolled loss, and the post-attention slot query not being
returned. The contact geometry is no longer the bottleneck, so a physics-grounded bottleneck between geometry and the
transition model is the right next experiment. The corrections below were folded into the implementation.

## Corrections (verified on `data/pilot1b`)

1. **§5.3 angular-impulse label.** `H[k+1]−H[k]` is not Chrono's discrete update: 0.5 % median / 4.3 % max error. The exact
   body-frame form `L_b = I_b(ω_b[k+1]−ω_b[k]) + dt·(ω_b × I_b ω_b)` is zero to 7e‑13 on 7,920 tumbling free-flight frames
   (Chrono subtracts the gyroscopic torque in `IntLoadResidual_F`). `ω_b` is already stored (`s_ang_vel_local`).
2. **§5.4 force audit is pre-answered.** `c_force_world` is a true force (N) on the box: `m(v[k+1]−v[k]) = (ΣF + m g)dt` to
   3.6e‑15, so `J_state ≡ dt·ΣF` and the recorded contact set is complete. Force-derived torque equals the exact `L` to
   3e‑14 with Chrono's *raw* contact points, but has 1.4 % median / 8 % p90 error with the canonical clamped points the
   encoder predicts — the decoder's aggregated `L` inherits this small lever-arm floor. 25 % of active slots carry exactly
   zero force (envelope contacts), 48–67 % at first impact: activation ≠ impulse; the decoder must output zero for active slots.
3. **Missing physics in the current gate (new finding, affects the baseline).** The gate forces the residual to zero when no
   slot is active, but a tumbling box changes `ω_world` in free flight through the gyroscopic term: median 0.57 rad/s
   (p90 5 rad/s) net per post-impact free-flight run, forced to 0 by every current model. The exact state-dependent prior
   `Δω_w = −dt·R·I_b⁻¹(ω_b × I_b ω_b)` reproduces Chrono to 1e‑12 and needs only inertia *ratios* (half-extents).
   Implemented as `gyro_prior` (default on for the new study); BASE-64 is retrained with it so the comparison is fair.
4. **Units.** Box mass spans 200× (0.125–27 kg). The decoder predicts `j_i/m`; the transition model receives the
   contact-induced `(Δv, Δω) = (J/m, I_w⁻¹L)` in the same units as its targets. The friction cone is scale invariant.
5. **JL-6 vs JL-6-R.** For a single rigid body on fixed ground, Newton–Euler with the exact wrench *is* Chrono's update, so
   NRD in JL-6 only has to learn a known linear map; JL-6-R (physics residual) is the expected ceiling and JL-6 the
   NeDM-transferable form. Decision: report both as co-primary.
6. **Keep the hard contact gate in the wrench modes** (§4.8 omits it). With the gyro prior, "no active slot ⇒ residual 0"
   is exactly true for both `v` and `ω`.
7. **Decoder training needs the learned encoder in the loop** (`slot_embedding`), so wrench-mode transition training uses
   `--train-contact-source learned`; the slot embedding is optional in the decoder (null embedding for gt/analytic geometry)
   so the "Chrono vs predicted geometry" ablation remains possible.
8. **Loss masking.** Frames whose label wrench is nonzero but where the encoder activates no slot are excluded from the
   decoder loss and reported as `missed_impulse_rate`.
9. **Repo conventions.** No YAML training-config system exists (`configs/*.yaml` are dataset generation only): argparse
   flags + `cluster/submit_ed.sh`. No separate `recom/physics/` package.
10. **Orientation metrics.** A general cuboid has only the D2 symmetry (4 elements); the 24-element octahedral group is
    exact only for a cube. `sym_rot_err_deg_final` (D2) is the primary symmetry-aware metric, `_oct` the shape-agnostic bound.

## Scope decisions

Pass 1 (implemented): ED0 audit script + frozen labels, slot-embedding exposure, `ContactImpulseDecoder` (cone head,
set attention over K slots), decoder pretraining, `wrench` / `wrench_lin` transition modes + `physics_residual`, gyro
prior, decoder-aware rollouts, impulse and settled-face/symmetry metrics, wrench zero/shuffle evaluation ablation,
cluster recipe BASE-64 / J-3 / JL-6 / JL-6-R × 3 seeds.
Pass 2 (after ED2/ED3 results): JL-6-C momentum-consistency loss, POOL-DEC, ORACLE-JL, decoder fine-tuning with
per-slot Hungarian-matched loss, free vs cone head, encoder-output cache.
