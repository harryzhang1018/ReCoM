#!/usr/bin/env bash
# Encoder-decoder study, pass 1 (2026-08-27): impulse-label audit -> impulse decoder (ED2) -> wrench-conditioned
# transition models (ED3) BASE-64 / J-3 / JL-6 / JL-6-R x 3 seeds, plus reference rows -> summary.
# All ED3 variants train on the learned point-encoder contacts (frozen), with the gravity + gyroscopic priors and the hard
# contact gate; the old gt-trained frozen NeDM (expD_r2_learned_frozen_point) is re-evaluated with the new metrics as a
# reference row, and ed3_base64_gt_s0 (explicit, gt-trained, gyro prior) isolates the effect of the prior.
# Assumes data/pilot1b and runs/expC_point_30k/final.pt exist on the cluster.
set -euo pipefail
mkdir -p cluster/logs runs results/audit_impulse_labels
P=${PARTITION:-research}; W=2; T="cluster/train.sbatch"
ENC=runs/expC_point_30k/final.pt
COMMON="--data data/pilot1b --steps 30000 --eval-every 10000 --workers $W --encoder-ckpt $ENC --eval-contact-sources learned,analytic"
sub() { sbatch --parsable --partition=$P "$@"; }
LONG="--time=16:00:00"   # learned-contact training runs the point encoder on 8192 frames per step (~0.33 s/step on a 4090)
AUD=$(sub --job-name=ed0_audit --gres=gpu:0 --time=01:00:00 $T scripts/audit_impulse_labels.py --data data/pilot1b --out results/audit_impulse_labels/pilot1b)
DEC=$(sub --job-name=ed2_dec_cone --time=04:00:00 $T scripts/train_impulse.py --data data/pilot1b --encoder-ckpt $ENC --out runs/ed2_dec_cone --steps 20000 --batch 512 --eval-every 5000 --workers $W)
DEP="--dependency=afterok:$DEC"
JOBS="$AUD:$DEC"
for s in 0 1 2; do
  B=$(sub --job-name=ed3_base64_s$s $LONG $T scripts/train_transition.py $COMMON --contact-mode explicit --train-contact-source learned --seed $s --out runs/ed3_base64_s$s)
  J=$(sub $DEP --job-name=ed3_j3_s$s $LONG $T scripts/train_transition.py $COMMON --contact-mode wrench_lin --train-contact-source learned --decoder-ckpt runs/ed2_dec_cone/final.pt --seed $s --out runs/ed3_j3_s$s)
  L=$(sub $DEP --job-name=ed3_jl6_s$s $LONG $T scripts/train_transition.py $COMMON --contact-mode wrench --train-contact-source learned --decoder-ckpt runs/ed2_dec_cone/final.pt --seed $s --out runs/ed3_jl6_s$s)
  R=$(sub $DEP --job-name=ed3_jl6r_s$s $LONG $T scripts/train_transition.py $COMMON --contact-mode wrench --physics-residual --train-contact-source learned --decoder-ckpt runs/ed2_dec_cone/final.pt --seed $s --out runs/ed3_jl6r_s$s)
  JOBS="$JOBS:$B:$J:$L:$R"
done
# reference rows
G=$(sub --job-name=ed3_base64_gt_s0 $LONG $T scripts/train_transition.py $COMMON --contact-mode explicit --train-contact-source gt --seed 0 --out runs/ed3_base64_gt_s0)
REF=$(sub --job-name=ed3_ref_expD $T scripts/train_transition.py --data data/pilot1b --contact-mode explicit --encoder-ckpt $ENC --eval-only runs/expD_r2_learned_frozen_point/final.pt --eval-contact-sources learned,analytic --workers $W --out runs/ed3_ref_expD_point)
# wrench-usage gate (#7): re-evaluate seed-0 JL-6 with the wrench zeroed / shuffled
Z=$(sub --dependency=afterok:$DEC --job-name=ed3_jl6_zero $T scripts/train_transition.py --data data/pilot1b --contact-mode wrench --train-contact-source learned --encoder-ckpt $ENC --decoder-ckpt runs/ed2_dec_cone/final.pt --wrench-ablation zero --eval-only runs/ed3_jl6_s0/final.pt --eval-contact-sources learned --workers $W --out runs/ed3_jl6_s0_wrench_zero)
S=$(sub --dependency=afterany:$JOBS:$G:$REF:$Z cluster/summary.sbatch)
echo "audit: $AUD decoder: $DEC ed3: $JOBS ref: $G $REF ablation: $Z summary: $S"
squeue -u $USER -o "%.10i %.20j %.9P %.2t %.10M %R" | head -30
