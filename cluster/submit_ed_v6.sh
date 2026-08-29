#!/usr/bin/env bash
# Encoder-decoder study, decoder v6 arm (2026-08-28): decoder with the closed-form frictional single-contact solver prior
# (--head-scale solver) -> JL-6-R and JL-6 (30 k steps, 3 seeds) and JL-6-R with the 8-step unrolled loss (3 seeds).
set -euo pipefail
mkdir -p cluster/logs runs
P=${PARTITION:-research}; W=2; T="cluster/train.sbatch"; LONG="--time=16:00:00"; XLONG="--time=24:00:00"
ENC=runs/expC_point_30k/final.pt; DECNAME=ed2_dec_v6; D="--decoder-ckpt runs/$DECNAME/final.pt"
COMMON="--data data/pilot1b --steps 30000 --eval-every 10000 --workers $W --encoder-ckpt $ENC --eval-contact-sources learned,analytic --train-contact-source learned"
sub() { sbatch --parsable --partition=$P "$@"; }
DEC=$(sub --job-name=$DECNAME --time=04:00:00 $T scripts/train_impulse.py --data data/pilot1b --encoder-ckpt $ENC --out runs/$DECNAME --steps 20000 --batch 512 --eval-every 10000 --workers $W --no-slot-embedding --yaw-aug --head-scale solver)
DEP="--dependency=afterok:$DEC"; JOBS="$DEC"
for s in 0 1 2; do
  R=$(sub $DEP --job-name=ed5_jl6r_v6_s$s $LONG $T scripts/train_transition.py $COMMON --contact-mode wrench --physics-residual $D --seed $s --out runs/ed5_jl6r_v6_s$s)
  L=$(sub $DEP --job-name=ed5_jl6_v6_s$s $LONG $T scripts/train_transition.py $COMMON --contact-mode wrench $D --seed $s --out runs/ed5_jl6_v6_s$s)
  H=$(sub $DEP --job-name=ed5_jl6r_v6_h8_s$s $XLONG $T scripts/train_transition.py $COMMON --contact-mode wrench --physics-residual $D --rollout-horizon 8 --seed $s --out runs/ed5_jl6r_v6_h8_s$s)
  JOBS="$JOBS:$R:$L:$H"
done
S=$(sub --dependency=afterany:$JOBS cluster/summary.sbatch)
echo "decoder: $DEC ed5 jobs: $JOBS summary: $S"
squeue -u $USER -h -o "%.10i %.20j %.2t %.10M %R" | wc -l
