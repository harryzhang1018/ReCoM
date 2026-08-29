#!/usr/bin/env bash
# Encoder-decoder study, wrench arm (resubmission after the decoder fixes, 2026-08-28): decoder v5 (exact Chrono gap from the
# predicted point, Delassus-scaled head, yaw augmentation, no slot embedding, near-L1 loss) -> J-3 / JL-6 / JL-6-R x 3 seeds ->
# wrench-usage gate (zeroed wrench) -> summary.  BASE-64 seeds and the reference rows were submitted by submit_ed.sh.
set -euo pipefail
mkdir -p cluster/logs runs
P=${PARTITION:-research}; W=2; T="cluster/train.sbatch"; LONG="--time=16:00:00"
ENC=runs/expC_point_30k/final.pt
DECNAME=${DECNAME:-ed2_dec_v5}
COMMON="--data data/pilot1b --steps 30000 --eval-every 10000 --workers $W --encoder-ckpt $ENC --eval-contact-sources learned,analytic --train-contact-source learned"
sub() { sbatch --parsable --partition=$P "$@"; }
DEC=$(sub --job-name=$DECNAME --time=04:00:00 $T scripts/train_impulse.py --data data/pilot1b --encoder-ckpt $ENC --out runs/$DECNAME --steps 20000 --batch 512 --eval-every 10000 --workers $W --no-slot-embedding --yaw-aug --head-scale delassus)
DEP="--dependency=afterok:$DEC"; JOBS="$DEC"; D="--decoder-ckpt runs/$DECNAME/final.pt"
for s in 0 1 2; do
  J=$(sub $DEP --job-name=ed3_j3_s$s $LONG $T scripts/train_transition.py $COMMON --contact-mode wrench_lin $D --seed $s --out runs/ed3_j3_s$s)
  L=$(sub $DEP --job-name=ed3_jl6_s$s $LONG $T scripts/train_transition.py $COMMON --contact-mode wrench $D --seed $s --out runs/ed3_jl6_s$s)
  R=$(sub $DEP --job-name=ed3_jl6r_s$s $LONG $T scripts/train_transition.py $COMMON --contact-mode wrench --physics-residual $D --seed $s --out runs/ed3_jl6r_s$s)
  JOBS="$JOBS:$J:$L:$R"
done
Z=$(sub --dependency=afterok:$JOBS --job-name=ed3_jl6_zero $T scripts/train_transition.py --data data/pilot1b --contact-mode wrench --train-contact-source learned --encoder-ckpt $ENC $D --wrench-ablation zero --eval-only runs/ed3_jl6_s0/final.pt --eval-contact-sources learned --workers $W --out runs/ed3_jl6_s0_wrench_zero)
S=$(sub --dependency=afterany:$JOBS:$Z cluster/summary.sbatch)
echo "decoder: $DEC wrench runs: $JOBS ablation: $Z summary: $S"
squeue -u $USER -o "%.10i %.20j %.9P %.2t %.10M %R" | head -30
