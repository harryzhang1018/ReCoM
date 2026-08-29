#!/usr/bin/env bash
# Encoder-decoder study, closed-loop arm (plan stage ED4, 2026-08-28): BASE-64 and JL-6-R trained with the 8-step unrolled
# loss (--rollout-horizon 8: predicted states, contacts and wrench recomputed every unrolled step), 30 k steps x 3 seeds.
# Uses the decoder from submit_ed_wrench.sh (runs/$DECNAME/final.pt must exist).
set -euo pipefail
mkdir -p cluster/logs runs
P=${PARTITION:-research}; W=2; T="cluster/train.sbatch"; LONG="--time=24:00:00"
ENC=runs/expC_point_30k/final.pt; DECNAME=${DECNAME:-ed2_dec_v5}; D="--decoder-ckpt runs/$DECNAME/final.pt"
COMMON="--data data/pilot1b --steps 30000 --eval-every 10000 --workers $W --encoder-ckpt $ENC --eval-contact-sources learned,analytic --train-contact-source learned --rollout-horizon 8"
sub() { sbatch --parsable --partition=$P "$@"; }
JOBS=""
for s in 0 1 2; do
  B=$(sub --job-name=ed4_base64_h8_s$s $LONG $T scripts/train_transition.py $COMMON --contact-mode explicit --seed $s --out runs/ed4_base64_h8_s$s)
  R=$(sub --job-name=ed4_jl6r_h8_s$s $LONG $T scripts/train_transition.py $COMMON --contact-mode wrench --physics-residual $D --seed $s --out runs/ed4_jl6r_h8_s$s)
  JOBS="$JOBS${JOBS:+:}$B:$R"
done
S=$(sub --dependency=afterany:$JOBS cluster/summary.sbatch)
echo "ed4 jobs: $JOBS summary: $S"
squeue -u $USER -o "%.10i %.20j %.9P %.2t %.10M %R" | head -30
