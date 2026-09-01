#!/bin/bash
# Decoder loss-variant sweep (PROGRESS §9.12 follow-up): n/t channel weighting, resting-equilibrium
# regularizer, low-speed solver anchoring, and combinations. Skips variants whose final.pt exists,
# so it is safe to re-run after an interruption.
set -u
cd /home/harry/ReCoM
source /home/harry/anaconda3/etc/profile.d/conda.sh
conda activate recom

BASE="python scripts/train_impulse.py --data data/pilot1b --encoder-ckpt runs/expC_point_30k/final.pt \
  --no-slot-embedding --yaw-aug --head-scale solver --steps 20000 --eval-every 10000 --workers 2"

run() {  # run <name> <extra flags...>
  local name=$1; shift
  if [ -f "runs/local_ed2_dec_v7_${name}/final.pt" ]; then
    echo "[sweep] v7_${name} already done, skipping"
    return 0
  fi
  echo "[sweep] $(date +%H:%M:%S) training v7_${name} $*"
  $BASE --out "runs/local_ed2_dec_v7_${name}" "$@" > "runs/v7_${name}.train.log" 2>&1 \
    && echo "[sweep] v7_${name} DONE" || { echo "[sweep] v7_${name} FAILED"; exit 1; }
}

run nt     --tan-weight 2
run rest   --rest-eq-weight 1.0
run solv   --solver-anchor-weight 1.0
run ntrest --tan-weight 2 --rest-eq-weight 1.0
run all    --tan-weight 2 --rest-eq-weight 1.0 --solver-anchor-weight 1.0
echo "[sweep] $(date +%H:%M:%S) ALL DONE"
