#!/bin/bash
# Closed-loop evaluation of the v7 decoder loss variants (FULL = drop-in with frozen NRD,
# N-OFF = pure physics path, no NRD compensation). Skips variants already in the summary rows.
set -u
cd /home/harry/ReCoM
source /home/harry/anaconda3/etc/profile.d/conda.sh
conda activate recom
for name in nt rest solv ntrest all; do
  if [ -f "results/ablate_bottleneck/val/rows_FULL_${name}.json" ] && [ -f "results/ablate_bottleneck/val/rows_N-OFF_${name}.json" ]; then
    echo "[eval] ${name} already done, skipping"
    continue
  fi
  echo "[eval] $(date +%H:%M:%S) ${name}"
  python scripts/ablate_bottleneck.py --split val --cells FULL N-OFF \
    --decoder-ckpt "runs/local_ed2_dec_v7_${name}/final.pt" --tag "_${name}" \
    --out results/ablate_bottleneck > "runs/v7_${name}.eval.log" 2>&1 || { echo "[eval] ${name} FAILED"; exit 1; }
done
echo "[eval] $(date +%H:%M:%S) ALL DONE"
