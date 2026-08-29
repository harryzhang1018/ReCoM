#!/usr/bin/env bash
# Re-evaluate finished transition checkpoints on ALL test / held-out-geometry episodes (255 / 301) for tight paired CIs.
#   bash cluster/submit_ed_eval255.sh ed4_base64_h8_s0 ed5_jl6r_v6_h8_s0 ...
set -euo pipefail
mkdir -p cluster/logs runs
P=${PARTITION:-research}; W=0; T="cluster/train.sbatch"; ENC=runs/expC_point_30k/final.pt
sub() { sbatch --parsable --partition=$P "$@"; }
for r in "$@"; do
  A=runs/$r/args.json
  mode=$(python -c "import json;print(json.load(open('$A'))['contact_mode'])"); dec=$(python -c "import json;print(json.load(open('$A')).get('decoder_ckpt') or '')"); pr=$(python -c "import json;print('--physics-residual' if json.load(open('$A')).get('physics_residual') else '')")
  extra=""; [ -n "$dec" ] && extra="--decoder-ckpt $dec"
  J=$(sub --job-name=ev255_$r --time=03:00:00 $T scripts/train_transition.py --data data/pilot1b --contact-mode $mode $pr --train-contact-source learned --encoder-ckpt $ENC $extra --eval-only runs/$r/final.pt --eval-contact-sources learned --max-eval-episodes 400 --workers $W --out runs/${r}_ev255)
  echo "$r -> job $J"
done
