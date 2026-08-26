#!/usr/bin/env bash
# Round 2 (2026-08-26): longer contact-encoder training, then D and E (E now with true joint fine-tuning + retained contact loss).
#   C_patch_30k -> {D_r2, E_r2} -> summary ;  C_point_30k in parallel.   Assumes data/ exists (SKIP_GEN).
set -euo pipefail
mkdir -p cluster/logs runs
P=${PARTITION:-research}; W=2; T="cluster/train.sbatch"
sub() { sbatch --parsable --partition=$P "$@"; }
CP=$(sub --job-name=expC_patch_30k $T scripts/train_contact.py --data data/pilot1b --encoder patch --out runs/expC_patch_30k --steps 30000 --batch 512 --eval-every 5000 --max-eval-episodes 120 --workers $W)
CQ=$(sub --job-name=expC_point_30k $T scripts/train_contact.py --data data/pilot1b --encoder point --out runs/expC_point_30k --steps 30000 --batch 512 --eval-every 5000 --max-eval-episodes 120 --workers $W)
D=$(sub --dependency=afterok:$CP --job-name=expD_r2 $T scripts/train_transition.py --data data/pilot1b --contact-mode explicit --out runs/expD_r2_learned_frozen --steps 30000 --eval-every 10000 --workers $W --encoder-ckpt runs/expC_patch_30k/final.pt --eval-contact-sources gt,analytic,learned)
E=$(sub --dependency=afterok:$CP --job-name=expE_r2 $T scripts/train_transition.py --data data/pilot1b --contact-mode explicit+latent --train-contact-source learned --encoder-ckpt runs/expC_patch_30k/final.pt --finetune-encoder --contact-loss-weight 1.0 --rollout-horizon 8 --out runs/expE_r2_joint --steps 30000 --eval-every 10000 --workers $W --eval-contact-sources learned,analytic)
S=$(sub --dependency=afterany:$CP:$CQ:$D:$E cluster/summary.sbatch)
echo "C: $CP $CQ  D: $D  E: $E  summary: $S"
squeue -u $USER -o "%.10i %.16j %.9P %.2t %.10M %R" | head
