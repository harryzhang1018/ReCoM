#!/usr/bin/env bash
# Submit the whole Study-1 pipeline with SLURM dependencies (run from ~/ReCoM on Euler):
#   gen_data -> {C_patch, C_point, C_analytic, A_fixed, B_fixed, A_pilot, B_pilot} -> {D, E} -> summary
# Options: SKIP_GEN=1 (data already present), PARTITION=sbel|research (default research).
set -euo pipefail
mkdir -p cluster/logs runs
P=${PARTITION:-research}
STEPS_T=${STEPS_T:-30000}     # transition-model steps
STEPS_C=${STEPS_C:-8000}      # contact-encoder steps
W=2                           # dataloader workers
sub() { sbatch --parsable --partition=$P "$@"; }

if [[ "${SKIP_GEN:-0}" == "1" ]]; then DEP=""; else
  GEN=$(sub cluster/gen_data.sbatch); echo "gen_data $GEN"; DEP="--dependency=afterok:$GEN"; fi

T="cluster/train.sbatch"
CP=$(sub $DEP --job-name=expC_patch    $T scripts/train_contact.py --data data/pilot1b --encoder patch    --out runs/expC_patch    --steps $STEPS_C --eval-every 2000 --max-eval-episodes 120 --workers $W)
CQ=$(sub $DEP --job-name=expC_point    $T scripts/train_contact.py --data data/pilot1b --encoder point    --out runs/expC_point    --steps $STEPS_C --eval-every 2000 --max-eval-episodes 120 --workers $W)
CA=$(sub $DEP --job-name=expC_analytic $T scripts/train_contact.py --data data/pilot1b --encoder analytic --out runs/expC_analytic --max-eval-episodes 120)
AF=$(sub $DEP --job-name=expA_fixed $T scripts/train_transition.py --data data/fixed1a --contact-mode none     --out runs/expA_state_only      --steps $STEPS_T --eval-every 10000 --workers $W --eval-contact-sources gt)
BF=$(sub $DEP --job-name=expB_fixed $T scripts/train_transition.py --data data/fixed1a --contact-mode explicit --out runs/expB_oracle_explicit --steps $STEPS_T --eval-every 10000 --workers $W --eval-contact-sources gt,analytic)
AP=$(sub $DEP --job-name=expA_pilot $T scripts/train_transition.py --data data/pilot1b --contact-mode none     --out runs/expA_pilot_state_only      --steps $STEPS_T --eval-every 10000 --workers $W --eval-contact-sources gt)
BP=$(sub $DEP --job-name=expB_pilot $T scripts/train_transition.py --data data/pilot1b --contact-mode explicit --out runs/expB_pilot_oracle_explicit --steps $STEPS_T --eval-every 10000 --workers $W --eval-contact-sources gt,analytic)
echo "C: $CP $CQ $CA   A/B: $AF $BF $AP $BP"

D=$(sub --dependency=afterok:$CP --job-name=expD_learned $T scripts/train_transition.py --data data/pilot1b --contact-mode explicit --out runs/expD_learned_frozen --steps $STEPS_T --eval-every 10000 --workers $W --encoder-ckpt runs/expC_patch/final.pt --eval-contact-sources gt,analytic,learned)
E=$(sub --dependency=afterok:$CP --job-name=expE_joint   $T scripts/train_transition.py --data data/pilot1b --contact-mode explicit+latent --train-contact-source learned --encoder-ckpt runs/expC_patch/final.pt --finetune-encoder --rollout-horizon 8 --out runs/expE_joint --steps $STEPS_T --eval-every 10000 --workers $W --eval-contact-sources learned,analytic)
echo "D/E: $D $E"

S=$(sub --dependency=afterany:$CP:$CQ:$CA:$AF:$BF:$AP:$BP:$D:$E cluster/summary.sbatch)
echo "summary: $S"
squeue -u $USER -o "%.10i %.14j %.9P %.2t %.10M %R" | head -20
