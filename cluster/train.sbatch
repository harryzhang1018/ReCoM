#!/usr/bin/env bash
#SBATCH --job-name=recom-train
#SBATCH --partition=research
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=cluster/logs/%x-%j.out
# Generic single-GPU training job.  Usage:
#   sbatch --job-name=<name> cluster/train.sbatch <script> <args...>
# e.g. sbatch --job-name=expC_patch cluster/train.sbatch scripts/train_contact.py --data data/pilot1b --encoder patch --out runs/expC_patch
source cluster/env.sh
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
echo "CMD: python $*"
python "$@"
echo TRAIN_DONE
