#!/usr/bin/env bash
#SBATCH --job-name=recom-gen
#SBATCH --partition=research
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=cluster/logs/%x-%j.out
# Generate + validate the three Study-1 datasets (deterministic; ~1 min each on 32 cores).
source cluster/env.sh
python -m pytest tests -q -x
for c in smoke fixed1a pilot1b; do
  python scripts/generate_dataset.py configs/$c.yaml --workers ${SLURM_CPUS_PER_TASK}
  python scripts/validate_dataset.py data/$c --replay 3
done
echo GEN_DONE
