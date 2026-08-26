#!/usr/bin/env bash
#SBATCH --job-name=recom-summary
#SBATCH --partition=research
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=cluster/logs/%x-%j.out
source cluster/env.sh
python scripts/summarize_runs.py runs | tee docs/RESULTS_cluster.md
