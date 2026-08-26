#!/usr/bin/env bash
# One-time environment creation on Euler (login node is fine; takes ~10 min).
set -euo pipefail
module load conda/miniforge
bootstrap-conda
conda create -n recom python=3.12 -y
conda install -n recom projectchrono::pychrono -c conda-forge -y
conda run -n recom pip install torch numpy scipy pyarrow pyyaml tqdm matplotlib pytest tensorboard
conda run -n recom python -c "import pychrono, torch; print('pychrono OK, torch', torch.__version__, 'cuda', torch.cuda.is_available())"
