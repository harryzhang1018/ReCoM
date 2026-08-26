# Source this at the top of every cluster job/script (Euler): conda via the site module.
module load conda/miniforge
bootstrap-conda
conda activate recom
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-8}
export PYTHONUNBUFFERED=1
cd "${RECOM_ROOT:-$HOME/ReCoM}"
