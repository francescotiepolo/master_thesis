#!/bin/bash
#SBATCH --job-name=psm_regression
#SBATCH --partition=rome
#SBATCH --account=gisr122457
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=calibration/logs/regression_%j.out
#SBATCH --error=calibration/logs/regression_%j.err

set -euo pipefail

module purge
module load 2023
module load Python/3.11.3-GCCcore-12.3.0
source $HOME/envs/thesis/bin/activate

export ON_CLUSTER=1
export REGRESSION_N_JOBS="$SLURM_CPUS_PER_TASK"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib_${SLURM_JOB_ID:-$$}"
mkdir -p "$MPLCONFIGDIR"

cd $SLURM_SUBMIT_DIR  # must be master_thesis root
mkdir -p calibration/logs

echo "Job started: $(date)"
echo "Node: $SLURMD_NODENAME  CPUs: $SLURM_CPUS_PER_TASK  REGRESSION_N_JOBS=$REGRESSION_N_JOBS"

python - <<'PY'
import importlib.util
required = ["numpy", "scipy", "sklearn", "joblib", "matplotlib", "pandas"]
missing = [m for m in required if importlib.util.find_spec(m) is None]
if missing:
    raise SystemExit(f"Missing Python packages: {missing}")
print("Preflight checks passed.")
PY

# Country growth regression: produces pi_matrix_fixed.npy + r_values.csv
# (~SC=19 units, bottlenecked by RidgeCV + 200 bootstrap fits per country)
echo
echo "=== Country growth regression ==="
python -u calibration/growth_regression.py

# Product growth regression: produces pi_competition_P.npy + r_P_regression.npy
# (~SP=100 units, same Ridge + bootstrap structure, fully parallelizable)
echo
echo "=== Product growth regression ==="
python -u calibration/product_growth_regression.py

echo
echo "Job finished: $(date)"
