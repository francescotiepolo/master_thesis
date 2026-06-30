#!/bin/bash
#SBATCH --job-name=psm_joint_globals
#SBATCH --partition=rome
#SBATCH --account=gisr122457
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=224G
#SBATCH --time=12:00:00
#SBATCH --output=calibration/logs/joint_globals_%j.out
#SBATCH --error=calibration/logs/joint_globals_%j.err

set -euo pipefail

module purge
module load 2023
module load Python/3.11.3-GCCcore-12.3.0
source $HOME/envs/thesis/bin/activate

export ON_CLUSTER=1
export DE_WORKERS="${DE_WORKERS:-8}"
export DE_N_CANDIDATES="${DE_N_CANDIDATES:-16}"
export DE_MAXITER="${DE_MAXITER:-20}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib_${SLURM_JOB_ID:-$$}"
mkdir -p "$MPLCONFIGDIR"

cd "$SLURM_SUBMIT_DIR"
mkdir -p calibration/logs

echo "Job started: $(date)"
echo "Node: $SLURMD_NODENAME  CPUs: $SLURM_CPUS_PER_TASK"
echo "DE_WORKERS=$DE_WORKERS  DE_N_CANDIDATES=$DE_N_CANDIDATES"

python - <<'PY'
import importlib.util
required = ["numpy", "scipy", "matplotlib"]
missing = [m for m in required if importlib.util.find_spec(m) is None]
if missing:
    raise SystemExit(f"Missing Python packages: {missing}")
print("Preflight checks passed.")
PY

python -u calibration/country_wise_calibration/shared_global_calibration.py

echo "Job finished: $(date)"
