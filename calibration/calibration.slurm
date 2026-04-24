#!/bin/bash
#SBATCH --job-name=psm_calibration
#SBATCH --partition=rome
#SBATCH --account=gisr122457
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=224G
#SBATCH --time=16:00:00
#SBATCH --output=calibration/logs/calibration_%j.out
#SBATCH --error=calibration/logs/calibration_%j.err

set -euo pipefail

module purge
module load 2023
module load Python/3.11.3-GCCcore-12.3.0
source $HOME/envs/thesis/bin/activate

export ON_CLUSTER=1
export CALIB_MAX_JOBS=128
export LHS_CHUNK_SIZE=1000
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib_${SLURM_JOB_ID:-$$}"
mkdir -p "$MPLCONFIGDIR"

cd $SLURM_SUBMIT_DIR  # must be master_thesis root
mkdir -p calibration/logs

echo "Job started: $(date)"
echo "Node: $SLURMD_NODENAME  CPUs: $SLURM_CPUS_PER_TASK"

python - <<'PY'
import importlib.util
import os
import sys

required_modules = ["numpy", "scipy", "numba", "joblib", "matplotlib", "pymoo"]
missing = [name for name in required_modules if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit(f"Missing Python packages in active environment: {missing}")

print("Preflight checks passed.")
PY

# Uncomment the phases you want to run.
# Dependency order:
#   extract -> lhs -> nsga2 -> validate

# Phase 0: data extraction
# python -u calibration/run_pipeline.py extract

# Phase 1: initial design
 python -u calibration/run_pipeline.py lhs "$SLURM_CPUS_PER_TASK"

# Phase 2: NSGA-II multi-objective optimizer
python -u calibration/run_pipeline.py nsga2 "$SLURM_CPUS_PER_TASK"

# Phase 3: validation on the best point
python -u calibration/run_pipeline.py validate "$SLURM_CPUS_PER_TASK"

echo "Job finished: $(date)"