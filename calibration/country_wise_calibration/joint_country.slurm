#!/bin/bash
#SBATCH --job-name=psm_joint_country
#SBATCH --partition=rome
#SBATCH --account=gisr122457
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=calibration/logs/joint_country_%j.out
#SBATCH --error=calibration/logs/joint_country_%j.err

set -euo pipefail

module purge
module load 2023
module load Python/3.11.3-GCCcore-12.3.0
source $HOME/envs/thesis/bin/activate

export ON_CLUSTER=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib_${SLURM_JOB_ID:-$$}"
mkdir -p "$MPLCONFIGDIR"

cd "$SLURM_SUBMIT_DIR"
mkdir -p calibration/logs

COUNTRY_N_JOBS="${COUNTRY_N_JOBS:-$SLURM_CPUS_PER_TASK}"
COUNTRIES_IN_PARALLEL="${COUNTRIES_IN_PARALLEL:-4}"
COUNTRY_LHS_SAMPLES="${COUNTRY_LHS_SAMPLES:-}"
COUNTRY_CODES="${COUNTRY_CODES:-}"

echo "Job started: $(date)"
echo "Node: $SLURMD_NODENAME  CPUs: $SLURM_CPUS_PER_TASK"
echo "COUNTRY_N_JOBS=$COUNTRY_N_JOBS  COUNTRIES_IN_PARALLEL=$COUNTRIES_IN_PARALLEL"
if [ -n "$COUNTRY_LHS_SAMPLES" ]; then
  echo "COUNTRY_LHS_SAMPLES=$COUNTRY_LHS_SAMPLES"
fi
if [ -n "$COUNTRY_CODES" ]; then
  echo "COUNTRY_CODES=$COUNTRY_CODES"
fi

python - <<'PY'
import importlib.util
required = ["numpy", "scipy", "matplotlib"]
missing = [m for m in required if importlib.util.find_spec(m) is None]
if missing:
    raise SystemExit(f"Missing Python packages: {missing}")
print("Preflight checks passed.")
PY

cmd=(
  python -u calibration/country_wise_calibration/calibration_country_wise.py
  --n-jobs "$COUNTRY_N_JOBS"
  --countries-in-parallel "$COUNTRIES_IN_PARALLEL"
)

if [ -n "$COUNTRY_LHS_SAMPLES" ]; then
  cmd+=(--lhs-samples "$COUNTRY_LHS_SAMPLES")
fi

if [ -n "$COUNTRY_CODES" ]; then
  read -r -a codes <<< "$COUNTRY_CODES"
  cmd+=(--countries "${codes[@]}")
fi

"${cmd[@]}"

echo "Job finished: $(date)"
