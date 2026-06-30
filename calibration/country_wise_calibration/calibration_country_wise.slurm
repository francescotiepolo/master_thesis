#!/bin/bash
#SBATCH --job-name=psm_countrywise
#SBATCH --partition=rome
#SBATCH --account=gisr122457
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=calibration/logs/calibration_country_wise_%j.out
#SBATCH --error=calibration/logs/calibration_country_wise_%j.err

set -euo pipefail

module purge
module load 2023
module load Python/3.11.3-GCCcore-12.3.0
source "$HOME/envs/thesis/bin/activate"

export ON_CLUSTER=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib_${SLURM_JOB_ID:-$$}"
mkdir -p "$MPLCONFIGDIR"

cd "$SLURM_SUBMIT_DIR"
mkdir -p calibration/logs

# Optional overrides:
COUNTRY_CODES="${COUNTRY_CODES:-}"
LHS_SAMPLES="${LHS_SAMPLES:-350}"
N_JOBS="${N_JOBS:-$SLURM_CPUS_PER_TASK}"
COUNTRIES_IN_PARALLEL="${COUNTRIES_IN_PARALLEL:-4}"
MODE="${MODE:-alpha_frozen}"  # conditioned | alpha_frozen
COUNTRY_SEED_SUMMARY="${COUNTRY_SEED_SUMMARY:-calibration/calibration_results/old_results/country_wise_alpha_frozen/summary.csv}"
export COUNTRY_SEED_SUMMARY
COUNTRY_START_FROM="${COUNTRY_START_FROM:-}"
export COUNTRY_START_FROM

FAST_DE_POPSIZE="${FAST_DE_POPSIZE:-6}"
export FAST_DE_POPSIZE

echo "Job started: $(date)"
echo "Node: ${SLURMD_NODENAME:-unknown}"
echo "CPUs: ${SLURM_CPUS_PER_TASK:-unknown}"
echo "MODE=$MODE"
echo "N_JOBS=$N_JOBS"
echo "FAST_DE_POPSIZE=$FAST_DE_POPSIZE"
echo "LHS_SAMPLES=$LHS_SAMPLES"
echo "COUNTRIES_IN_PARALLEL=$COUNTRIES_IN_PARALLEL"
echo "COUNTRY_CODES=${COUNTRY_CODES:-<all>}"
echo "COUNTRY_SEED_SUMMARY=$COUNTRY_SEED_SUMMARY"

python - <<'PY'
import importlib.util

required_modules = ["numpy", "scipy", "numba", "matplotlib"]
missing = [name for name in required_modules if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit(f"Missing Python packages in active environment: {missing}")

print("Preflight checks passed.")
PY

CMD=(
  python -u calibration/country_wise_calibration/calibration_country_wise.py
  --n-jobs "$N_JOBS"
  --lhs-samples "$LHS_SAMPLES"
  --countries-in-parallel "$COUNTRIES_IN_PARALLEL"
  --mode "$MODE"
)

if [[ -n "$COUNTRY_CODES" ]]; then
  read -r -a COUNTRY_ARRAY <<< "$COUNTRY_CODES"
  CMD+=(--countries "${COUNTRY_ARRAY[@]}")
fi

printf 'Running command:'
printf ' %q' "${CMD[@]}"
printf '\n'

"${CMD[@]}"

echo "Job finished: $(date)"
