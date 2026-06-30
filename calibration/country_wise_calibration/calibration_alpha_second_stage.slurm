#!/bin/bash
#SBATCH --job-name=psm_alpha2
#SBATCH --partition=rome
#SBATCH --account=gisr122457
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=128G
#SBATCH --time=36:00:00
#SBATCH --output=calibration/logs/calibration_alpha_second_stage_%j.out
#SBATCH --error=calibration/logs/calibration_alpha_second_stage_%j.err

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
LHS_SAMPLES="${LHS_SAMPLES:-120}"
DE_MAXITER="${DE_MAXITER:-300}"
N_JOBS="${N_JOBS:-$SLURM_CPUS_PER_TASK}"
COUNTRIES_IN_PARALLEL="${COUNTRIES_IN_PARALLEL:-4}"
TIMEOUT_S="${TIMEOUT_S:-180}"
C_TOLERANCE="${C_TOLERANCE:-0.05}"
C_PENALTY="${C_PENALTY:-100}"
ALPHA_RANK_WEIGHT="${ALPHA_RANK_WEIGHT:-0.5}"
PARAMS_DIR="${PARAMS_DIR:-calibration/calibration_results/country_wise_alpha_frozen}"
OUTPUT_DIR="${OUTPUT_DIR:-calibration/calibration_results/country_wise_alpha_second_stage}"
RESUME="${RESUME:-1}"
NO_PLOTS="${NO_PLOTS:-0}"

FAST_DE_POPSIZE="${FAST_DE_POPSIZE:-6}"
export FAST_DE_POPSIZE

echo "Job started: $(date)"
echo "Node: ${SLURMD_NODENAME:-unknown}"
echo "CPUs: ${SLURM_CPUS_PER_TASK:-unknown}"
echo "N_JOBS=$N_JOBS"
echo "COUNTRIES_IN_PARALLEL=$COUNTRIES_IN_PARALLEL"
echo "FAST_DE_POPSIZE=$FAST_DE_POPSIZE"
echo "LHS_SAMPLES=$LHS_SAMPLES"
echo "DE_MAXITER=$DE_MAXITER"
echo "TIMEOUT_S=$TIMEOUT_S"
echo "C_TOLERANCE=$C_TOLERANCE"
echo "C_PENALTY=$C_PENALTY"
echo "ALPHA_RANK_WEIGHT=$ALPHA_RANK_WEIGHT"
echo "PARAMS_DIR=$PARAMS_DIR"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "RESUME=$RESUME"
echo "NO_PLOTS=$NO_PLOTS"
echo "COUNTRY_CODES=${COUNTRY_CODES:-<all>}"

python - <<'PY'
import importlib.util

required_modules = ["numpy", "scipy", "numba", "matplotlib"]
missing = [name for name in required_modules if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit(f"Missing Python packages in active environment: {missing}")

print("Preflight checks passed.")
PY

CMD=(
  python -u calibration/country_wise_calibration/calibration_alpha_second_stage.py
  --params-dir "$PARAMS_DIR"
  --output-dir "$OUTPUT_DIR"
  --n-jobs "$N_JOBS"
  --countries-in-parallel "$COUNTRIES_IN_PARALLEL"
  --lhs-samples "$LHS_SAMPLES"
  --de-maxiter "$DE_MAXITER"
  --timeout-s "$TIMEOUT_S"
  --c-tolerance "$C_TOLERANCE"
  --c-penalty "$C_PENALTY"
  --alpha-rank-weight "$ALPHA_RANK_WEIGHT"
)

if [[ "$RESUME" == "1" ]]; then
  CMD+=(--resume)
fi

if [[ "$NO_PLOTS" == "1" ]]; then
  CMD+=(--no-plots)
fi

if [[ -n "$COUNTRY_CODES" ]]; then
  read -r -a COUNTRY_ARRAY <<< "$COUNTRY_CODES"
  CMD+=(--countries "${COUNTRY_ARRAY[@]}")
fi

printf 'Running command:'
printf ' %q' "${CMD[@]}"
printf '\n'

"${CMD[@]}"

echo "Job finished: $(date)"
