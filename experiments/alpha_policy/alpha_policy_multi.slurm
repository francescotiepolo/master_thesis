#!/bin/bash
#SBATCH --job-name=alpha_policy_multi
#SBATCH --partition=rome
#SBATCH --account=gisr122457
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=224G
#SBATCH --time=24:00:00
#SBATCH --array=0-3
#SBATCH --output=calibration/logs/alpha_policy_%A_%a.out
#SBATCH --error=calibration/logs/alpha_policy_%A_%a.err

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

# One country per array task. Override the list via the COUNTRIES env var
# (space-separated) and match the array range to its length, e.g.:
#   sbatch --array=0-4 --export=ALL,COUNTRIES="FRA CHN KOR BRA TUR" \
#       experiments/alpha_policy/alpha_policy_multi.slurm
read -r -a COUNTRY_ARR <<< "${COUNTRIES:-FRA CHN KOR CAN}"
if (( SLURM_ARRAY_TASK_ID >= ${#COUNTRY_ARR[@]} )); then
    echo "Array task ${SLURM_ARRAY_TASK_ID} >= ${#COUNTRY_ARR[@]} countries; nothing to do."
    exit 0
fi
COUNTRY="${COUNTRY_ARR[$SLURM_ARRAY_TASK_ID]}"

PARAMS_DIR="${PARAMS_DIR:-calibration/calibration_results/country_wise_alpha_frozen}"
STRENGTH="${STRENGTH:-0.50}"
BASKET_SIZE="${BASKET_SIZE:-10}"
SWEEP_N="${SWEEP_N:-50}"
HEATMAP_N="${HEATMAP_N:-50}"
TIMEOUT_S="${TIMEOUT_S:-600}"
N_JOBS="${N_JOBS:-$SLURM_CPUS_PER_TASK}"

echo "Job started: $(date)"
echo "Array task: ${SLURM_ARRAY_TASK_ID} -> COUNTRY=$COUNTRY"
echo "Node: ${SLURMD_NODENAME:-unknown}"
echo "CPUs: ${SLURM_CPUS_PER_TASK:-unknown}"
echo "PARAMS_DIR=$PARAMS_DIR"
echo "STRENGTH=$STRENGTH BASKET_SIZE=$BASKET_SIZE"
echo "N_JOBS=$N_JOBS TIMEOUT_S=$TIMEOUT_S"

python -u experiments/alpha_policy/run_alpha_injection.py \
  --country "$COUNTRY" \
  --params-dir "$PARAMS_DIR" \
  --strength "$STRENGTH" \
  --basket-size "$BASKET_SIZE" \
  --sweep-n "$SWEEP_N" \
  --heatmap-n "$HEATMAP_N" \
  --timeout-s "$TIMEOUT_S" \
  --n-jobs "$N_JOBS"

echo "Job finished: $(date)"
