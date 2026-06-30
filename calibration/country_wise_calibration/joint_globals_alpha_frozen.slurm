#!/bin/bash
#SBATCH --job-name=psm_joint_globals_af
#SBATCH --partition=rome
#SBATCH --account=gisr122457
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=224G
#SBATCH --time=12:00:00
#SBATCH --output=calibration/logs/joint_globals_alpha_frozen_%j.out
#SBATCH --error=calibration/logs/joint_globals_alpha_frozen_%j.err

set -euo pipefail

module purge
module load 2023
module load Python/3.11.3-GCCcore-12.3.0
source "$HOME/envs/thesis/bin/activate"

export ON_CLUSTER=1
export STAGE3_MODE=alpha_frozen
export STAGE3_COUNTRY_SUMMARY="${STAGE3_COUNTRY_SUMMARY:-calibration/calibration_results/country_wise_alpha_frozen/summary.csv}"
export STAGE3_OUTPUT_DIR="${STAGE3_OUTPUT_DIR:-calibration/calibration_results/joint_alpha_frozen}"
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
echo "Node: ${SLURMD_NODENAME:-unknown}  CPUs: ${SLURM_CPUS_PER_TASK:-unknown}"
echo "STAGE3_MODE=$STAGE3_MODE"
echo "STAGE3_COUNTRY_SUMMARY=$STAGE3_COUNTRY_SUMMARY"
echo "STAGE3_OUTPUT_DIR=$STAGE3_OUTPUT_DIR"
echo "DE_WORKERS=$DE_WORKERS  DE_N_CANDIDATES=$DE_N_CANDIDATES"

python - <<'PY'
import importlib.util
import os
from pathlib import Path

required = ["numpy", "scipy", "matplotlib"]
missing = [m for m in required if importlib.util.find_spec(m) is None]
if missing:
    raise SystemExit(f"Missing Python packages: {missing}")

required_files = [
    os.environ["STAGE3_COUNTRY_SUMMARY"],
    "calibration/calibration_results/joint/bootstrap_globals.json",
    "calibration/calibration_results/growth_regression/r_P_regression.npy",
    "calibration/calibration_results/growth_regression/pi_competition_P.npy",
    "calibration/calibration_results/growth_regression/r_values_fixed.csv",
]
missing_files = [p for p in required_files if not Path(p).exists()]
if missing_files:
    raise SystemExit(f"Missing required input files: {missing_files}")

positive_ints = ("DE_WORKERS", "DE_N_CANDIDATES", "DE_MAXITER")
bad = []
parsed = {}
for name in positive_ints:
    try:
        value = int(os.environ[name])
    except (KeyError, ValueError):
        bad.append(f"{name}={os.environ.get(name)!r}")
        continue
    parsed[name] = value
    if value <= 0:
        bad.append(f"{name}={value}")
if parsed.get("DE_N_CANDIDATES", 5) < 5:
    bad.append(f"DE_N_CANDIDATES={parsed['DE_N_CANDIDATES']} (<5)")
if bad:
    raise SystemExit(f"Invalid DE settings: {bad}")

print("Preflight checks passed.")
PY

python -u calibration/country_wise_calibration/shared_global_calibration.py

echo "Job finished: $(date)"
