#!/bin/bash
#SBATCH --job-name=psm_joint_pack
#SBATCH --partition=rome
#SBATCH --account=gisr122457
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=calibration/logs/joint_pack_%j.out
#SBATCH --error=calibration/logs/joint_pack_%j.err

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

echo "Job started: $(date)"
echo "Node: $SLURMD_NODENAME"

python - <<'PY'
import importlib.util
required = ["numpy"]
missing = [m for m in required if importlib.util.find_spec(m) is None]
if missing:
    raise SystemExit(f"Missing Python packages: {missing}")
print("Preflight checks passed.")
PY

python -u calibration/country_wise_calibration/joint_params.py

echo "Job finished: $(date)"
