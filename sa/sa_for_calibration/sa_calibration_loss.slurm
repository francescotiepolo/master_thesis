#!/bin/bash
#SBATCH --job-name=sa_calib_loss
#SBATCH --partition=rome
#SBATCH --account=gisr122457
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=224G
#SBATCH --time=12:00:00
#SBATCH --output=sa/sa_for_calibration/logs/sa_calib_loss_%j.out
#SBATCH --error=sa/sa_for_calibration/logs/sa_calib_loss_%j.err

module purge
module load 2023
module load Python/3.11.3-GCCcore-12.3.0
source $HOME/envs/thesis/bin/activate

export ON_CLUSTER=1
export N_JOBS_OVERRIDE=128
export NUMBA_NUM_THREADS=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export NUMBA_CACHE_DIR=/tmp/numba_cache_$$
export MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib_${SLURM_JOB_ID:-$$}"
cd $SLURM_SUBMIT_DIR
mkdir -p sa/sa_for_calibration/logs
mkdir -p "$MPLCONFIGDIR"

echo "Job started: $(date)"
echo "Node: $SLURMD_NODENAME  CPUs: $SLURM_CPUS_PER_TASK"

python -u sa/sa_for_calibration/sa_calibration_loss.py 2>&1

echo "Job finished: $(date)"
