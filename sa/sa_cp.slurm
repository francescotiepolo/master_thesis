#!/bin/bash
#SBATCH --job-name=sa_cp
#SBATCH --partition=rome
#SBATCH --account=gisr122457
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=224G
#SBATCH --time=06:00:00
#SBATCH --output=sa/logs/sa_cp_%j.out
#SBATCH --error=sa/logs/sa_cp_%j.err

module purge
module load 2023
module load Python/3.11.3-GCCcore-12.3.0
source $HOME/envs/thesis/bin/activate

export ON_CLUSTER=1
export CP_ONLY=1
export SA_STRUCTURE_SEED=42
export N_JOBS_OVERRIDE=16
export NUMBA_NUM_THREADS=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export NUMBA_CACHE_DIR=/tmp/numba_cache_$$
export MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib_${SLURM_JOB_ID:-$$}"
cd $SLURM_SUBMIT_DIR
mkdir -p sa/logs
mkdir -p "$MPLCONFIGDIR"

echo "Job started: $(date)"
echo "Node: $SLURMD_NODENAME  CPUs: $SLURM_CPUS_PER_TASK"

python -u sensitivity_analysis.py 2>&1

echo "Job finished: $(date)"