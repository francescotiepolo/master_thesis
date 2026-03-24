#!/bin/bash
#SBATCH --job-name=sa_calib_loss
#SBATCH --partition=rome
#SBATCH --account=cpuucl002
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
cd $SLURM_SUBMIT_DIR  # directory where sbatch was called (must be master_thesis root)
mkdir -p sa/sa_for_calibration/logs

echo "Job started: $(date)"
echo "Node: $SLURMD_NODENAME  CPUs: $SLURM_CPUS_PER_TASK"

python -u sa/sa_for_calibration/sa_calibration_loss.py 2>&1

echo "Job finished: $(date)"