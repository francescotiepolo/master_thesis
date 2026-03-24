#!/bin/bash
#SBATCH --job-name=psm_calibration
#SBATCH --partition=rome
#SBATCH --account=cpuucl002
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=224G
#SBATCH --time=24:00:00
#SBATCH --output=calibration/logs/calibration_%j.out
#SBATCH --error=calibration/logs/calibration_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=francesco.tiepolo@student.uva.nl

module purge
module load 2023
module load Python/3.11.3-GCCcore-12.3.0
source $HOME/envs/thesis/bin/activate

export ON_CLUSTER=1
cd $SLURM_SUBMIT_DIR  # must be master_thesis root
mkdir -p calibration/logs

echo "Job started: $(date)"
echo "Node: $SLURMD_NODENAME  CPUs: $SLURM_CPUS_PER_TASK"

# Uncomment one phase per job submission.

# python -u calibration/run_pipeline.py extract
# python -u calibration/run_pipeline.py hm $SLURM_CPUS_PER_TASK
# python -u calibration/run_pipeline.py bo $SLURM_CPUS_PER_TASK
# python -u calibration/run_pipeline.py mcmc
# python -u calibration/run_pipeline.py validate $SLURM_CPUS_PER_TASK

# Or run the full pipeline in one job:
python -u calibration/run_pipeline.py all $SLURM_CPUS_PER_TASK

echo "Job finished: $(date)"