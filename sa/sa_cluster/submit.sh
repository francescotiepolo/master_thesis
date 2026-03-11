#!/bin/bash
# Full Sobol SA pipeline on Snellius.
#
# This script submits four jobs:
#   1. generate: creates Sobol sample .npy files (runs once, fast)
#   2. array_base: SLURM array over BaseModel samples
#   3. array_ps: SLURM array over ProductSpaceModel samples
#   4. collect: assembles results, runs Sobol analysis, saves plots/CSVs
#
# Jobs 2 & 3 wait for job 1. Job 4 waits for jobs 2 & 3.

# Configuration
PARTITION="rome"
ACCOUNT="cpuucl002"
TIME_GENERATE="01:00:00"
TIME_ARRAY="10:00:00"
TIME_COLLECT="01:00:00"
MEM="4G"
PYTHON_MODULE="Python/3.11.3-GCCcore-12.3.0"
VENV="$HOME/envs/thesis"
CHUNK_SIZE=20  # Number of samples per array job

# Directories
CLUSTER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)" # sa/sa_cluster/
SA_DIR="$(dirname "$CLUSTER_DIR")" # sa/
cd "$SA_DIR"
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null
find . -name "*.pyc" -delete 2>/dev/null
echo "Working directory: $(pwd)"

mkdir -p logs
source $VENV/bin/activate
set +e

# Resolve array bounds directly from the SA config
BOUNDS=$(python3 - <<'EOF'
import sys
sys.path.insert(0, ".")
from SALib.sample import saltelli
from sensitivity_analysis import PROBLEM_BASE, PROBLEM_PS, N_SOBOL, SECOND_ORDER
pv_base = saltelli.sample(PROBLEM_BASE, N_SOBOL, calc_second_order=SECOND_ORDER)
pv_ps   = saltelli.sample(PROBLEM_PS,   N_SOBOL, calc_second_order=SECOND_ORDER)
print(f"{len(pv_base)-1} {len(pv_ps)-1}")
EOF
)
ARRAY_BASE=$(echo $BOUNDS | cut -d' ' -f1)
ARRAY_PS=$(echo $BOUNDS | cut -d' ' -f2)
deactivate
echo "Array bounds: BaseModel=0-${ARRAY_BASE}, ProductSpaceModel=0-${ARRAY_PS}"
echo "ARRAY_BASE: $ARRAY_BASE"
echo "CHUNK_SIZE: $CHUNK_SIZE"
echo "Array bound: $(( (ARRAY_BASE + CHUNK_SIZE - 1) / CHUNK_SIZE - 1 ))"

ARRAY_BASE_CHUNK=$(( (ARRAY_BASE + CHUNK_SIZE - 1) / CHUNK_SIZE - 1 ))
ARRAY_PS_CHUNK=$(( (ARRAY_PS + CHUNK_SIZE - 1) / CHUNK_SIZE - 1 ))

CHDIR="$SA_DIR"

# Job 1: generate samples
JOB_GENERATE=$(sbatch --parsable \
    --job-name=sa_generate \
    --ntasks=1 \
    --cpus-per-task=1 \
    --time=$TIME_GENERATE \
    --partition=$PARTITION \
    --account=$ACCOUNT \
    --mem=$MEM \
    --chdir=$CHDIR \
    --output=logs/generate.out \
    --error=logs/generate.err \
    --wrap="
        module load 2023
        module load $PYTHON_MODULE
        ${VENV:+source $VENV/bin/activate}
        python sa_cluster/generate_samples.py
    ")
echo "Submitted generate job: $JOB_GENERATE"

# Job 2: BaseModel array
JOB_BASE=$(sbatch --parsable \
    --job-name=sa_base \
    --array=0-${ARRAY_BASE_CHUNK} \
    --ntasks=1 \
    --cpus-per-task=1 \
    --time=$TIME_ARRAY \
    --partition=$PARTITION \
    --account=$ACCOUNT \
    --mem=$MEM \
    --chdir=$CHDIR \
    --output=logs/base_%A_%a.out \
    --error=logs/base_%A_%a.err \
    --dependency=afterok:$JOB_GENERATE \
    --wrap="
        module load 2023
        module load $PYTHON_MODULE
        ${VENV:+source $VENV/bin/activate}
        python sa_cluster/run_single.py BaseModel \$SLURM_ARRAY_TASK_ID $CHUNK_SIZE
    ")
echo "Submitted BaseModel array job: $JOB_BASE"

# Job 3: ProductSpaceModel array
JOB_PS=$(sbatch --parsable \
    --job-name=sa_ps \
    --array=0-${ARRAY_PS_CHUNK} \
    --ntasks=1 \
    --cpus-per-task=1 \
    --time=$TIME_ARRAY \
    --partition=$PARTITION \
    --account=$ACCOUNT \
    --mem=$MEM \
    --chdir=$CHDIR \
    --output=logs/ps_%A_%a.out \
    --error=logs/ps_%A_%a.err \
    --dependency=afterok:$JOB_GENERATE \
    --wrap="
        module load 2023
        module load $PYTHON_MODULE
        ${VENV:+source $VENV/bin/activate}
        python sa_cluster/run_single.py ProductSpaceModel \$SLURM_ARRAY_TASK_ID $CHUNK_SIZE
    ")
echo "Submitted ProductSpaceModel array job: $JOB_PS"

# Job 4: collect and analyse
JOB_COLLECT=$(sbatch --parsable \
    --job-name=sa_collect \
    --ntasks=1 \
    --cpus-per-task=1 \
    --time=$TIME_COLLECT \
    --partition=$PARTITION \
    --account=$ACCOUNT \
    --mem=$MEM \
    --chdir=$CHDIR \
    --output=logs/collect.out \
    --error=logs/collect.err \
    --dependency=afterok:${JOB_BASE}:${JOB_PS} \
    --wrap="
        module load 2023
        module load $PYTHON_MODULE
        ${VENV:+source $VENV/bin/activate}
        python sa_cluster/collect.py
    ")
echo "Submitted collect job: $JOB_COLLECT"