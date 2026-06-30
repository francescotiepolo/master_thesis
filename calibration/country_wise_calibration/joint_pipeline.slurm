#!/bin/bash
# =============================================================================
# Master pipeline: joint all-country calibration
#
# Submit from the master_thesis root:
#   sbatch calibration/country_wise_calibration/joint_pipeline.slurm
#
# Monitor with a single tail command:
#   tail -f calibration/logs/joint_pipeline_latest.log
#
# Stage order (active = uncommented run_stage call):
#   0   regression       — country + product growth regressions      [SKIPPED]
#   2b  bootstrap_seed   — build bootstrap pack used to seed Stage 3 [SKIPPED]
#   2c  country_wise     — optional fresh per-country calibration    (gated by RUN_COUNTRY_WISE_STAGE=1)
#   3   shared_globals   — global calibration using the independent per-country vectors
#   4p  joint_params     — assemble canonical parameter pack
#   4s  joint_sim        — free all-country simulation + losses + plots
#
# This script is the end-to-end entrypoint for the joint pipeline. By default it
# reuses the existing independent country-wise summary in
# calibration_results/country_wise/summary.csv and does not rerun the expensive
# per-country calibration unless RUN_COUNTRY_WISE_STAGE=1 is set. Stages 0/2b
# are commented out because the calibration artefacts they produce on disk are
# still current; only Stages 3, 4p, 4s need to re-run after Stage 3 bounds /
# loss changes.
#
# To skip a stage that is already done: comment out the corresponding
# run_stage line.  Everything else (headers, env-var blocks) can stay.
# =============================================================================

#SBATCH --job-name=joint_pipeline
#SBATCH --partition=rome
#SBATCH --account=gisr122457
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=112G
# NOTE: bump back to --cpus-per-task=128 --mem=224G if RUN_COUNTRY_WISE_STAGE=1
# (Stage 2c uses CW_COUNTRIES_IN_PARALLEL=4 × 32 inner workers = 128 CPUs).
# Memory must stay ≤ cpus * 1792 MiB to avoid being charged for extra 1/8-node units.
#SBATCH --time=48:00:00
#SBATCH --output=calibration/logs/joint_pipeline_%j.log
#SBATCH --error=calibration/logs/joint_pipeline_%j.log

set -euo pipefail

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
module purge
module load 2023
module load Python/3.11.3-GCCcore-12.3.0
source "$HOME/envs/thesis/bin/activate"

export ON_CLUSTER=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib_${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR"
mkdir -p calibration/logs

# Single-file log symlink so "tail -f calibration/logs/joint_pipeline_latest.log" always works
ln -sf "joint_pipeline_${SLURM_JOB_ID}.log" calibration/logs/joint_pipeline_latest.log

# ---------------------------------------------------------------------------
# Tunable parameters (edit here, not below)
# ---------------------------------------------------------------------------

# Regression (Stage 0) settings
export REGRESSION_N_JOBS="${SLURM_CPUS_PER_TASK}"

# Country-wise (Stage 2c) settings — env vars passed to calibration_country_wise.py
export COUNTRY_N_JOBS="${SLURM_CPUS_PER_TASK}"
export COUNTRIES_IN_PARALLEL=4
export COUNTRY_LHS_SAMPLES=350
export COUNTRY_CODES="${COUNTRY_CODES:-}"
export RUN_COUNTRY_WISE_STAGE="${RUN_COUNTRY_WISE_STAGE:-0}"

# Shared-global (Stage 3) settings — env vars read by shared_global_calibration.py
export DE_WORKERS=64
export DE_N_CANDIDATES=64
export DE_MAXITER=200
export DE_TOL=0.005
# Per-candidate wall-clock kill. Stage 3 now evaluates a mixed objective
# (STAGE3_W_WINDOW * windowed + STAGE3_W_FREE * fully-free) so each candidate
# runs ~2 simulations; if many candidates time out at 300s, bump to 600.
export STAGE3_TIMEOUT_S=300
export STAGE3_WINDOWED=1
# Mixed-loss weights. STAGE3_W_FREE > 0 forces DE to also score candidates on
# the fully-free 26-year rollout, blocking the "uniform decay between resets"
# corner the previous pure-windowed run found. Set STAGE3_W_FREE=0 to revert.
export STAGE3_W_WINDOW=0.7
export STAGE3_W_FREE=0.3
# Grouped correction on country beta_trade_off (declining/rising/stable) with
# L2 prior regularization toward the independent country-wise vector.
export STAGE3_BETA_REG_LAMBDA=0.1

# Stage 4 validation mode: reseed (default) uses observed 2014 state for stability;
# continuous uses simulated 2014 state (requires recalibration of Stage 3 to be robust).
export JOINT_VALIDATION_MODE="reseed"

# Country vectors fed into Stage 3.
# By default this points at the existing independent run.
export STAGE3_COUNTRY_SUMMARY="calibration/calibration_results/country_wise/summary.csv"

# ---------------------------------------------------------------------------
# Helper: run_stage <label> <description> <command...>
# To skip a stage, comment out its run_stage call.
# ---------------------------------------------------------------------------
_PIPELINE_T0=$(date +%s)

run_stage() {
    local label="$1"
    local desc="$2"
    shift 2
    local t0
    t0=$(date +%s)
    echo ""
    echo "╔══════════════════════════════════════════════════════════════╗"
    printf  "║  STAGE %-54s║\n" "$label: $desc"
    echo "╠══════════════════════════════════════════════════════════════╣"
    echo "║  Started : $(date)$(printf '%*s' $((31 - ${#desc})) '')║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo ""
    "$@"
    local exit_code=$?
    local elapsed=$(( $(date +%s) - t0 ))
    local total=$(( $(date +%s) - _PIPELINE_T0 ))
    echo ""
    echo "── Stage $label done in $(( elapsed/3600 ))h$(( (elapsed%3600)/60 ))m$(( elapsed%60 ))s"\
         " | pipeline total $(( total/3600 ))h$(( (total%3600)/60 ))m$(( total%60 ))s ──"
    return $exit_code
}

# ---------------------------------------------------------------------------
echo "Job $SLURM_JOB_ID started: $(date)"
echo "Node: $SLURMD_NODENAME  CPUs: $SLURM_CPUS_PER_TASK  Mem: ${SLURM_MEM_PER_NODE}MB"

python - <<'PY'
import importlib.util
for m in ["numpy", "scipy", "numba", "matplotlib", "sklearn", "joblib", "pandas"]:
    if importlib.util.find_spec(m) is None:
        raise SystemExit(f"Missing Python package: {m}")
print("Preflight checks passed.")
PY

# =============================================================================
# STAGE 0 — Regression refresh
# Regenerates country/product growth regression artefacts used by both the
# country-wise and joint stages (r_values_fixed.csv, pi_matrix_fixed.npy,
# pi_competition_P.npy, r_P_regression.npy).
#
# SKIPPED by default: the regression artefacts on disk are still current and
# do not depend on the calibration bounds / loss-objective changes. Re-enable
# (uncomment) if the underlying trade data changes or the regression code
# changes.
# =============================================================================
# run_stage "0" "Refresh growth regressions" \
#     python -u calibration/growth_regression.py
#
# run_stage "0p" "Refresh product growth regression" \
#     python -u calibration/product_growth_regression.py


# =============================================================================
# STAGE 2b — Bootstrap pack for Stage 3
# Builds the shared seed pack used to initialise Stage 3 differential evolution.
# With the default pipeline this uses the existing independent country-wise
# summary for kappa/sigma medians and the regressions for a_C/b_C/a_P/b_P.
#
# SKIPPED by default: the existing bootstrap_globals.json on disk has all 8
# keys (kappa, sigma, h_P, s_pi_P, a_C, b_C, a_P, b_P) and the seed values are
# inside the new tightened bounds (see shared_global_calibration.py:73-89).
# Re-enable (uncomment) if bootstrap_globals.py output changes.
# =============================================================================
# run_stage "2b" "Bootstrap pack for country-wise stage" \
#     python -u calibration/country_wise_calibration/bootstrap_globals.py


# =============================================================================
# STAGE 2c — Optional independent country-wise calibration refresh
# Set RUN_COUNTRY_WISE_STAGE=1 to regenerate calibration_results/country_wise/
# summary.csv. The default pipeline reuses the existing independent run.
# =============================================================================
if [ "$RUN_COUNTRY_WISE_STAGE" = "1" ]; then
    COUNTRY_CMD=(
        python -u calibration/country_wise_calibration/calibration_country_wise.py
        --n-jobs "$COUNTRY_N_JOBS"
        --lhs-samples "$COUNTRY_LHS_SAMPLES"
        --countries-in-parallel "$COUNTRIES_IN_PARALLEL"
    )
    if [ -n "$COUNTRY_CODES" ]; then
        read -r -a COUNTRY_CODE_ARRAY <<< "$COUNTRY_CODES"
        COUNTRY_CMD+=(--countries "${COUNTRY_CODE_ARRAY[@]}")
    fi
    run_stage "2c" "Independent country-wise calibration" "${COUNTRY_CMD[@]}"
    run_stage "2r" "Refresh bootstrap pack for Stage 3" \
        python -u calibration/country_wise_calibration/bootstrap_globals.py
else
    echo "Skipping Stage 2c: reusing existing country-wise summary at $STAGE3_COUNTRY_SUMMARY"
fi


# =============================================================================
# STAGE 3 — Shared global calibration
# Optimises globals (kappa, sigma, h_P, s_pi_P, a_C, b_C, a_P, b_P) plus
# grouped beta_trade_off corrections (declining/rising/stable) using the
# independent country-wise summary plus the refreshed growth regressions.
# =============================================================================
# run_stage "3" "Shared global calibration" \
#     python -u calibration/country_wise_calibration/shared_global_calibration.py


# =============================================================================
# STAGE 4p — Assemble joint parameter pack
# Combines Stage-3 globals + independent per-country vectors into
# joint_params.json/npz.
# =============================================================================
# run_stage "4p" "Assemble joint parameter pack" \
#     python -u calibration/country_wise_calibration/joint_params.py


# =============================================================================
# STAGE 4s — Joint free all-country simulation
# Runs the free simulation over calibration and validation windows and writes
# loss_components.json + trajectory plots.
# =============================================================================
run_stage "4s" "Joint free simulation + losses" \
    python -u calibration/country_wise_calibration/joint_simulation.py


# ---------------------------------------------------------------------------
total=$(( $(date +%s) - _PIPELINE_T0 ))
echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  Pipeline complete: $(date)"
printf "  Total wall time: %dh%02dm%02ds\n" \
    $(( total/3600 )) $(( (total%3600)/60 )) $(( total%60 ))
echo "  Loss summary:"
python - <<'PY'
import json
try:
    with open("calibration/calibration_results/joint/loss_components.json") as f:
        d = json.load(f)
    for period, vals in d.items():
        print(f"    {period}: loss={vals['loss_total']:.4f}  "
              f"nrmse_C={vals['nrmse_C']:.4f}  "
              f"nrmse_P={vals['nrmse_P']:.4f}  "
              f"rank_products={vals['rank_products']:.4f}")
except Exception as e:
    print(f"    (could not read loss_components.json: {e})")
PY
echo "════════════════════════════════════════════════════════════════"
