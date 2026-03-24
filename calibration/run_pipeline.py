"""
Master script: runs all calibration phases in order.

Usage:
  python run_pipeline.py                   # Full pipeline
  python run_pipeline.py extract           # Data extraction only
  python run_pipeline.py hm       [n_jobs] # History matching
  python run_pipeline.py bo       [n_jobs] # Bayesian optimisation
  python run_pipeline.py mcmc              # MCMC on GP emulator
  python run_pipeline.py validate [n_jobs] # Validation

Phases must be run in order. Each phase saves its outputs so subsequent
phases can be run independently.
"""

import sys
import time
import os

from calibration_config import CALIB_DIR, HM_DIR, BO_DIR, MCMC_DIR, VAL_DIR

for d in [CALIB_DIR, HM_DIR, BO_DIR, MCMC_DIR, VAL_DIR]:
    os.makedirs(d, exist_ok=True)


def run_extract():
    print("PHASE 0: Data extraction (1988-2024)")
    import data_extraction_all_years  # Runs on import


def run_hm(n_jobs=-1):
    print("PHASE 1: History matching")
    import history_matching
    history_matching.N_JOBS = n_jobs
    from calibration_utils import load_data
    data = load_data()
    history_matching.run_history_matching(data)


def run_bo(n_jobs=-1):
    print("PHASE 2: Bayesian optimisation + GP emulator")
    import bo_emulator
    bo_emulator.N_JOBS = n_jobs
    from calibration_utils import load_data
    data = load_data()
    results = bo_emulator.run_bo(data)
    bo_emulator.validate_gp(results["gp"], results["theta_all"], results["loss_all"], results["bounds"])


def run_mcmc():
    print("PHASE 3: MCMC on GP emulator")
    from mcmc_posterior import run_mcmc as _run_mcmc
    _run_mcmc()


def run_validate(n_jobs=-1):
    print("PHASE 4: Validation")
    import validation
    validation.N_JOBS = n_jobs
    from calibration_utils import load_data
    os.makedirs(VAL_DIR, exist_ok=True)
    data = load_data()
    validation.posterior_predictive_check(data)
    validation.out_of_sample_validation(data)
    validation.structural_stability_check(data)
    validation.sobol_on_gp()
    validation.plot_posterior_corner()
    print(f"\nAll outputs saved to: {VAL_DIR}/")


if __name__ == "__main__":
    phase = sys.argv[1] if len(sys.argv) > 1 else "all"
    n_jobs = int(sys.argv[2]) if len(sys.argv) > 2 else -1

    t0 = time.time()

    if phase == "all":
        run_extract()
        run_hm(n_jobs)
        run_bo(n_jobs)
        run_mcmc()
        run_validate(n_jobs)
    elif phase == "extract":
        run_extract()
    elif phase == "hm":
        run_hm(n_jobs)
    elif phase == "bo":
        run_bo(n_jobs)
    elif phase == "mcmc":
        run_mcmc()
    elif phase == "validate":
        run_validate(n_jobs)
    else:
        print(f"Unknown phase: {phase}")
        sys.exit(1)

    print(f"\nTotal time: {(time.time() - t0)/60:.1f} min")