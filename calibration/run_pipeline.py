"""
Master script: runs the calibration phases in order.

Usage:
  python run_pipeline.py                   # Phases 1-3 (LHS -> NSGA-II -> validation)
  python run_pipeline.py extract           # Data extraction only
  python run_pipeline.py lhs      [n_jobs] # Initial design (LHS)
  python run_pipeline.py nsga2    [n_jobs] # NSGA-II multi-objective optimizer
  python run_pipeline.py validate [n_jobs] # Validation

Phase 0 extraction is explicit and not included in the default "all" run.
Each phase saves its outputs so subsequent phases can be run independently.
"""

import sys
import time
import os

try:
    from calibration_config import CALIB_DIR, LHS_DIR, NSGA2_DIR, VAL_DIR, MAX_PARALLEL_JOBS
except ModuleNotFoundError as exc:
    if exc.name != "calibration_config":
        raise
    from .calibration_config import CALIB_DIR, LHS_DIR, NSGA2_DIR, VAL_DIR, MAX_PARALLEL_JOBS

for d in [CALIB_DIR, LHS_DIR, NSGA2_DIR, VAL_DIR]:
    os.makedirs(d, exist_ok=True)


def run_extract():
    """
    Phase 0: Data extraction for all years (1988-2024).
    """
    print("PHASE 0: Data extraction (1988-2024)")
    try:
        import data_extraction_all_years
    except ModuleNotFoundError as exc:
        if exc.name != "data_extraction_all_years":
            raise
        from . import data_extraction_all_years


def run_lhs(n_jobs=-1):
    """
    Phase 1: Initial design (LHS).
    """
    print("PHASE 1: Initial design (LHS)")
    try:
        import initial_design
        from calibration_utils import load_data
    except ModuleNotFoundError as exc:
        if exc.name not in {"initial_design", "calibration_utils"}:
            raise
        from . import initial_design
        from .calibration_utils import load_data
    initial_design.N_JOBS = n_jobs
    data = load_data()
    initial_design.run_initial_design(data)


def run_nsga2(n_jobs=-1):
    """
    Phase 2: NSGA-II multi-objective optimizer.
    """
    print("PHASE 2: NSGA-II multi-objective optimizer")
    try:
        import nsga2_optimizer
        from calibration_utils import load_data
    except ModuleNotFoundError as exc:
        if exc.name not in {"nsga2_optimizer", "calibration_utils"}:
            raise
        from . import nsga2_optimizer
        from .calibration_utils import load_data
    data = load_data()
    nsga2_optimizer.run_nsga2(data, n_jobs=n_jobs)


def run_validate(n_jobs=-1):
    """
    Phase 3: Validation of the NSGA-II compromise point.
    """
    print("PHASE 3: Validation")
    try:
        import validation
        from calibration_utils import load_data
    except ModuleNotFoundError as exc:
        if exc.name not in {"validation", "calibration_utils"}:
            raise
        from . import validation
        from .calibration_utils import load_data
    os.makedirs(VAL_DIR, exist_ok=True)
    data = load_data()
    validation.evaluate_best_point(data)
    print(f"\nAll outputs saved to: {VAL_DIR}/")


if __name__ == "__main__":
    phase = sys.argv[1] if len(sys.argv) > 1 else "all"
    n_jobs = int(sys.argv[2]) if len(sys.argv) > 2 else -1
    if n_jobs == -1 or n_jobs > MAX_PARALLEL_JOBS:
        n_jobs = MAX_PARALLEL_JOBS

    t0 = time.time()

    if phase == "all":
        run_lhs(n_jobs)
        run_nsga2(n_jobs)
        run_validate(n_jobs)
    elif phase == "extract":
        run_extract()
    elif phase == "lhs":
        run_lhs(n_jobs)
    elif phase == "nsga2":
        run_nsga2(n_jobs)
    elif phase == "validate":
        run_validate(n_jobs)
    else:
        print(f"Unknown phase: {phase}")
        sys.exit(1)

    print(f"\nTotal time: {(time.time() - t0)/60:.1f} min")