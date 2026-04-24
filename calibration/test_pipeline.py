"""
Quick test for the calibration pipeline.
Overrides config values to use small settings.
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import calibration_config as cfg

# Override with small settings
cfg.LHS_N_SAMPLES = 8
cfg.TRAJECTORY_TIMEOUT_S = 60
cfg.NSGA2_POP_SIZE = 2
cfg.NSGA2_N_GEN = 1
cfg.SIM_STEPS_PER_YEAR = 300
cfg.CALIB_YEARS = [1988, 1989, 1990]
cfg.VALID_YEARS = [2010, 2011, 2012]
# Use a separate output directory
TEST_DIR = os.path.join(cfg.CALIB_DIR, "_test")
cfg.LHS_DIR = os.path.join(TEST_DIR, "initial_design")
cfg.NSGA2_DIR = os.path.join(TEST_DIR, "nsga2_optimizer")
cfg.VAL_DIR = os.path.join(TEST_DIR, "validation")

for d in [cfg.LHS_DIR, cfg.NSGA2_DIR, cfg.VAL_DIR]:
    os.makedirs(d, exist_ok=True)

# Also overwrite calibration_utils since it already imported config values
import calibration_utils as cu
cu.LHS_DIR = cfg.LHS_DIR
cu.NSGA2_DIR = cfg.NSGA2_DIR
cu.TRAJECTORY_TIMEOUT_S = cfg.TRAJECTORY_TIMEOUT_S
cu.SIM_STEPS_PER_YEAR = cfg.SIM_STEPS_PER_YEAR
cu.CALIB_YEARS = cfg.CALIB_YEARS

from run_pipeline import run_lhs, run_nsga2, run_validate

# Keep local tests stable
N_JOBS = 1

phases = sys.argv[1:] if len(sys.argv) > 1 else ["lhs", "nsga2", "validate"]

t0 = time.time()
print(f"Test workers: {N_JOBS}")
for phase in phases:
    print(f"\n{'='*60}")
    print(f"TEST: {phase}")
    print(f"{'='*60}")
    if phase == "lhs":
        run_lhs(N_JOBS)
    elif phase == "nsga2":
        run_nsga2(N_JOBS)
    elif phase == "validate":
        run_validate(N_JOBS)
    else:
        print(f"Unknown phase: {phase}")

elapsed = time.time() - t0
print(f"\nTest completed in {elapsed/60:.1f} min")
print(f"Test outputs in: {TEST_DIR}")