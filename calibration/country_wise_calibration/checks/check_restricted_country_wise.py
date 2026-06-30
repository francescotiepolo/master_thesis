"""Run a tiny restricted country-wise calibration on USA only."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import calibration.country_wise_calibration.calibration_country_wise as cw

cw.FAST_LHS_N_SAMPLES = 16
cw.FAST_DE_POPSIZE = 4
cw.FAST_DE_MAXITER = 2

if __name__ == "__main__":
    sys.argv = ["check", "--countries", "USA", "--n-jobs", "2", "--lhs-samples", "16"]
    cw.main()
    print("ok restricted country-wise smoke")
