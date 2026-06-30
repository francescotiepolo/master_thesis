"""Tiny smoke run of Stage 3."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
import calibration.country_wise_calibration.shared_global_calibration as sg

os.environ["STAGE3_COUNTRY_SUMMARY"] = "calibration/calibration_results/country_wise/summary.csv"
sg.DE_MAXITER = 2
sg.DE_N_CANDIDATES = 5
sg.STAGE3_YEARS = [y for y in sg.COUNTRY_CALIB_YEARS if y <= 1992]

if __name__ == "__main__":
    sg.main()
    print("ok shared-global smoke")
