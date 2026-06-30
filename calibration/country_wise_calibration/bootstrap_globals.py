"""
Build the seed for Stage 3 DE. Sources: kappa/sigma from cross-country
median of country_wise/summary.csv; h_P from NSGA-II point; s_pi/s_pi_P fixed
defaults; a_C/b_C seeded to reproduce raw regression r_C; a_P/b_P seeded to
reproduce r_P_regression (fallback (0,1) if missing). beta_trade_off is
per-country and excluded from this global pack.
"""
import csv
import json
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from calibration.calibration_config import CALIB_DIR  # noqa

NSGA2_LOG_PATH = os.path.join(CALIB_DIR, "nsga2_optimizer", "nsga2_log.json")
LEGACY_SUMMARY_PATH = os.path.join(CALIB_DIR, "country_wise", "summary.csv")
OUTPUT_DIR = os.path.join(CALIB_DIR, "joint")
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "bootstrap_globals.json")
R_VALUES_PATH = os.path.join(CALIB_DIR, "growth_regression", "r_values_fixed.csv")

# Params shared between per-country run and Stage 3 globals.
_SHARED_PARAMS = ("kappa", "sigma")


def _medians_from_legacy_summary():
    """
    Read the country-wise summary and return cross-country medians for
    the shared global parameters.  Returns None if the file is absent or has no
    usable rows.
    """
    if not os.path.exists(LEGACY_SUMMARY_PATH):
        return None
    values = {p: [] for p in _SHARED_PARAMS}
    with open(LEGACY_SUMMARY_PATH, newline="") as f:
        for row in csv.DictReader(f):
            for p in _SHARED_PARAMS:
                try:
                    values[p].append(float(row[p]))
                except (KeyError, ValueError):
                    pass
    if not all(values[p] for p in _SHARED_PARAMS):
        return None
    return {p: float(np.median(values[p])) for p in _SHARED_PARAMS}


def build_bootstrap_pack():
    if not os.path.exists(NSGA2_LOG_PATH):
        raise FileNotFoundError(
            f"Missing NSGA-II log: {NSGA2_LOG_PATH}. "
            "Run the NSGA-II optimizer first."
        )

    with open(NSGA2_LOG_PATH) as f:
        log = json.load(f)

    theta = log["compromise_theta"]

    # h_P has no per-country analogue — use the NSGA-II value.
    # a_C / b_C and a_P / b_P are global affine-map parameters:
    #   model.r_C = a_C + b_C * centered r_C_regression
    #   model.r_P = a_P + b_P * centered r_P_regression
    # Seed at the regression means with unit slopes so the bootstrap point
    # reproduces the raw regressions exactly.
    a_C_seed, b_C_seed = 0.0, 1.0
    if os.path.exists(R_VALUES_PATH):
        vals = []
        with open(R_VALUES_PATH, newline="") as f:
            for row in csv.DictReader(f):
                try:
                    vals.append(float(row["r_i"]))
                except (KeyError, ValueError):
                    pass
        if vals:
            a_C_seed = float(np.mean(vals))
    # a_P / b_P are new global affine-map parameters (model.r_P = a_P + b_P *
    # centered r_P_regression). Seed at (mean(r_P_regression), 1.0) when the
    # regression file is available — that exactly reproduces the raw regression
    # r_P, which is the regime previous Stage 3 runs operated under and gives
    # DE a known starting point. Falls back to (0.0, 1.0) if the file is missing.
    a_P_seed, b_P_seed = 0.0, 1.0
    r_P_reg_path = os.path.join(
        CALIB_DIR, "growth_regression", "r_P_regression.npy"
    )
    if os.path.exists(r_P_reg_path):
        r_P_reg = np.load(r_P_reg_path).astype(float)
        a_P_seed = float(r_P_reg.mean())
    pack = {
        "kappa":          float(theta["kappa"]),
        "sigma":          float(theta["sigma"]),
        "h_P":            float(theta["h_mean"]),
        "s_pi":           0.05,
        "s_pi_P":         1.0,
        "a_C":            a_C_seed,
        "b_C":            b_C_seed,
        "a_P":            a_P_seed,
        "b_P":            b_P_seed,
    }

    provenance_source = NSGA2_LOG_PATH
    medians = _medians_from_legacy_summary()
    if medians is not None:
        for p in _SHARED_PARAMS:
            pack[p] = medians[p]
        provenance_source = LEGACY_SUMMARY_PATH
        print(f"Using cross-country medians from {LEGACY_SUMMARY_PATH} for "
              f"{', '.join(_SHARED_PARAMS)}")
    else:
        print(f"Legacy summary not found; falling back to NSGA-II compromise point.")

    pack["_provenance"] = {
        "source": provenance_source,
        "compromise_objectives": log.get("compromise_objectives"),
    }

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(pack, f, indent=2)
    print(f"Wrote {OUTPUT_PATH}")
    for k, v in pack.items():
        if not k.startswith("_"):
            print(f"  {k}: {v}")
    return pack


if __name__ == "__main__":
    build_bootstrap_pack()
