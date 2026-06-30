"""
Stage 4 prep: assemble the canonical joint parameter pack from Stage 2 + Stage 3.

Schema:
  globals (JSON):    GLOBAL_PARAM_NAMES from shared_global_calibration
                                         (kappa, sigma, h_P, s_pi_P, a_C, b_C, a_P, b_P,
                                            beta_declining_delta, beta_rising_delta, beta_stable_delta)
  per-country (NPZ): s_pi, G, nu, h_C, entry_threshold, beta_trade_off
                                         + beta_trade_off_corrected (derived with grouped deltas)
"""
import json
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from calibration.calibration_config import CALIB_DIR, FIXED
from calibration.country_wise_calibration.shared_global_calibration import (
    GLOBAL_PARAM_NAMES, GLOBAL_PARAMS_PATH, load_country_vectors,
    apply_beta_group_correction,
)
from calibration.country_wise_calibration.calibration_country_wise import (
    load_country_index,
)

OUTPUT_DIR = os.path.join(CALIB_DIR, "joint")
JOINT_JSON = os.path.join(OUTPUT_DIR, "joint_params.json")
JOINT_NPZ = os.path.join(OUTPUT_DIR, "joint_params.npz")


def build_joint_param_pack():
    if not os.path.exists(GLOBAL_PARAMS_PATH):
        raise FileNotFoundError(f"Missing Stage 3 output: {GLOBAL_PARAMS_PATH}")
    with open(GLOBAL_PARAMS_PATH) as f:
        globals_raw = json.load(f)
    missing_globals = [k for k in GLOBAL_PARAM_NAMES if k not in globals_raw]
    if missing_globals:
        raise KeyError(
            f"Stage 3 output {GLOBAL_PARAMS_PATH} is missing required "
            f"global keys {missing_globals}. The pack is stale relative to "
            f"the current GLOBAL_PARAM_NAMES = {GLOBAL_PARAM_NAMES}. "
            "Re-run Stage 3."
        )
    globals_pack = {k: globals_raw[k] for k in GLOBAL_PARAM_NAMES}

    country_vecs = load_country_vectors()
    beta_corr = apply_beta_group_correction(country_vecs["beta_trade_off"], globals_pack)
    rows = load_country_index()

    country_codes = [r["location_code"] for r in rows]

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    np.savez(
        JOINT_NPZ,
        s_pi=country_vecs["s_pi"],
        G=country_vecs["G"], nu=country_vecs["nu"],
        h_C=country_vecs["h_C"], entry_threshold=country_vecs["entry_threshold"],
        beta_trade_off=country_vecs["beta_trade_off"],
        beta_trade_off_corrected=beta_corr,
    )
    pack_meta = {
        "globals": globals_pack,
        "fixed": {k: float(v) if isinstance(v, (int, float)) else v
                  for k, v in FIXED.items()},
        "country_codes": country_codes,
        "schema_version": 4,  # grouped beta_trade_off correction in Stage 3 + corrected vector stored in NPZ
    }
    with open(JOINT_JSON, "w") as f:
        json.dump(pack_meta, f, indent=2)
    print(f"Wrote {JOINT_JSON} and {JOINT_NPZ}")


if __name__ == "__main__":
    build_joint_param_pack()
