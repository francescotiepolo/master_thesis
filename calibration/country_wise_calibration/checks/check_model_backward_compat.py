"""
Pin the legacy trajectory_loss for a fixed theta. Re-run this script after every
subsequent task; the printed loss must not change.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import numpy as np
from calibration.calibration_utils import trajectory_loss, load_data
from calibration.calibration_config import CALIB_YEARS

theta = np.array([
    0.10,   # C_offdiag_mean
    0.50,   # beta_trade_off
    1.00,   # sigma
    0.30,   # h_mean
    1.00,   # C_diag_mean
    0.01,   # kappa
    0.50,   # nu
    0.50,   # G
    0.00,   # r_C_declining
    0.05,   # r_C_rising
    0.05,   # r_C_stable
    5.00,   # entry_threshold
])

data = load_data()
loss = trajectory_loss(theta, data, years=CALIB_YEARS[:6])
print(f"BASELINE_LOSS={loss:.10f}")

# Baseline value (recorded 2026-04-26): BASELINE_LOSS=0.6290545469
