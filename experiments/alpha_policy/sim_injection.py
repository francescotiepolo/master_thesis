"""
Alpha-injection conditioned simulation.

The target country's alpha row is deterministically pinned each year to a
counterfactual alpha row built from observed alpha + a basket transfer.
All other countries' C and alpha are pinned to observed values at every year
boundary. P is reconstructed from the conditioning identity so that observed
product totals are preserved at year boundaries.

When strength=0 this collapses to the alpha-frozen mode used in calibration:
sim alpha = observed alpha for every country. That mode's C trajectories are
already validated by the first-stage calibration, so it is a credible
baseline for counterfactual comparison.
"""
import os
import sys
import time
import warnings

import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(THIS_DIR))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "calibration"))
sys.path.insert(0, os.path.join(ROOT, "calibration", "country_wise_calibration"))

from calibration_config import (
    FIXED,
    MAX_SOLVER_STEPS,
    SIM_STEPS_PER_YEAR,
    SOLVE_TIMEOUT_S,
    TRAJECTORY_TIMEOUT_S,
    YEAR_START,
)
from calibration_utils import _SolveTimeout, _update_growth_rates, _wall_clock_timeout

from alpha_injection import inject


def simulate_injection_conditioned(
    model,
    data,
    country_idx: int,
    years,
    basket,
    strength: float,
    start_year: int = None,
    max_time: float = TRAJECTORY_TIMEOUT_S,
    solve_timeout_s: float = SOLVE_TIMEOUT_S,
):
    """
    Returns {year: {"P": (SP,), "C": (SC,), "alpha": (SC, SP)}} or None on failure.

    Each annual step runs the full calibrated ODE, matching the alpha-frozen
    calibration simulator. At each year boundary:
      - target country's alpha row <- inject(alpha_obs[year, target], basket,
                                            strength)
      - all other countries' alpha rows <- alpha_obs[year]
      - target country's C <- ODE-evolved C
      - all other countries' C <- C_obs[year]
      - P reconstructed via the conditioning identity (observed P plus the
        sim-vs-obs delta for the target country's exports).
    """
    if start_year is None:
        start_year = YEAR_START
    sorted_years = sorted(set(years))

    P_obs_start = data["P_obs"][start_year].astype(float).copy()
    C = data["C_obs"][start_year].astype(float).copy()
    alpha_obs_start = data["alpha_obs"][start_year].astype(float).copy()
    alpha = alpha_obs_start.copy()

    # Apply injection at the start year too so the trajectory is internally
    # consistent (sim alpha at t=0 equals what the simulator will see). Rebuild
    # P by replacing the observed target exports with injected target exports.
    alpha[country_idx] = inject(alpha[country_idx], basket, strength)
    exports_obs_target = C[country_idx] * alpha_obs_start[country_idx]
    exports_sim_target = C[country_idx] * alpha[country_idx]
    P = np.clip(P_obs_start - exports_obs_target + exports_sim_target, 1e-12, None)
    y0 = np.concatenate([P, C, alpha.flatten()])
    out = {start_year: {"P": P.copy(), "C": C.copy(), "alpha": alpha.copy()}}
    prev_year = start_year

    t0 = time.time()
    try:
        with _wall_clock_timeout(max_time):
            for year in sorted_years:
                if year == start_year:
                    continue
                if max_time and time.time() - t0 > max_time:
                    return None
                _update_growth_rates(model, data, year)
                n_yr = year - prev_year
                if n_yr <= 0:
                    continue

                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        with _wall_clock_timeout(solve_timeout_s):
                            model.solve(
                                t_end=float(n_yr),
                                d_C=float(FIXED.get("d_C", 0.0)),
                                n_steps=SIM_STEPS_PER_YEAR * n_yr,
                                y0=y0,
                                max_solver_steps=MAX_SOLVER_STEPS,
                                rtol=1e-3,
                                atol=1e-6,
                            )
                except _SolveTimeout:
                    return None
                except Exception:
                    return None

                if model.y is None or model.y.shape[1] == 0:
                    return None
                if model.y_partial is None or model.y_partial.shape[1] == 0:
                    return None

                P_sim = np.maximum(model.y[: model.SP, -1], 0.0)
                C_sim = np.maximum(model.y[model.SP : model.N, -1], 0.0)
                if (
                    not np.all(np.isfinite(P_sim))
                    or not np.all(np.isfinite(C_sim))
                    or np.max(P_sim) > 1e6
                    or np.max(C_sim) > 1e6
                ):
                    return None

                # Build the conditioned state at this year boundary.
                C = data["C_obs"][year].astype(float).copy()
                alpha = data["alpha_obs"][year].astype(float).copy()
                C[country_idx] = C_sim[country_idx]
                alpha[country_idx] = inject(alpha[country_idx], basket, strength)

                # Conditioning identity for P: replace observed target exports
                # with simulated target exports (using the injected alpha row).
                exports_obs_target = (
                    data["C_obs"][year][country_idx]
                    * data["alpha_obs"][year][country_idx]
                )
                exports_sim_target = C[country_idx] * alpha[country_idx]
                P = (
                    data["P_obs"][year].astype(float).copy()
                    - exports_obs_target
                    + exports_sim_target
                )
                P = np.clip(P, 1e-12, None)

                model.activate_new_links(P, C, alpha)

                out[year] = {"P": P.copy(), "C": C.copy(), "alpha": alpha.copy()}
                y0 = np.concatenate([P, C, alpha.flatten()])
                prev_year = year
    except _SolveTimeout:
        return None
    return out
