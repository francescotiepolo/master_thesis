"""
Phase 1: Bayesian history matching with GP emulation.

Runs HM_N_WAVES waves. Each wave:
  1. Latin Hypercube sample in current NROY space
  2. Parallel simulation through the model and loss evaluation (trajectory_loss)
  3. GP emulator fit on all data so far (always normalised to prior bounds
     for consistency — the GP must see the same [0,1]^d space across waves, since it is sensitive to scale)
  4. Rule out I(θ) > HM_THRESHOLD
  5. NROY updated via point-wise filtering (preserves non-convex regions) - the use of
     a bounding box might include some ruled-out points

Reference: Vernon, Goldstein & Bower (2010); Andrianakis et al. (2015).
"""

import os
import json
import time
import numpy as np
from calibration_config import (
    PARAM_NAMES, PARAM_BOUNDS, N_PARAMS,
    HM_N_WAVES, HM_SIMS_PER_WAVE, HM_SCREEN_N, HM_THRESHOLD,
    HM_DIR, SEED, SIGMA_OBS, SIGMA_MODEL,
)
from calibration_utils import (
    PENALTY, fit_botorch_gp, gp_predict, evaluate_batch, lhc_sample
)

N_JOBS = -1


# Helpers

def implausibility(mu, sigma, obs_target=0.0, obs_var=None, model_var=None):
    """
    How many combined std devs the emulator mean is from the observed target,
    taking into account emulator uncertainty, observation noise and model discrepancy.

    I(θ) = |E[f(θ)] - z_obs| / sqrt(Var_emu + Var_obs + Var_model)
    """
    if obs_var is None:
        obs_var = SIGMA_OBS ** 2
    if model_var is None:
        model_var = SIGMA_MODEL ** 2
    total_var = sigma**2 + obs_var + model_var
    total_std = np.sqrt(np.maximum(total_var, 1e-20))
    return np.abs(mu - obs_target) / total_std


def gp_diagnostics(gp, X_raw, y_raw, bt):
    """
    Check how well the GP emulator fits its training data.
    Residuals are computed on the same points the GP was trained on (in-sample),
    then normalised by the GP's own uncertainty estimate.
    A well-calibrated GP should have most normalised residuals within ±2 (since they are standardised
    and GP prediction errors are normally distributed; ~95% of a normal distribution falls within ±2 standard deviations).
    """
    mu, sigma = gp_predict(gp, X_raw, bt)
    residuals = y_raw - mu
    train_rmse = float(np.sqrt(np.mean(residuals**2)))

    # Standardised residuals
    sigma_safe = np.maximum(sigma, 1e-10)
    std_resid = np.abs(residuals) / sigma_safe
    frac_outlier = float(np.mean(std_resid > 2.0))

    return train_rmse, frac_outlier


def nroy_bounding_box(survivors):
    """
    Bounding box of surviving points with 5% padding, clipped to prior bounds.
    Used ONLY for efficient LHC sampling in the next wave — the actual
    NROY is defined by the actual survivors points (no bounds).
    """
    lo = survivors.min(axis=0)
    hi = survivors.max(axis=0)
    width = hi - lo
    lo = lo - 0.05 * width
    hi = hi + 0.05 * width
    for i, (p0, p1) in enumerate(PARAM_BOUNDS):
        lo[i] = max(lo[i], p0)
        hi[i] = min(hi[i], p1)
    return [(lo[i], hi[i]) for i in range(N_PARAMS)]


# History matching

def run_history_matching(data):
    """
    Run HM_N_WAVES waves of history matching.
    Returns dict with nroy_bounds, theta_all, loss_all, survivors, wave_stats.
    """
    os.makedirs(HM_DIR, exist_ok=True)

    theta_all = np.empty((0, N_PARAMS))
    loss_all = np.empty(0)
    current_bounds = list(PARAM_BOUNDS)
    survivors = None
    wave_stats = []

    for wave in range(1, HM_N_WAVES + 1):
        t0 = time.time()
        seed = SEED + wave * 100
        print(f"Wave {wave}/{HM_N_WAVES}")
        for name, (lo, hi) in zip(PARAM_NAMES, current_bounds):
            print(f"  {name:8s}: [{lo:.4f}, {hi:.4f}]")

        # LHC sample in current NROY bounding box
        theta_wave = lhc_sample(current_bounds, HM_SIMS_PER_WAVE, seed)


        # Parallel evaluation of loss function on the sampled points
        # One run = one loss computed on the full trajectory (1988-2010)
        print(f"Evaluating {HM_SIMS_PER_WAVE} simulator runs...")
        loss_wave = evaluate_batch(theta_wave, data, n_jobs=N_JOBS)

        theta_all = np.vstack([theta_all, theta_wave])
        loss_all  = np.concatenate([loss_all, loss_wave])

        # Fit GP on finite (=successful) evaluations, normalised to current wave bounds
        finite = loss_all < PENALTY * 0.1
        if finite.sum() < N_PARAMS + 2:
            print("WARNING: not enough finite evaluations to fit GP.")
            break
        print(f"Fitting GP on {finite.sum()} finite evaluations...")
        gp, bt = fit_botorch_gp(theta_all[finite], loss_all[finite], current_bounds)

        # GP diagnostics
        train_rmse, frac_outlier = gp_diagnostics(
            gp, theta_all[finite], loss_all[finite], bt
        )
        print(f"GP RMSE: {train_rmse:.4f}")
        print(f"GP std residuals > 2: {frac_outlier:.1%} (target < 5%)")
        if frac_outlier > 0.10:
            print("WARNING: GP fit may be poor")

        # New LHC + previous survivors for continuity
        theta_screen = lhc_sample(current_bounds, HM_SCREEN_N, seed + 1)
        if survivors is not None and len(survivors) > 0:
            theta_screen = np.vstack([theta_screen, survivors])

        # Implausibility screening
        mu, sigma = gp_predict(gp, theta_screen, bt)
        impl = implausibility(mu, sigma)
        survivors = theta_screen[impl <= HM_THRESHOLD]
        survival_rate = len(survivors) / len(theta_screen)
        print(f"Survivors: {len(survivors)}/{len(theta_screen)} ({survival_rate:.1%})")

        n_failed = (loss_wave >= PENALTY * 0.1).sum()
        if n_failed:
            print(f"WARNING: {n_failed}/{HM_SIMS_PER_WAVE} runs returned penalty or high loss values")

        wave_stats.append({
            "wave": wave,
            "loss_min": float(loss_all[finite].min()),
            "loss_median": float(np.median(loss_all[finite])),
            "n_finite": int(finite.sum()),
            "n_failed": int(n_failed),
            "survival_rate": float(survival_rate),
            "train_rmse": float(train_rmse),
            "frac_outlier": float(frac_outlier),
            "time_s": float(time.time() - t0),
        })

        if len(survivors) < 10:
            print("WARNING: fewer than 10 survivors -- stopping early.")
            break

        if wave < HM_N_WAVES:
            current_bounds = nroy_bounding_box(survivors)

    # Save
    np.save(os.path.join(HM_DIR, "theta_all.npy"), theta_all)
    np.save(os.path.join(HM_DIR, "loss_all.npy"), loss_all)
    np.save(os.path.join(HM_DIR, "survivors.npy"), survivors)

    # Save evaluated points inside NROY with finite losses (ready for BO to reuse)
    bounds = nroy_bounding_box(survivors)
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])
    finite = loss_all < PENALTY * 0.1
    inside = np.all((theta_all >= lo) & (theta_all <= hi), axis=1)
    nroy_mask = finite & inside
    np.save(os.path.join(HM_DIR, "nroy_theta.npy"), theta_all[nroy_mask])
    np.save(os.path.join(HM_DIR, "nroy_loss.npy"), loss_all[nroy_mask])
    with open(os.path.join(HM_DIR, "nroy_bounds.json"), "w") as f:
        json.dump({"params": PARAM_NAMES, "bounds": bounds}, f, indent=2)
    with open(os.path.join(HM_DIR, "wave_stats.json"), "w") as f:
        json.dump(wave_stats, f, indent=2)

    # Summary
    finite = loss_all < PENALTY * 0.1
    best_loss = loss_all[finite].min()
    best_theta = theta_all[finite][loss_all[finite].argmin()]
    print(f"\nHistory matching complete")
    print(f"  Total runs : {len(theta_all)}")
    print(f"  Finite runs: {finite.sum()}")
    print(f"  Best loss  : {best_loss:.5f}")
    for name, val in zip(PARAM_NAMES, best_theta):
        print(f"  {name:8s} = {val:.5f}")
    print(f"  NROY bounds:")
    for name, (lo, hi) in zip(PARAM_NAMES, bounds):
        prior_lo, prior_hi = PARAM_BOUNDS[PARAM_NAMES.index(name)]
        reduction = 1 - (hi - lo) / (prior_hi - prior_lo)
        print(f"    {name:8s}: [{lo:.4f}, {hi:.4f}]  ({reduction:.0%} reduction)")

    return {
        "nroy_bounds": bounds,
        "theta_all": theta_all,
        "loss_all": loss_all,
        "survivors": survivors,
        "wave_stats" : wave_stats,
    }


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        N_JOBS = int(sys.argv[1])
    from calibration_utils import load_data
    print("Loading empirical data...")
    data = load_data()
    run_history_matching(data)