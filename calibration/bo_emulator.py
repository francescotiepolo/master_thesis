"""
Phase 2: Bayesian optimisation on the NROY space from history_matching.py.

Saves:
  bo_emulator/theta_all.npy   -- all evaluated parameter vectors
  bo_emulator/loss_all.npy    -- corresponding trajectory losses
  bo_emulator/best_theta.npy  -- MAP estimate
  bo_emulator/nroy_bounds.npy -- NROY bounds used
  bo_emulator/gp_state.pth    -- BoTorch GP state dict for MCMC reuse
  bo_emulator/gp_train_X.npy  -- GP training inputs (normalised)
  bo_emulator/gp_train_Y.npy  -- GP training outputs (negated loss)
  bo_emulator/bo_log.json     -- iteration log
"""

import os
import json
import time
import numpy as np


import torch
from botorch.acquisition import qLogExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.utils.transforms import normalize, unnormalize

from calibration_config import (
    PARAM_NAMES, N_PARAMS,
    BO_N_INIT, BO_N_ROUNDS, BO_BATCH_SIZE,
    HM_DIR, BO_DIR, SEED,
)
from calibration_utils import evaluate_batch, load_nroy_bounds, fit_botorch_gp, gp_predict, lhc_sample, PENALTY

torch.set_default_dtype(torch.float64)

N_JOBS = -1


# Bayesian optimisation

def run_bo(data, nroy_bounds=None):
    """
    Run BO on the NROY space and return the final GP emulator.
    """
    os.makedirs(BO_DIR, exist_ok=True)

    if nroy_bounds is None:
        nroy_bounds = load_nroy_bounds()

    unit_bounds = torch.stack([torch.zeros(N_PARAMS), torch.ones(N_PARAMS)])

    # Reuse pre-filtered HM evaluations inside NROY
    nroy_theta_path = os.path.join(HM_DIR, "nroy_theta.npy")
    nroy_loss_path = os.path.join(HM_DIR, "nroy_loss.npy")
    if os.path.exists(nroy_theta_path):
        theta_all = np.load(nroy_theta_path)
        loss_all = np.load(nroy_loss_path)
        print(f"{len(theta_all)} finite HM points inside NROY.")
    else:
        theta_all = np.empty((0, N_PARAMS))
        loss_all = np.empty(0)

    # Fill initial GP training set
    n_init_needed = max(BO_N_INIT - len(theta_all), N_PARAMS)
    print(f"Initial LHC design: {n_init_needed} points...")
    theta_init = lhc_sample(nroy_bounds, n_init_needed, SEED) # Sample in NROY bounds
    loss_init = evaluate_batch(theta_init, data, n_jobs=N_JOBS) # Evaluate on simulator
    theta_all = np.vstack([theta_all, theta_init])
    loss_all = np.concatenate([loss_all, loss_init])

    bo_log = []
    best_loss = np.inf
    best_theta = None

    for rnd in range(1, BO_N_ROUNDS + 1):
        t0 = time.time()
        finite = loss_all < PENALTY * 0.1
        if finite.sum() < N_PARAMS + 2:
            print("Not enough finite evaluations -- stopping.")
            break

        # Fit GP on raw data (fit_botorch_gp normalises internally); negate loss (BoTorch maximises)
        gp, bt = fit_botorch_gp(theta_all[finite], -loss_all[finite], nroy_bounds)
        best_f = torch.tensor(-loss_all[finite].min(), dtype=torch.float64)
        # Optimize acquisition function to get new candidates
        acqf = qLogExpectedImprovement(model=gp, best_f=best_f) # Scores candidate points by how much they're expected to improve over the current best
        cands_unit, _ = optimize_acqf(
            acq_function=acqf, bounds=unit_bounds,
            q=BO_BATCH_SIZE, num_restarts=20, raw_samples=512,
        )
        candidates = unnormalize(cands_unit, bt).detach().numpy() # Map back to original space
        new_losses = evaluate_batch(candidates, data, n_jobs=N_JOBS)

        theta_all = np.vstack([theta_all, candidates])
        loss_all = np.concatenate([loss_all, new_losses])

        cur_finite = loss_all < PENALTY * 0.1
        cur_best_loss = loss_all[cur_finite].min()
        if cur_best_loss < best_loss:
            best_loss = cur_best_loss
            best_theta = theta_all[cur_finite][loss_all[cur_finite].argmin()]

        elapsed = time.time() - t0
        bo_log.append({
            "round": rnd,
            "n_total": int(len(theta_all)),
            "n_finite": int(cur_finite.sum()),
            "best_loss": float(best_loss),
            "time_s": float(elapsed),
        })
        print(f"  Round {rnd:2d}/{BO_N_ROUNDS} | best={best_loss:.5f} | "
              f"evals={len(theta_all)} | {elapsed:.0f}s")

    # Final GP on all finite data — this is the emulator for MCMC
    finite = loss_all < PENALTY * 0.1
    gp_final, bt = fit_botorch_gp(theta_all[finite], -loss_all[finite], nroy_bounds)

    # Reconstruct normalised training data for saving (needed by load_bo_gp)
    X_final = normalize(torch.tensor(theta_all[finite], dtype=torch.float64), bt)
    Y_final = -torch.tensor(loss_all[finite], dtype=torch.float64).unsqueeze(-1)

    # Save everything
    np.save(os.path.join(BO_DIR, "theta_all.npy"), theta_all)
    np.save(os.path.join(BO_DIR, "loss_all.npy"), loss_all)
    np.save(os.path.join(BO_DIR, "best_theta.npy"), best_theta)
    np.save(os.path.join(BO_DIR, "nroy_bounds.npy"), np.array([[b[0], b[1]] for b in nroy_bounds]))
    with open(os.path.join(BO_DIR, "bo_log.json"), "w") as f:
        json.dump(bo_log, f, indent=2)

    # Save GP state for exact reuse in MCMC
    torch.save(gp_final.state_dict(), os.path.join(BO_DIR, "gp_state.pth"))
    np.save(os.path.join(BO_DIR, "gp_train_X.npy"), X_final.numpy())
    np.save(os.path.join(BO_DIR, "gp_train_Y.npy"), Y_final.numpy())

    print(f"\nBO complete")
    print(f"  Total evaluations : {len(theta_all)}")
    print(f"  Finite evaluations: {finite.sum()}")
    print(f"  Best loss         : {best_loss:.5f}")
    for name, val in zip(PARAM_NAMES, best_theta):
        print(f"  {name:8s} = {val:.5f}")

    return {
        "theta_all": theta_all,
        "loss_all": loss_all,
        "best_theta": best_theta,
        "best_loss": best_loss,
        "gp": gp_final,
        "bounds": bt,
    }


# GP Validation

def validate_gp(gp, theta_all, loss_all, bounds_tensor, n_holdout=50):
    """
    Check GP emulator accuracy by comparing its predictions against actual
    simulator outputs on a random subset of evaluated points.
    """
    rng = np.random.default_rng(SEED + 133)
    finite = loss_all < PENALTY * 0.1
    n_avail = finite.sum()
    n_ho = min(n_holdout, n_avail // 5) # Hold out at most 20% of finite points to keep enough data for training the GP
    if n_ho < 5:
        print("Not enough data for GP validation.")
        return

    idx = rng.choice(np.where(finite)[0], size=n_ho, replace=False)
    mu, sigma = gp_predict(gp, theta_all[idx], bounds_tensor)
    y_true = -loss_all[idx]
    resid = y_true - mu # Residuals between true values and GP predictions
    within = np.abs(resid) <= 1.645 * sigma # Check if true values fall within 90% confidence interval (1.645 std devs for normal dist)
    ss_res = np.sum(resid**2) # Residual sum of squares
    ss_tot = np.sum((y_true - y_true.mean())**2) # Total sum of squares
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0 # R² = 1 - (unexplained variance / total variance)

    print(f"\nGP hold-out validation ({n_ho} points):")
    print(f"  RMSE            = {np.sqrt(np.mean(resid**2)):.4f}")
    print(f"  MAE             = {np.mean(np.abs(resid)):.4f}")
    print(f"  R²              = {r2:.3f}")
    print(f"  90% CI coverage = {within.mean():.1%}  (target ~90%)")

    if within.mean() < 0.7:
        print("WARNING: CI coverage below 70%")
    if r2 < 0.5:
        print("WARNING: R² < 0.5")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        N_JOBS = int(sys.argv[1])
    from calibration_utils import load_data
    print("Loading empirical data...")
    data = load_data()
    results = run_bo(data)
    validate_gp(results["gp"], results["theta_all"],
                results["loss_all"], results["bounds"])
