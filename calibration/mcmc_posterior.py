"""
Phase 3: MCMC on the GP emulator to obtain the full posterior P(theta | data).

Uses emcee (ensemble sampler) with the BoTorch GP as a fast proxy likelihood.
The GP emulator is loaded exactly as saved in bo_emulator.py.

Log-posterior (Bayes' theorem: posterior ∝ likelihood × prior):
  log P(θ|data) = log L(θ) + log π(θ)

  The GP predicts Y = -loss (BoTorch convention: higher is better).

  Full Gaussian log-likelihood with proper normalisation:
    log L(θ) = -0.5 * log(τ²) - 0.5 * loss(θ)² / τ²
    where τ² = σ_GP(θ)² + σ_obs² + σ_model²

  The -0.5*log(τ²) term penalises high uncertainty. Without it, the
  sampler would favour uncertain regions simply because loss(θ)²/τ²
  becomes small when τ² is large, even if the loss itself is not low.

  Prior: π(θ) = Uniform over the NROY region with c_prime ≤ c.
  All valid points get equal prior weight, so the posterior shape
  comes entirely from the likelihood (the GP-predicted loss).

Saves:
  mcmc/samples.npy      -- (n_draws, d) posterior draws after burn-in + thinning
  mcmc/log_prob.npy     -- corresponding log-posterior values
  mcmc/chain.npy        -- full chain (n_steps/thin, n_walkers, d) for diagnostics
  mcmc/summary.csv      -- mean, std, 5%, median, 95% per parameter
  mcmc/autocorr.npy     -- integrated autocorrelation times
  mcmc/diagnostics.json -- R-hat, ESS, acceptance rate
"""

import os
import warnings
import numpy as np
import pandas as pd
import json
import emcee

import torch
from botorch.models import SingleTaskGP
from botorch.utils.transforms import normalize

from calibration_config import (
    PARAM_NAMES, PARAM_BOUNDS, N_PARAMS,
    MCMC_N_WALKERS, MCMC_N_STEPS, MCMC_BURNIN, MCMC_THIN,
    BO_DIR, MCMC_DIR, SEED, FIXED, SIGMA_OBS, SIGMA_MODEL,
)
from calibration_utils import load_bo_gp, PENALTY

warnings.filterwarnings("ignore")
torch.set_default_dtype(torch.float64)


# Log-posterior

def _make_log_prob(gp, lo, hi, bt):
    """
    Construct log-posterior function for emcee.

    The GP predicts -loss (higher = better). We convert back to loss and
    define the log-likelihood as:

      log L(θ) = -0.5 * loss_GP(θ)² / (σ_GP² + σ_obs² + σ_model²)

    where loss_GP(θ) = -μ_GP(θ) is the predicted loss.

    This correctly assigns:
      - High probability to θ with low predicted loss
      - Broader probability when GP uncertainty is high
      - Total variance τ² combining GP, observation and model uncertainty
    """
    obs_var = SIGMA_OBS ** 2
    model_var = SIGMA_MODEL ** 2

    def log_prob(theta):
        # Prior: uniform on NROY bounds (outise bounds prob is -inf)
        if np.any(theta < lo) or np.any(theta > hi):
            return -np.inf
        # Constraint: c_prime <= c
        params = dict(zip(PARAM_NAMES, theta))
        if params.get("c_prime", 0) > FIXED["c"]:
            return -np.inf

        # GP prediction: mu is -loss (higher = better fit)
        theta_norm = (theta - lo) / (hi - lo)
        X = torch.tensor(theta_norm, dtype=torch.float64).unsqueeze(0)
        with torch.no_grad():
            posterior = gp.posterior(X)
            mu = float(posterior.mean.squeeze()) # -loss
            var_gp = float(posterior.variance.squeeze())

        # Convert to loss: loss = -mu
        predicted_loss = -mu
        # Total variance: emulator + observation + model discrepancy
        total_var = var_gp + obs_var + model_var

        # Full Gaussian log-likelihood with normalisation term.
        # The -0.5*log(total_var) prevents the posterior from being biased
        # toward regions where the GP is uncertain (where var_gp is large,
        # the loss penalty is weaker, but without this term, there's no
        # cost for that uncertainty).
        # Kennedy, M.C. & O'Hagan, A. (2001). "Bayesian calibration of computer models." Journal of the Royal Statistical Society: Series B, 63(3), 425–464.
        return -0.5 * np.log(total_var) - 0.5 * (predicted_loss ** 2) / total_var

    return log_prob


# Convergence diagnostics

def gelman_rubin(chain):
    """
    Gelman-Rubin R-hat convergence diagnostic for multi-chain/walker MCMC.
    Compares between-chain variance to within-chain variance: if all
    walkers are sampling from the same distribution these
    should be roughly equal (R-hat ~ 1). R-hat > 1.1 signals that
    chains may be exploring different regions of the posterior.
    Simplified large-sample formulation (drops Student-t df correction):
      var_hat = ((n-1)/n)*W + B/n;  R-hat = sqrt(var_hat / W)
    Refs: Gelman & Rubin (1992), Statistical Science, 7(4), 457-472 (original);
          Gelman et al. (2013), Bayesian Data Analysis (3rd ed.), sec. 11.4 (formula used).
    chain: (n_steps, n_walkers, n_params)
    """
    n_steps, n_walkers, n_params = chain.shape
    r_hat = np.zeros(n_params)

    for p in range(n_params):
        chain_means = chain[:, :, p].mean(axis=0)
        chain_vars = chain[:, :, p].var(axis=0, ddof=1)

        W = chain_vars.mean()
        B = n_steps * chain_means.var(ddof=1)
        var_hat = (1 - 1/n_steps) * W + B / n_steps
        r_hat[p] = np.sqrt(var_hat / W) if W > 0 else np.inf

    return r_hat


def effective_sample_size(chain):
    """
    Estimate effective sample size (number of independent samples the
    chain is equivalent to) per parameter from the full chain.
    chain: (n_steps, n_walkers, n_params)
    ESS = n / (1 + 2*sum(autocorrelations))
    Autocorrelations at long lags (correlation between samples at different time lags)
    are noisy and can go negative by chance. Summing them all would underestimate
    the true autocorrelation time, giving an inflated ESS.
    Here the autocorrelation sum is truncated at the first negative lag (Geyer 1992)
    rather than summed to the full chain length.
    Ref: Geyer (1992), Statistical Science, 7(4), 473-483.
    """
    n_steps, n_walkers, n_params = chain.shape
    ess = np.zeros(n_params)

    for p in range(n_params):
        flat = chain[:, :, p].flatten()
        n = len(flat)
        if n < 10:
            ess[p] = n
            continue
        mean = flat.mean()
        var = flat.var()
        if var < 1e-20:
            ess[p] = n
            continue

        # Initial positive sequence estimator (Geyer 1992)
        max_lag = min(n // 2, 1000) # Limit max lag to avoid excessive noise, also unnecessary given truncation
        acf_sum = 0.0
        for lag in range(1, max_lag):
            acf = np.mean((flat[:n-lag] - mean) * (flat[lag:] - mean)) / var # Autocorrelation function
            if acf < 0.0:
                break
            acf_sum += acf
        ess[p] = max(n / (1 + 2 * acf_sum), 1.0)

    return ess


# MCMC - Markov Chain Monte Carlo

def run_mcmc():
    """
    Run emcee on the GP emulator and save posterior samples.
    """
    os.makedirs(MCMC_DIR, exist_ok=True)

    gp, lo, hi, bt = load_bo_gp()
    log_prob_fn = _make_log_prob(gp, lo, hi, bt)

    # Initialise walkers around the BO best point
    rng = np.random.default_rng(SEED)
    best_path = os.path.join(BO_DIR, "best_theta.npy")
    if os.path.exists(best_path):
        best = np.load(best_path)
        scale = 0.03 * (hi - lo)
        pos = np.clip(
            best[np.newaxis] + rng.normal(0, 1, (MCMC_N_WALKERS, N_PARAMS)) * scale,
            lo, hi
        )
    else:
        pos = lo + rng.uniform(0, 1, (MCMC_N_WALKERS, N_PARAMS)) * (hi - lo)

    # Verify all initial positions have valid log-prob
    for i in range(MCMC_N_WALKERS):
        lp = log_prob_fn(pos[i])
        if not np.isfinite(lp):
            pos[i] = lo + rng.uniform(0, 1, N_PARAMS) * (hi - lo)

    n_production = MCMC_N_STEPS - MCMC_BURNIN
    n_draws = MCMC_N_WALKERS * n_production // MCMC_THIN # Total posterior draws after burn-in and thinning
    print(f"\nRunning emcee:")
    print(f"  Walkers : {MCMC_N_WALKERS}")
    print(f"  Steps   : {MCMC_N_STEPS}  (burn-in: {MCMC_BURNIN}, thin: {MCMC_THIN})")
    print(f"  σ_obs   : {SIGMA_OBS:.3f}  σ_model: {SIGMA_MODEL:.3f}")
    print(f"  Expected posterior draws: {n_draws:,}")

    sampler = emcee.EnsembleSampler(MCMC_N_WALKERS, N_PARAMS, log_prob_fn)

    # Burn-in
    print("  Running burn-in...")
    pos, _, _ = sampler.run_mcmc(pos, MCMC_BURNIN, progress=True)
    sampler.reset()

    # Production
    print("  Running production chain...")
    sampler.run_mcmc(pos, n_production, progress=True, thin_by=MCMC_THIN)

    flat_samples = sampler.get_chain(flat=True) # Flattened to compute posterior statistics (shape: n_steps × n_walkers, n_params)
    flat_log_prob = sampler.get_log_prob(flat=True)
    full_chain = sampler.get_chain() # Non-flattened for gelman-rubin and ESS (shape: n_steps/thin, n_walkers, n_params)

    # Acceptance rate 
    accept_rate = float(np.mean(sampler.acceptance_fraction))
    print(f"\n  Mean acceptance rate: {accept_rate:.1%}")
    if accept_rate < 0.15:
        print("  WARNING: Low acceptance rate")
    if accept_rate > 0.5:
        print("  WARNING: High acceptance rate")

    # Autocorrelation diagnostics
    try:
        autocorr = sampler.get_autocorr_time(quiet=True) # Number of steps for chain to become effectively independent from starting point
        np.save(os.path.join(MCMC_DIR, "autocorr.npy"), autocorr)
        print("\n Autocorrelation times:")
        for name, tau in zip(PARAM_NAMES, autocorr):
            status = "OK" if tau < n_production / (MCMC_THIN * 50) else "WARNING: may need more steps"
            print(f"    {name:8s}: τ = {tau:.1f}  ({status})")
    except emcee.autocorr.AutocorrError:
        autocorr = None
        print("  WARNING: chain too short for autocorrelation estimate")

    # Gelman-Rubin R-hat
    r_hat = gelman_rubin(full_chain)
    print(f"\n  Gelman-Rubin R-hat:")
    for name, rh in zip(PARAM_NAMES, r_hat):
        status = "converged" if rh < 1.1 else "NOT converged"
        print(f"    {name:8s}: R-hat = {rh:.3f}  ({status})")

    # Effective sample size
    ess = effective_sample_size(full_chain)
    print(f"\n  Effective sample size:")
    for name, e in zip(PARAM_NAMES, ess):
        print(f"    {name:8s}: ESS = {e:.0f}")

    # Posterior summary
    rows = []
    for i, name in enumerate(PARAM_NAMES):
        vals = flat_samples[:, i]
        rows.append({
            "parameter": name,
            "mean"     : float(np.mean(vals)),
            "std"      : float(np.std(vals)),
            "5%"       : float(np.percentile(vals, 5)),
            "median"   : float(np.median(vals)),
            "95%"      : float(np.percentile(vals, 95)),
        })
    summary_df = pd.DataFrame(rows)

    # Save
    np.save(os.path.join(MCMC_DIR, "samples.npy"), flat_samples)
    np.save(os.path.join(MCMC_DIR, "log_prob.npy"), flat_log_prob)
    np.save(os.path.join(MCMC_DIR, "chain.npy"), full_chain)
    summary_df.to_csv(os.path.join(MCMC_DIR, "summary.csv"), index=False)

    diagnostics = {
        "acceptance_rate": accept_rate,
        "r_hat"          : {name: float(rh) for name, rh in zip(PARAM_NAMES, r_hat)},
        "ess"            : {name: float(e) for name, e in zip(PARAM_NAMES, ess)},
        "n_draws"        : len(flat_samples),
        "sigma_obs"      : SIGMA_OBS,
        "sigma_model"    : SIGMA_MODEL
    }
    if autocorr is not None:
        diagnostics["autocorr"] = {name: float(t) for name, t in zip(PARAM_NAMES, autocorr)}
    with open(os.path.join(MCMC_DIR, "diagnostics.json"), "w") as f:
        json.dump(diagnostics, f, indent=2)

    print(f"\nMCMC complete")
    print(f"  Posterior draws : {len(flat_samples):,}")
    print(f"\n  Posterior summary:")
    print(summary_df.to_string(index=False, float_format="{:.5f}".format))

    return {"samples": flat_samples, "log_prob": flat_log_prob,
            "chain": full_chain, "summary": summary_df}


if __name__ == "__main__":
    run_mcmc()