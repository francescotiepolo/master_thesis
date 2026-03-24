"""
Phase 4: Validation and diagnostics.

  1. Posterior predictive check -- does the model under the posterior
                                    reproduce 1988-2010 calibration targets?
  2. Out-of-sample validation   -- simulate 2010-2024 with no re-fitting
  3. Structural stability       -- MAP estimates across three sub-windows
  4. Sobol on GP emulator       -- sensitivity analysis on BoTorch GP
  5. Posterior corner plot      -- marginal and pairwise joint distributions
"""

import os
import copy
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from SALib.sample import sobol as saltelli
from SALib.analyze import sobol


from calibration_config import (
    PARAM_NAMES, PARAM_BOUNDS, N_PARAMS,
    YEAR_START, CALIB_END, CALIB_YEARS, VALID_YEARS,
    STABILITY_WINDOWS, MCMC_DIR, BO_DIR, VAL_DIR, SEED, FIXED,
    GP_SOBOL_SAMPLES, N_PPC_DRAWS, N_OOS_DRAWS,
)
from calibration_utils import (
    load_data, simulate, compute_stats, trajectory_loss,
    load_bo_gp, gp_predict, PENALTY,
)

torch.set_default_dtype(torch.float64)
N_JOBS = -1


# Shared simulation helper

def _simulate_draws(draws, sim_data, obs_data, years):
    """
    Run the model for each posterior draw and collect per-year statistics.
    sim_data: data dict passed to simulate() (sets initial conditions)
    obs_data: data dict used for computing stats against observed values
    Returns (n_draws, n_years, 2) array or None if all fail.
    """
    def _sim_one(theta):
        traj = simulate(theta, sim_data, years)
        if traj is None:
            return None
        row = []
        for yr in years:
            if yr not in traj or yr not in obs_data["alpha_obs"]:
                row.append([np.nan, np.nan])
                continue
            alpha_s, C_s, _ = traj[yr]
            s = compute_stats(alpha_s, C_s, obs_data["alpha_obs"][yr], obs_data["C_obs"][yr])
            row.append([s["rank_C"], s["alpha"]])
        return np.array(row)

    results = [r for r in Parallel(n_jobs=N_JOBS)(delayed(_sim_one)(d) for d in draws) if r is not None]
    if len(results) == 0:
        return None
    return np.stack(results)


# Posterior predictive check (1988-2010)

def posterior_predictive_check(data, n_draws=None):
    """
    Posterior predictive check: run the model forward over the calibration
    period (1988-2010) for n_draws parameter vectors sampled from the MCMC
    posterior, then compare the resulting trajectories against observed data.
    Saves ppc_trajectories.npy and ppc_calibration.pdf
    """
    if n_draws is None:
        n_draws = N_PPC_DRAWS

    samples = np.load(os.path.join(MCMC_DIR, "samples.npy"))
    rng = np.random.default_rng(SEED)
    draws = samples[rng.choice(len(samples), size=min(n_draws, len(samples)), replace=False)]
    years = CALIB_YEARS
    print(f"PPC: {len(draws)} draws x {len(years)} years")

    ppc = _simulate_draws(draws, data, data, years)
    if ppc is None:
        print("  WARNING: All PPC simulations failed")
        return None
    np.save(os.path.join(VAL_DIR, "ppc_trajectories.npy"), ppc)
    print(f"  {len(ppc)}/{len(draws)} simulations succeeded.")

    stat_names = ["1 - Spearman(C rank)", "1 - Spearman(alpha)"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle("Posterior Predictive Check (1988-2010)", fontsize=12, fontweight="bold")
    for k, (ax, sname) in enumerate(zip(axes, stat_names)):
        median = np.nanmedian(ppc[:, :, k], axis=0)
        lo5 = np.nanpercentile(ppc[:, :, k], 5, axis=0)
        hi95 = np.nanpercentile(ppc[:, :, k], 95, axis=0)
        ax.fill_between(years, lo5, hi95, alpha=0.3, color="#4C72B0", label="90% CI")
        ax.plot(years, median, color="#4C72B0", lw=2, label="Posterior median")
        ax.set_title(sname, fontsize=10, fontweight="bold")
        ax.set_xlabel("Year")
        ax.legend(fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(VAL_DIR, "ppc_calibration.pdf"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: ppc_calibration.pdf")
    return ppc


# Out-of-sample validation (2010-2024)

def out_of_sample_validation(data, n_draws=None):
    """
    Simulate 2010-2024 under posterior draws. No re-fitting.
    The OOS simulation needs different initial conditions (2010 observed
    state instead of 1988), thus override alpha_init, C_init, P_init.
    """
    if n_draws is None:
        n_draws = N_OOS_DRAWS

    samples = np.load(os.path.join(MCMC_DIR, "samples.npy"))
    rng = np.random.default_rng(SEED + 1)
    draws = samples[rng.choice(len(samples), size=min(n_draws, len(samples)), replace=False)]
    years = VALID_YEARS
    print(f"Out-of-sample: {len(draws)} draws x {len(years)} years")

    # Deep copy to avoid changing the original data
    data_2010 = copy.deepcopy(data)
    if CALIB_END in data["alpha_obs"]:
        data_2010["alpha_init"] = data["alpha_obs"][CALIB_END].copy()
        data_2010["C_init"] = data["C_obs"][CALIB_END].copy()
        data_2010["P_init"] = data["P_obs"][CALIB_END].copy()
    else:
        print(f"  WARNING: No observed data for year {CALIB_END}, using original")

    val_ppc = _simulate_draws(draws, data_2010, data, years)
    if val_ppc is None:
        print("  WARNING: All OOS simulations failed.")
        return None
    np.save(os.path.join(VAL_DIR, "oos_trajectories.npy"), val_ppc)
    print(f"  {len(val_ppc)}/{len(draws)} simulations succeeded.")

    stat_names = ["1 - Spearman(C rank)", "1 - Spearman(alpha)"]
    fig, axes  = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle("Out-of-Sample Validation (2010-2024)", fontsize=12, fontweight="bold")
    for k, (ax, sname) in enumerate(zip(axes, stat_names)):
        median = np.nanmedian(val_ppc[:, :, k], axis=0)
        lo5 = np.nanpercentile(val_ppc[:, :, k], 5, axis=0)
        hi95 = np.nanpercentile(val_ppc[:, :, k], 95, axis=0)
        ax.fill_between(years, lo5, hi95, alpha=0.3, color="#DD8452", label="90% posterior CI")
        ax.plot(years, median, color="#DD8452", lw=2, label="Posterior median")
        ax.axvline(CALIB_END, color="gray", linestyle=":", lw=1.5, label="Calibration end")
        ax.set_title(sname, fontsize=10, fontweight="bold")
        ax.set_xlabel("Year")
        ax.legend(fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(VAL_DIR, "oos_validation.pdf"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: oos_validation.pdf")
    return val_ppc


# Structural stability check

def structural_stability_check(data, n_candidates=20):
    """
    Test whether the best-fit parameters change when calibrated on
    different time sub-windows (e.g. 1988-2000, 2000-2012, 2012-2024).
    Large shifts would indicate the model is overfitting to a specific
    period rather than capturing stable dynamics.

    The GP emulator pre-screens the BO archive to find the top candidates,
    then only those are evaluated with the full simulator on each window.
    """
    theta_all = np.load(os.path.join(BO_DIR, "theta_all.npy"))
    loss_all = np.load(os.path.join(BO_DIR, "loss_all.npy"))
    finite = loss_all < PENALTY * 0.1
    theta_fin = theta_all[finite]
    print(f"Stability check: {len(theta_fin)} archive points x {len(STABILITY_WINDOWS)} windows")

    # Use GP emulator to quickly pre-screen candidates
    gp, lo, hi, bt = load_bo_gp()
    mu_gp, _ = gp_predict(gp, theta_fin, bt)
    # mu_gp is -loss; sort by predicted loss (best first)
    top_idx = np.argsort(-mu_gp)[:n_candidates]
    theta_top = theta_fin[top_idx]
    print(f"  Pre-screened to top {len(theta_top)} candidates via GP emulator.")

    window_maps = {}
    for (wstart, wend) in STABILITY_WINDOWS:
        years_w = [y for y in range(wstart, wend + 1, 2) if y in data["alpha_obs"]]
        if len(years_w) < 2:
            continue
        losses_w = np.array([
            trajectory_loss(theta_top[i], data, years=years_w)
            for i in range(len(theta_top))
        ])
        fin_w = losses_w < PENALTY * 0.1
        if not fin_w.any():
            print(f"  {wstart}-{wend}: no finite evaluations -- skipping.")
            continue
        best_theta_w = theta_top[fin_w][losses_w[fin_w].argmin()]
        window_maps[(wstart, wend)] = {
            "best_theta": best_theta_w.tolist(),
            "best_loss" : float(losses_w[fin_w].min()),
        }
        print(f"  {wstart}-{wend}:")
        for name, val in zip(PARAM_NAMES, best_theta_w):
            print(f"    {name:8s} = {val:.5f}")

    # Stability plot
    if len(window_maps) >= 2:
        colors = ["#4C72B0", "#DD8452", "#55A868"]
        fig, axes = plt.subplots(1, N_PARAMS, figsize=(3 * N_PARAMS, 3.5))
        for k, (name, ax) in enumerate(zip(PARAM_NAMES, axes)):
            for j, ((ws, we), info) in enumerate(window_maps.items()):
                ax.axvline(info["best_theta"][k], color=colors[j % 3], lw=2,
                           label=f"{ws}-{we}")
            ax.set_title(name, fontsize=9, fontweight="bold")
            ax.set_yticks([])
            if k == 0:
                ax.legend(fontsize=7)
            ax.spines[["top", "right", "left"]].set_visible(False)
        plt.suptitle("Parameter MAP Estimates Across Sub-Windows",
                     fontsize=11, fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(VAL_DIR, "stability_map.pdf"), dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: stability_map.pdf")

    with open(os.path.join(VAL_DIR, "stability_maps.json"), "w") as f:
        json.dump({str(k): v for k, v in window_maps.items()}, f, indent=2)
    return window_maps


# Sobol on GP emulator

def sobol_on_gp(n_samples=None):
    """
    Run SALib Sobol analysis on the GP emulator trained in the BO phase,
    to identify which parameters drive the calibration loss.
    """
    if n_samples is None:
        n_samples = GP_SOBOL_SAMPLES

    gp, lo, hi, bt = load_bo_gp()

    problem = {
        "num_vars": N_PARAMS,
        "names": PARAM_NAMES,
        "bounds": [[lo[i], hi[i]] for i in range(N_PARAMS)],
    }
    param_values = saltelli.sample(problem, n_samples, calc_second_order=True)

    # Evaluate the GP emulator (cheap) instead of the full simulator at each Sobol sample point
    mu, _ = gp_predict(gp, param_values, bt)
    # GP predicts -loss; Sobol needs loss (positive)
    Y_gp = -mu

    Si = sobol.analyze(problem, Y_gp, calc_second_order=True, print_to_console=False)

    # First-order and total-effect indices
    sa_df = pd.DataFrame({
        "parameter": PARAM_NAMES,
        "S1": Si["S1"],
        "S1_conf": Si["S1_conf"],
        "ST": Si["ST"],
        "ST_conf": Si["ST_conf"],
    })
    sa_df.to_csv(os.path.join(VAL_DIR, "sobol_on_gp.csv"), index=False)

    # Second-order interaction indices
    s2_rows = []
    for i in range(N_PARAMS):
        for j in range(i + 1, N_PARAMS):
            s2_rows.append({
                "param_i": PARAM_NAMES[i],
                "param_j": PARAM_NAMES[j],
                "S2": Si["S2"][i, j],
                "S2_conf": Si["S2_conf"][i, j],
            })
    s2_df = pd.DataFrame(s2_rows)
    s2_df.to_csv(os.path.join(VAL_DIR, "sobol_on_gp_s2.csv"), index=False)

    # Bar plot S1 and ST
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(N_PARAMS)
    w = 0.35
    ax.bar(x - w/2, Si["S1"], w, yerr=Si["S1_conf"], label="S1 (first-order)",
           color="#4C72B0", capsize=4, alpha=0.9)
    ax.bar(x + w/2, Si["ST"], w, yerr=Si["ST_conf"], label="ST (total-effect)",
           color="#DD8452", capsize=4, alpha=0.9)
    ax.axhline(0.05, color="red", linestyle="--", lw=1, label="0.05 threshold")
    ax.set_xticks(x)
    ax.set_xticklabels(PARAM_NAMES, rotation=30, ha="right")
    ax.set_ylabel("Sobol index")
    ax.set_title("Sobol Sensitivity on GP Emulator (calibration loss)",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=8)
    ax.set_ylim(-0.1, 1.1)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(VAL_DIR, "sobol_on_gp.pdf"), dpi=150, bbox_inches="tight")
    plt.close()

    # Heatmap: S2 interactions
    s2_matrix = np.full((N_PARAMS, N_PARAMS), np.nan)
    for i in range(N_PARAMS):
        for j in range(i + 1, N_PARAMS):
            s2_matrix[i, j] = Si["S2"][i, j]
            s2_matrix[j, i] = Si["S2"][i, j]
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(s2_matrix, cmap="YlOrRd", vmin=0)
    ax.set_xticks(range(N_PARAMS))
    ax.set_yticks(range(N_PARAMS))
    ax.set_xticklabels(PARAM_NAMES, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(PARAM_NAMES, fontsize=8)
    ax.set_title("Second-Order Sobol Interactions (S2)",
                 fontsize=10, fontweight="bold")
    plt.colorbar(im, ax=ax, shrink=0.8, label="S2 index")
    plt.tight_layout()
    plt.savefig(os.path.join(VAL_DIR, "sobol_on_gp_s2.pdf"), dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\nSobol on GP emulator:")
    print(sa_df.to_string(index=False, float_format="{:.4f}".format))
    print(f"\n  Top S2 interactions:")
    print(s2_df.nlargest(10, "S2").to_string(index=False, float_format="{:.4f}".format))
    print(f"  Saved: sobol_on_gp.pdf, sobol_on_gp_s2.pdf")
    return Si


# Posterior corner plot

def plot_posterior_corner():
    """
    Triangle plot of the joint posterior: marginal histograms on the diagonal,
    pairwise scatter plots below.
    """
    samples = np.load(os.path.join(MCMC_DIR, "samples.npy"))
    fig, axes = plt.subplots(N_PARAMS, N_PARAMS, figsize=(2.5 * N_PARAMS, 2.5 * N_PARAMS))
    for i in range(N_PARAMS):
        for j in range(N_PARAMS):
            ax = axes[i, j]
            if i == j:
                ax.hist(samples[:, i], bins=40, color="#4C72B0", alpha=0.8,
                        edgecolor="white", linewidth=0.5)
                ax.axvline(np.median(samples[:, i]), color="k", lw=1.5, linestyle="--")
            elif i > j:
                ax.scatter(samples[::10, j], samples[::10, i],
                           alpha=0.15, s=3, color="#4C72B0")
            else:
                ax.set_visible(False)
            if j == 0:
                ax.set_ylabel(PARAM_NAMES[i], fontsize=8)
            if i == N_PARAMS - 1:
                ax.set_xlabel(PARAM_NAMES[j], fontsize=8)
            ax.tick_params(labelsize=6)
            ax.spines[["top", "right"]].set_visible(False)
    plt.suptitle("Posterior P(θ | data)", fontsize=12, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(VAL_DIR, "posterior_corner.pdf"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: posterior_corner.pdf")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        N_JOBS = int(sys.argv[1])
    os.makedirs(VAL_DIR, exist_ok=True)
    print("Loading empirical data...")
    data = load_data()

    print("\n1. Posterior predictive check...")
    posterior_predictive_check(data)

    print("\n2. Out-of-sample validation (2010-2024)...")
    out_of_sample_validation(data)

    print("\n3. Structural stability check...")
    structural_stability_check(data)

    print("\n4. Sobol on GP emulator...")
    sobol_on_gp()

    print("\n5. Posterior corner plot...")
    plot_posterior_corner()

    print(f"\nAll outputs saved to: {VAL_DIR}/")