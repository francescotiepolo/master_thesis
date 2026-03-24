"""
Sobol global sensitivity analysis of the calibration loss function.

Evaluates how all 6 PS-only parameters (s, c, c_prime, gamma, kappa, sigma)
affect the calibration loss and its two components (rank_C, alpha),
using empirical UN Comtrade data (1988-2010) as fixed input.

This complements the hysteresis SA in ../sensitivity_analysis.py providing a
basis for the free/fixed parameter selection used in calibration.

Outputs (per parameter):
  loss_total — weighted aggregate calibration loss
  rank_C — 1 - Spearman(C_sim, C_obs), country activity rank
  alpha — 1 - Spearman(entropy_sim, entropy_obs), diversification ordering
"""

import os
import sys
import csv
import warnings
import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

from SALib.sample import sobol as saltelli
from SALib.analyze import sobol

# Directories
_this_dir = os.path.dirname(os.path.abspath(__file__))
_root_dir = os.path.dirname(os.path.dirname(_this_dir))
_calib_dir = os.path.join(_root_dir, "calibration")
sys.path.insert(0, _root_dir)
sys.path.insert(0, _calib_dir)

from calibration_config import (
    FIXED, CALIB_YEARS, LOSS_WEIGHTS, TIME_WEIGHT_SLOPE,
    YEAR_START, SIM_STEPS_PER_YEAR,
)
from calibration_utils import load_data, compute_stats, _patch_model
from product_space_model import ProductSpaceModel

# Configuration
N_SOBOL = 1024
N_JOBS = -1 # -1 = all cores

RESULTS_DIR = os.path.join(_this_dir, "results")

# Parameters bounds
SA_PARAM_NAMES = ["s", "c", "c_prime", "gamma", "kappa", "sigma", "nu", "beta_trade_off", "h_mean", "C_diag_mean", "C_offdiag_mean", "entry_threshold"]
SA_PARAM_BOUNDS = [
    (0.0, 1.0),    # s
    (0.001, 2.0),  # c
    (0.0, 2.0),    # c_prime
    (0.0, 5.0),    # gamma
    (0.0, 1.0),    # kappa
    (0.1, 2.0),    # sigma
    (0.0, 1.0),    # nu
    (0.0, 1.0),    # beta_trade_off
    (0.05, 1.0),   # h_mean
    (0.1, 3.0),    # C_diag_mean
    (0.001, 0.5),  # C_offdiag_mean
    (0.01, 10.0),  # entry_threshold
]

PROBLEM = {
    "num_vars": len(SA_PARAM_NAMES),
    "names": SA_PARAM_NAMES,
    "bounds": SA_PARAM_BOUNDS,
}

OUTPUT_NAMES = ["loss_total", "rank_C", "alpha"]

COLORS = {
    "loss_total": ("#333333", "#999999"),
    "rank_C": ("#E64B35", "#F4A582"),
    "alpha": ("#4DBBD5", "#92C5DE"),
}


# Model construction

def _build_model_sa(s, c, c_prime, gamma, kappa, sigma, nu, beta_trade_off, h_mean, C_diag_mean, C_offdiag_mean, entry_threshold, data):
    """
    Build ProductSpaceModel with all 12 SA parameters and empirical data.
    """
    fp = FIXED
    model = ProductSpaceModel(
        N_products = data["SP"],
        n_countries = data["SC"],
        patch_network = True,
        seed = 133,
        phi_space = data["phi_space"],
        s = float(s),
        c = float(c),
        c_prime = float(c_prime),
        gamma = float(gamma),
        kappa = float(kappa),
        sigma = float(sigma),
        nu = float(nu),
        G = float(fp["G"]),
        q = float(fp["q"]),
        mu = float(fp["mu"]),
        beta_trade_off = float(beta_trade_off),
        enable_entry = True,
        entry_threshold = float(entry_threshold),
    )
    _patch_model(model, data, h_mean=float(h_mean),
                 C_diag_mean=float(C_diag_mean), C_offdiag_mean=float(C_offdiag_mean))
    return model


# Simulation

def _simulate_sa(s, c, c_prime, gamma, kappa, sigma, nu, beta_trade_off, h_mean, C_diag_mean, C_offdiag_mean, entry_threshold, data, years):
    """
    Year-by-year simulation from YEAR_START with the 12-param model.
    Returns dict {year: (alpha, C, P)} or None if failure.
    """
    model = _build_model_sa(s, c, c_prime, gamma, kappa, sigma, nu, beta_trade_off, h_mean, C_diag_mean, C_offdiag_mean, entry_threshold, data)
    y0 = np.concatenate([data["P_init"], data["C_init"], model.alpha.flatten()])

    results = {}
    prev_year = YEAR_START

    for year in sorted(set(years)):
        n_yr = year - prev_year
        if n_yr <= 0:
            results[year] = (model.alpha.copy(), data["C_init"].copy(), data["P_init"].copy())
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.solve(
                    t_end = float(n_yr),
                    d_C = float(FIXED["d_C"]),
                    n_steps = 10 * n_yr, # SA only needs final state
                    y0 = y0
                )
        except Exception:
            return None

        if model.y is None or model.y.shape[1] == 0:
            return None

        P_t = np.maximum(model.y[:model.SP, -1], 0.0)
        C_t = np.maximum(model.y[model.SP:model.N, -1], 0.0)

        if model.y_partial is None or model.y_partial.shape[1] == 0:
            return None
        alpha_t = np.clip(model.y_partial[:, -1].reshape(model.SC, model.SP), 0.0, 1.0)

        if not np.all(np.isfinite(P_t)) or not np.all(np.isfinite(C_t)):
            return None
        if np.max(P_t) > 1e6 or np.max(C_t) > 1e6:
            return None

        # Activate new product links between yearly steps
        model.activate_new_links(P_t, C_t, alpha_t)

        results[year] = (alpha_t, C_t, P_t)
        y0 = np.concatenate([P_t, C_t, alpha_t.flatten()])
        prev_year = year

    return results


# Evaluation

def _evaluate(params_row, data):
    """
    Evaluate calibration loss components for one parameter vector.
    Returns array [loss_total, rank_C_loss, alpha_loss].
    Returns NaN array on failure.
    """
    nan_out = np.full(len(OUTPUT_NAMES), np.nan)
    s, c, c_prime, gamma, kappa, sigma, nu, beta_trade_off, h_mean, C_diag_mean, C_offdiag_mean, entry_threshold = params_row

    try:
        traj = _simulate_sa(s, c, c_prime, gamma, kappa, sigma, nu, beta_trade_off, h_mean, C_diag_mean, C_offdiag_mean, entry_threshold, data, CALIB_YEARS)
        if traj is None:
            return nan_out

        w = LOSS_WEIGHTS
        totals = {"rank_C": 0.0, "alpha": 0.0}
        total_w = 0.0

        for yr in CALIB_YEARS:
            if yr not in traj or yr not in data["alpha_obs"]:
                continue
            alpha_sim, C_sim, _ = traj[yr]
            stats = compute_stats(alpha_sim, C_sim, data["alpha_obs"][yr], data["C_obs"][yr])
            wt = 1.0 + TIME_WEIGHT_SLOPE * (yr - YEAR_START) # Make later years more important
            for k in totals:
                totals[k] += wt * stats[k]
            total_w += wt

        if total_w == 0:
            return nan_out

        # Compute weighted mean loss and total loss
        rc = totals["rank_C"] / total_w
        al = totals["alpha"] / total_w
        lt = w["rank_C"] * rc + w["alpha"] * al

        return np.array([lt, rc, al])

    except Exception:
        return nan_out


# Sobol run

def run_sobol(data, n=N_SOBOL):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    param_values = saltelli.sample(PROBLEM, n, calc_second_order=False)
    n_runs = len(param_values)

    print("=" * 60)
    print("Sobol SA — Calibration Loss")
    print(f"Params : {PROBLEM['num_vars']}   N: {n}   Runs: {n_runs}")
    print(f"Calib years: {CALIB_YEARS}")
    print("=" * 60)

    # Warm up numba JIT cache 
    print("Warming up numba cache...", flush=True)
    _evaluate(param_values[0], data)
    print("Numba cache ready.", flush=True)

    # Run in chunks with explicit worker restarts between chunks.
    from joblib import cpu_count
    from joblib.externals.loky import get_reusable_executor
    n_jobs_eff = cpu_count() if N_JOBS == -1 else N_JOBS
    CHUNK = n_jobs_eff * 8 # Each worker handles ~8 tasks per chunk
    results = []
    for chunk_start in range(0, n_runs, CHUNK):
        chunk_end = min(chunk_start + CHUNK, n_runs)
        print(f"  chunk {chunk_start}–{chunk_end} / {n_runs}", flush=True)
        batch = Parallel(n_jobs=N_JOBS, backend="loky", verbose=0,
                         prefer="processes")(
            delayed(_evaluate)(param_values[i], data)
            for i in range(chunk_start, chunk_end)
        )
        results.extend(batch)
        # Kill worker pool so next chunk starts with fresh processes (no memory carryover)
        get_reusable_executor(max_workers=n_jobs_eff).shutdown(wait=True, kill_workers=True)
    sys.stdout.flush()

    Y = np.array(results)

    n_failed = np.isnan(Y).any(axis=1).sum()
    if n_failed:
        frac = n_failed / len(Y)
        print(f"  WARNING: {n_failed}/{len(Y)} ({frac:.1%}) runs failed or violated "
              f"c_prime <= c — imputing with column medians.")
        for col in range(Y.shape[1]):
            mask = np.isnan(Y[:, col])
            Y[mask, col] = np.nanmedian(Y[:, col])

    np.save(os.path.join(RESULTS_DIR, "Y.npy"), Y)
    np.save(os.path.join(RESULTS_DIR, "param_values.npy"), param_values)

    si_dict = {}
    for k, out_name in enumerate(OUTPUT_NAMES):
        Si = sobol.analyze(PROBLEM, Y[:, k].astype(float), calc_second_order=False,
                           print_to_console=False)
        for key in ("S1", "S1_conf", "ST", "ST_conf"):
            Si[key] = np.array([float(np.squeeze(v)) for v in Si[key]])
        si_dict[out_name] = Si
        print(f"\n  [{out_name}]")
        print(f"    {'Param':<12}  S1 +/- conf      ST +/- conf")
        for j, pname in enumerate(PROBLEM["names"]):
            print(f"    {pname:<12}"
                  f"  {float(Si['S1'][j]):+.3f} +/- {float(Si['S1_conf'][j]):.3f}"
                  f"  {float(Si['ST'][j]):+.3f} +/- {float(Si['ST_conf'][j]):.3f}")

    return {"Y": Y, "param_values": param_values, "Si": si_dict}


# Plotting

def _bar_pair(ax, names, S1, S1c, ST, STc, title, c1, c2):
    x = np.arange(len(names))
    w = 0.35
    ax.bar(x - w/2, S1, w, yerr=S1c, label="S1 (first-order)", color=c1, capsize=4, alpha=0.9)
    ax.bar(x + w/2, ST, w, yerr=STc, label="ST (total-order)", color=c2, capsize=4, alpha=0.85)
    ax.axhline(0, color="black", linewidth=0.7, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_ylabel("Sobol index")
    ax.legend(fontsize=8)
    ax.set_ylim(-0.1, 1.1)
    ax.spines[["top", "right"]].set_visible(False)


def plot_results(result):
    names = PROBLEM["names"]
    Si    = result["Si"]

    fig, axes = plt.subplots(
        len(OUTPUT_NAMES), 1,
        figsize=(max(8, len(names) * 1.4), len(OUTPUT_NAMES) * 3.5),
    )
    fig.suptitle("Sobol SA — Calibration Loss Components",
                 fontsize=13, fontweight="bold", y=1.01)

    for out_name, ax in zip(OUTPUT_NAMES, axes):
        s = Si[out_name]
        c1, c2 = COLORS[out_name]
        _bar_pair(ax, names, s["S1"], s["S1_conf"], s["ST"], s["ST_conf"],
                  f"Output: {out_name}", c1, c2)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(RESULTS_DIR, "sobol_calibration_loss.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close()


def plot_output_distributions(result):
    Y = result["Y"]
    fig, axes = plt.subplots(1, len(OUTPUT_NAMES),
                             figsize=(len(OUTPUT_NAMES) * 3.5, 3.5))
    fig.suptitle("Output Distributions Across Sobol Samples",
                 fontsize=11, fontweight="bold")

    for k, (out_name, ax) in enumerate(zip(OUTPUT_NAMES, axes)):
        vals = Y[:, k][~np.isnan(Y[:, k])]
        c1, _ = COLORS[out_name]
        ax.hist(vals, bins=30, color=c1, alpha=0.85, edgecolor="white")
        ax.axvline(np.median(vals), color="black", linestyle="--", linewidth=1.2,
                   label=f"median={np.median(vals):.3f}")
        ax.set_title(out_name, fontsize=9, fontweight="bold")
        ax.set_xlabel("Loss")
        ax.set_ylabel("Count")
        ax.legend(fontsize=7)
        ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "output_distributions.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close()


def save_indices_csv(result):
    path  = os.path.join(RESULTS_DIR, "sobol_indices.csv")
    names = PROBLEM["names"]
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["output", "parameter", "S1", "S1_conf", "ST", "ST_conf"])
        for out_name in OUTPUT_NAMES:
            Si = result["Si"][out_name]
            for j, pname in enumerate(names):
                writer.writerow([
                    out_name, pname,
                    round(Si["S1"][j], 5), round(Si["S1_conf"][j], 5),
                    round(Si["ST"][j], 5), round(Si["ST_conf"][j], 5),
                ])
    print(f"Saved: {path}")


# Main

if __name__ == "__main__":
    extracted_dir = os.path.join(_calib_dir, "extracted_data")
    data = load_data(extracted_dir=extracted_dir, years=list(range(1988, 2025)))

    result = run_sobol(data, n=N_SOBOL)
    plot_results(result)
    plot_output_distributions(result)
    save_indices_csv(result)

    print(f"\nAll results saved to: {RESULTS_DIR}/")
    print("Done.")
