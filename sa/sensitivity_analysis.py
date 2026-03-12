"""
Sobol global sensitivity analysis for BaseModel and ProductSpaceModel.
Output metrics:
  d_collapse
  d_recovery
  hysteresis_width - d_collapse minus d_recovery

Three analyses:
  1. BaseModel - 5 shared parameters
  2. ProductSpaceModel - 5 shared + 6 PS-only parameters (13 total)
  3. Comparison of shared parameters across both models

- Choose option "fix_q_1" to test is hysteresis still feasible
  with full rivalry over available market (SA run over every other parameter)
- Choose option "second_order" to compute second-order Sobol indices
"""

import os
import sys
import socket
import csv
import warnings
import contextlib
import io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from joblib import Parallel, delayed

from SALib.sample import saltelli
from SALib.analyze import sobol

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from base_model import BaseModel
from product_space_model import ProductSpaceModel

# Choose if full SA or fix q=1
FIX_Q_1 = False
# Choose if second-order indices should be calculated 
SECOND_ORDER = True 

# Configuration
N_SOBOL = 2048 # Saltelli draws N*(2D+2) samples per model
N_JOBS = -1 # -1 = all cores, 1 = serial 
BASE_SEED = 133 # Each sample gets BASE_SEED + sample_index

# Hysteresis settings
D_C_MIN = 0.0
D_C_MAX = 5.0
D_C_STEP = 0.2
STEPS_AFTER_COLLAPSE = 3 
T_INIT = 2_000 # Integration time for initial equilibrium
T_STEP = 2_000 # Integration time per d_C step

# Directory for results
ON_CLUSTER = os.environ.get("ON_CLUSTER", "0") == "1" # Flag if running on cluster
_base_dir = "sa_results_q_fix" if FIX_Q_1 else "sa_results"
RESULTS_DIR = _base_dir + ("_cluster" if ON_CLUSTER else "")

# Fixed parameters for both models
FIXED_NESTEDNESS = 0.6
FIXED_CONNECTANCE = 0.15

# Parameter definitions
_SHARED_NAMES_BASE = ["nu", "G", "log10_mu", "beta_trade_off",
                      "h_mean", "r_mean", "C_diag_mean"]
_SHARED_BOUNDS_BASE = [
    [0.0, 1.0],   # nu
    [0.1, 2.0],   # G
    [-5, -3],     # log10_mu
    [0.1, 0.9],  # beta_trade_off
    [0.15, 0.30], # h_mean — h_P and h_C vectors set uniformly to this value
    [0.10, 0.35], # r_mean — r_P and r_C vectors set uniformly to this value
    [0.80, 1.10], # C_diag_mean — diagonal of C_PP and C_CC set uniformly
]

_SHARED_NAMES_FULL  = ["nu", "G", "q", "log10_mu", "beta_trade_off",
                       "h_mean", "r_mean", "C_diag_mean"]
_SHARED_BOUNDS_FULL = [
    [0.0, 1.0],   # nu
    [0.1, 2.0],   # G
    [0.0, 0.5],   # q
    [-5, -3],     # log10_mu
    [0.1, 0.9],   # beta_trade_off
    [0.15, 0.30], # h_mean
    [0.10, 0.35], # r_mean
    [0.80, 1.10], # C_diag_mean
]

SHARED_PARAMS = (
    {"names": _SHARED_NAMES_BASE, "bounds": _SHARED_BOUNDS_BASE}
    if FIX_Q_1 else
    {"names": _SHARED_NAMES_FULL, "bounds": _SHARED_BOUNDS_FULL}
)

PS_ONLY_PARAMS = {
    "names":  ["s", "c", "c_prime", "gamma", "kappa", "sigma"],
    "bounds": [
        [0.0, 0.10],  # s
        [0.001,0.10], # c
        [0.0, 0.05],  # c_prime
        [0.0, 0.20],  # gamma
        [0.0, 0.20],  # kappa
        [0.5, 2.0],   # sigma
    ],
}

# Problem definitions for SALib
PROBLEM_BASE = {
    "num_vars": len(SHARED_PARAMS["names"]),
    "names": SHARED_PARAMS["names"],
    "bounds": SHARED_PARAMS["bounds"],
}

PROBLEM_PS = {
    "num_vars": len(SHARED_PARAMS["names"]) + len(PS_ONLY_PARAMS["names"]),
    "names": SHARED_PARAMS["names"] + PS_ONLY_PARAMS["names"],
    "bounds": SHARED_PARAMS["bounds"] + PS_ONLY_PARAMS["bounds"],
}

OUTPUT_NAMES = ["d_collapse", "d_recovery", "hysteresis_width"]

# Colors for plotting
COLORS = {
    "d_collapse": ("#E64B35", "#F4A582"),
    "d_recovery": ("#4DBBD5", "#92C5DE"),
    "hysteresis_width": ("#00A087", "#80CDC1"),
}


# Helpers

def _unpack_shared(params_row):
    """
    Unpack shared parameters.
    Set to 1.0 when FIX_Q_1.
    """
    if FIX_Q_1:
        nu, G, log10_mu, beta_trade_off, h_mean, r_mean, C_diag_mean = params_row
        q = 1.0
    else:
        nu, G, q, log10_mu, beta_trade_off, h_mean, r_mean, C_diag_mean = params_row
    mu = 10.0 ** log10_mu
    return nu, G, q, mu, beta_trade_off, h_mean, r_mean, C_diag_mean

def _apply_shared_overrides(m, h_mean, r_mean, C_diag_mean):
    """
    Override sampled parameters.
    """
    m.h_P[:] = h_mean
    m.h_C[:] = h_mean
    m.r_P[:] = r_mean
    m.r_C[:] = r_mean
    np.fill_diagonal(m.C_PP, C_diag_mean)
    np.fill_diagonal(m.C_CC, C_diag_mean)


# Hysteresis evaluation function

def _run_hysteresis(model):
    """
    Re-implementation of find_critical_points to make SA faster
    """
    thresh = model.extinct_threshold # Use model's extinction threshold for collapse/recovery
    dCs = np.arange(D_C_MIN, D_C_MAX + D_C_STEP / 2, D_C_STEP)
    nan_out = {"d_collapse": np.nan, "d_recovery": np.nan, "hysteresis_width": np.nan} # In case of failure

    # Initial equilibrium
    model.solve(T_INIT, d_C=0.0, stop_on_equilibrium=True, save_period=0)
    if model.y is None:
        return nan_out

    # Forward pass
    C_fwd = []
    dC_fwd = []
    collapse_counter = 0
    collapsed = False

    for d_C in dCs:
        y0 = np.concatenate((model.y[:, -1], model.y_partial[:, -1]))
        model.solve(T_STEP, d_C=d_C, y0=y0, stop_on_equilibrium=True, save_period=0)
        if model.y is None:
            break
        C_now = model.y[model.SP:model.N, -1]
        C_fwd.append(C_now.copy())
        dC_fwd.append(d_C)
        if (C_now < thresh).all(): # Collapse condition
            if not collapsed:
                collapsed = True
            collapse_counter += 1
            if collapse_counter >= STEPS_AFTER_COLLAPSE:
                break

    if not C_fwd:
        return nan_out

    dC_fwd_arr = np.array(dC_fwd)
    C_fwd_arr  = np.array(C_fwd)

    # d_collapse
    col_mask = (C_fwd_arr < thresh).all(axis=1)
    d_collapse = float(dC_fwd_arr[np.argmax(col_mask)]) if col_mask.any() else D_C_MAX

    # Backward pass
    C_bwd  = []
    dC_bwd = []

    for d_C in np.flip(dC_fwd_arr):
        y0 = np.concatenate((model.y[:, -1], model.y_partial[:, -1]))
        model.solve(T_STEP, d_C=d_C, y0=y0, stop_on_equilibrium=True, save_period=0)
        if model.y is None:
            break
        C_bwd.append(model.y[model.SP:model.N, -1].copy())
        dC_bwd.append(d_C)

    if not C_bwd:
        return {"d_collapse": d_collapse, "d_recovery": np.nan, "hysteresis_width": np.nan}

    dC_bwd_arr = np.array(dC_bwd)
    C_bwd_arr  = np.array(C_bwd)

    # d_recovery
    rec_mask = (C_bwd_arr > thresh).any(axis=1)
    d_recovery = float(dC_bwd_arr[np.argmax(rec_mask)]) if rec_mask.any() else D_C_MIN

    hysteresis_width = max(0.0, d_collapse - d_recovery)

    return {
        "d_collapse":       d_collapse,
        "d_recovery":       d_recovery,
        "hysteresis_width": hysteresis_width,
    }


# Evaluation functions for Sobol runs

def _evaluate_base(params_row, seed): # For base model
    """
    One BaseModel hysteresis evaluation.
    """
    nu, G, q, mu, beta_trade_off, h_mean, r_mean, C_diag_mean = _unpack_shared(params_row)
    nan_out = np.full(len(OUTPUT_NAMES), np.nan)
    try:
        with contextlib.redirect_stdout(io.StringIO()): # Suppress printouts from model
            m = BaseModel(
                N_products=15, n_countries=35,
                nestedness=FIXED_NESTEDNESS, connectance=FIXED_CONNECTANCE,
                nu=float(nu), G=float(G), q=float(q),
                mu=float(mu), beta_trade_off=float(beta_trade_off),
                feasible=False, seed=int(seed),
            )
        _apply_shared_overrides(m, float(h_mean), float(r_mean), float(C_diag_mean))
        r = _run_hysteresis(m) # Run hysteresis analysis and extract critical points
        return np.array([r["d_collapse"], r["d_recovery"], r["hysteresis_width"]])
    except Exception as e:
        import traceback
        traceback.print_exc()
        return nan_out


def _evaluate_ps(params_row, seed): # For product space model (same as for base + PS params)
    """
    One ProductSpaceModel hysteresis evaluation.
    """
    n_shared = len(SHARED_PARAMS["names"])
    shared_row = params_row[:n_shared]
    ps_row = params_row[n_shared:]

    nu, G, q, mu, beta_trade_off, h_mean, r_mean, C_diag_mean = _unpack_shared(shared_row)
    s, c, c_prime, gamma, kappa, sigma = ps_row

    nan_out = np.full(len(OUTPUT_NAMES), np.nan)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            m = ProductSpaceModel(
                N_products=15, n_countries=35,
                nestedness=FIXED_NESTEDNESS, connectance=FIXED_CONNECTANCE,
                nu=float(nu), G=float(G), q=float(q),
                mu=float(mu), beta_trade_off=float(beta_trade_off),
                feasible=False, seed=int(seed),
                s=float(s), c=float(c), c_prime=float(c_prime),
                gamma=float(gamma), kappa=float(kappa), sigma=float(sigma),
            )
        _apply_shared_overrides(m, float(h_mean), float(r_mean), float(C_diag_mean))
        r = _run_hysteresis(m)
        return np.array([r["d_collapse"], r["d_recovery"], r["hysteresis_width"]])
    except Exception as e:
        import traceback
        traceback.print_exc()
        return nan_out


# Sobol analysis

def run_sobol(model_name, problem, eval_fn, n=N_SOBOL):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    param_values = saltelli.sample(problem, n, calc_second_order=SECOND_ORDER)
    n_runs = len(param_values)
    n_dC = len(np.arange(D_C_MIN, D_C_MAX + D_C_STEP / 2, D_C_STEP))
    print(f"{'='*60}")
    print(f"Sobol SA  {model_name}")
    print(f"Params: {problem['num_vars']}   N: {n}   Runs: {n_runs}"
          + ("  [second-order]" if SECOND_ORDER else ""))
    print(f"d_C steps per run: ~{n_dC * 2}  (fwd + bwd)")
    print(f"{'='*60}")

    seeds = BASE_SEED + np.arange(param_values.shape[0]) # Unique seed per run

    # Run evaluations in parallel
    results = Parallel(n_jobs=N_JOBS, backend="loky", verbose=5)(
        delayed(eval_fn)(param_values[i], int(seeds[i]))
        for i in range(param_values.shape[0])
    )

    Y = np.array(results)

    # Check for NaNs and impute with column medians if any (to allow Sobol analysis to proceed)
    n_failed = np.isnan(Y).any(axis=1).sum()
    if n_failed:
        print(f"  WARNING: {n_failed}/{len(Y)} failed runs imputed with medians.")
        for col in range(Y.shape[1]):
            mask = np.isnan(Y[:, col])
            Y[mask, col] = np.nanmedian(Y[:, col])

    np.save(os.path.join(RESULTS_DIR, f"{model_name}_Y.npy"), Y)
    np.save(os.path.join(RESULTS_DIR, f"{model_name}_param_values.npy"), param_values)

    # Perform Sobol analysis for each output metric
    si_dict = {}
    for k, out_name in enumerate(OUTPUT_NAMES):
        Si = sobol.analyze(problem, Y[:, k], calc_second_order=SECOND_ORDER, print_to_console=False)
        si_dict[out_name] = Si
        print(f"\n  [{out_name}]")
        print(f"    {'Param':<18}  S1 +/- conf      ST +/- conf")
        for j, pname in enumerate(problem["names"]):
            print(f"    {pname:<18}"
                  f"{Si['S1'][j]:+.3f} +/- {Si['S1_conf'][j]:.3f}"
                  f"{Si['ST'][j]:+.3f} +/- {Si['ST_conf'][j]:.3f}")

    return {"problem": problem, "Y": Y, "param_values": param_values, "Si": si_dict}


# Plotting

def _bar_pair(ax, names, S1, S1c, ST, STc, title, c1, c2):
    x = np.arange(len(names))
    w = 0.35
    ax.bar(x - w/2, S1, w, yerr=S1c, label="S1 (first-order)", color=c1, capsize=4, alpha=0.9)
    ax.bar(x + w/2, ST, w, yerr=STc, label="ST (total-order)", color=c2, capsize=4, alpha=0.9)
    ax.axhline(0, color="black", linewidth=0.7, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=35, ha="right", fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_ylabel("Sobol index")
    ax.legend(fontsize=8)
    ax.set_ylim(-0.1, 1.1)
    ax.spines[["top", "right"]].set_visible(False)


def plot_single_model(result, model_name):
    problem = result["problem"]
    names   = problem["names"]
    Si      = result["Si"]
    fig, axes = plt.subplots(len(OUTPUT_NAMES), 1,
                             figsize=(max(10, len(names) * 0.95), len(OUTPUT_NAMES) * 3.8))
    title_suffix = " - q = 1.0" if FIX_Q_1 else (" [second-order]" if SECOND_ORDER else "")
    fig.suptitle(f"Sobol Sensitivity {model_name}{title_suffix}",
                 fontsize=13, fontweight="bold", y=1.01)
    for out_name, ax in zip(OUTPUT_NAMES, axes):
        s = Si[out_name]
        c1, c2 = COLORS[out_name]
        _bar_pair(ax, names, s["S1"], s["S1_conf"], s["ST"], s["ST_conf"],
                  f"Output: {out_name}", c1, c2)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(RESULTS_DIR, f"{model_name}_sobol.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close()

def plot_s2_heatmap(result, model_name):
    if not SECOND_ORDER:
        return
    problem = result["problem"]
    names = problem["names"]
    n = len(names)
    Si = result["Si"]

    fig, axes = plt.subplots(1, len(OUTPUT_NAMES),
                             figsize=(len(OUTPUT_NAMES) * max(5, n * 0.55 + 1),
                             max(5, n * 0.55 + 1)))
    fig.suptitle(f"S2 Second-Order Sobol Indices — {model_name}" +
                 (" (q = 1.0)" if FIX_Q_1 else ""),
                 fontsize=13, fontweight="bold")
    if len(OUTPUT_NAMES) == 1:
        axes = [axes]

    for ax, out_name in zip(axes, OUTPUT_NAMES):
        S2 = Si[out_name]["S2"]
        # SALib only fills upper triangle, symmetrise for display
        S2_sym = np.where(np.isnan(S2), S2.T, S2)
        np.fill_diagonal(S2_sym, np.nan)
        im = ax.imshow(S2_sym, vmin=0, vmax=1, cmap="YlOrRd", aspect="auto")
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(names, fontsize=8)
        ax.set_title(f"{out_name}", fontsize=10, fontweight="bold")
        for i in range(n):
            for j in range(n):
                val = S2_sym[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                            fontsize=6, color="black" if val < 0.5 else "white")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="S2")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(RESULTS_DIR, f"{model_name}_sobol_S2.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close()

def plot_comparison(result_base, result_ps):
    shared_names = SHARED_PARAMS["names"]
    n_shared = len(shared_names)
    si_base = result_base["Si"]
    si_ps = result_ps["Si"]
    x = np.arange(n_shared)
    w = 0.35

    fig = plt.figure(figsize=(16, len(OUTPUT_NAMES) * 3.5))
    fig.suptitle("Shared Parameters: BaseModel vs ProductSpaceModel" + (" - q = 1.0" if FIX_Q_1 else ""), fontsize=13, fontweight="bold")
    gs = gridspec.GridSpec(len(OUTPUT_NAMES), 2, figure=fig, hspace=0.6, wspace=0.35)

    for k, out_name in enumerate(OUTPUT_NAMES):
        sb = si_base[out_name]
        sp = si_ps[out_name]
        for col, (idx_label, base_vals, ps_vals) in enumerate([
            ("S1", sb["S1"][:n_shared], sp["S1"][:n_shared]),
            ("ST", sb["ST"][:n_shared], sp["ST"][:n_shared]),
        ]):
            base_conf = sb[f"{idx_label}_conf"][:n_shared]
            ps_conf = sp[f"{idx_label}_conf"][:n_shared]
            ax = fig.add_subplot(gs[k, col])
            ax.bar(x - w/2, base_vals, w, yerr=base_conf, label="BaseModel", color="#4C72B0", capsize=4, alpha=0.85)
            ax.bar(x + w/2, ps_vals, w, yerr=ps_conf, label="ProductSpaceModel", color="#DD8452", capsize=4, alpha=0.85)
            ax.axhline(0, color="black", linewidth=0.6, linestyle="--")
            ax.set_xticks(x)
            ax.set_xticklabels(shared_names, rotation=35, ha="right", fontsize=8)
            ax.set_title(f"{out_name}  {idx_label}", fontsize=9, fontweight="bold")
            ax.set_ylabel(f"Sobol {idx_label}")
            ax.set_ylim(-0.1, 1.1)
            ax.legend(fontsize=7)
            ax.spines[["top", "right"]].set_visible(False)

    path = os.path.join(RESULTS_DIR, "comparison_shared_params.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {path}")
    plt.close()


def plot_ps_only(result_ps):
    n_shared = len(SHARED_PARAMS["names"])
    ps_names = PS_ONLY_PARAMS["names"]
    si_ps = result_ps["Si"]
    fig, axes = plt.subplots(len(OUTPUT_NAMES), 1, figsize=(max(8, len(ps_names) * 1.3), len(OUTPUT_NAMES) * 3.5))
    fig.suptitle("ProductSpaceModel  PS-Only Parameter Sensitivity" + (" - q = 1.0" if FIX_Q_1 else ""), fontsize=13, fontweight="bold")
    for out_name, ax in zip(OUTPUT_NAMES, axes):
        sp  = si_ps[out_name]
        c1, c2 = COLORS[out_name]
        _bar_pair(ax, ps_names, sp["S1"][n_shared:], sp["S1_conf"][n_shared:],
                  sp["ST"][n_shared:], sp["ST_conf"][n_shared:], f"Output: {out_name}", c1, c2)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(RESULTS_DIR, "ps_only_params.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close()


def plot_output_distributions(result_base, result_ps):
    fig, axes = plt.subplots(len(OUTPUT_NAMES), 2, figsize=(12, len(OUTPUT_NAMES) * 3.0))
    fig.suptitle("Output Distributions Across Sobol Samples" + (" - q = 1.0" if FIX_Q_1 else ""), fontsize=12, fontweight="bold")
    for k, out_name in enumerate(OUTPUT_NAMES):
        for col, (name, res) in enumerate([("BaseModel", result_base), ("ProductSpaceModel", result_ps)]):
            ax = axes[k, col]
            vals = res["Y"][:, k]
            vals = vals[~np.isnan(vals)]
            c1, _ = COLORS[out_name]
            ax.hist(vals, bins=30, color=c1, alpha=0.85, edgecolor="white")
            ax.axvline(np.median(vals), color="black", linestyle="--", linewidth=1.2,
                       label=f"median = {np.median(vals):.2f}")
            ax.set_title(f"{name}  {out_name}", fontsize=9, fontweight="bold")
            ax.set_xlabel(out_name)
            ax.set_ylabel("Count")
            ax.legend(fontsize=7)
            ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "output_distributions.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close()


def save_indices_csv(result, model_name):
    path = os.path.join(RESULTS_DIR, f"{model_name}_sobol_indices.csv")
    names = result["problem"]["names"]
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["output", "parameter", "S1", "S1_conf", "ST", "ST_conf"])
        for out_name in OUTPUT_NAMES:
            Si = result["Si"][out_name]
            for j, pname in enumerate(result["problem"]["names"]):
                writer.writerow([out_name, pname,
                                  round(Si["S1"][j], 5), round(Si["S1_conf"][j], 5),
                                  round(Si["ST"][j], 5), round(Si["ST_conf"][j], 5)])
        if SECOND_ORDER:
            writer.writerow([])
            writer.writerow(["output", "param_i", "param_j", "S2", "S2_conf"])
            for out_name in OUTPUT_NAMES:
                Si = result["Si"][out_name]
                for i, pi in enumerate(names):
                    for j, pj in enumerate(names):
                        if j > i:
                            writer.writerow([out_name, pi, pj,
                                              round(Si["S2"][i, j], 5),
                                              round(Si["S2_conf"][i, j], 5)])
    print(f"Saved: {path}")


# Run analysis

if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print("HYSTERESIS SENSITIVITY ANALYSIS")
    print(f"N_SOBOL       = {N_SOBOL}")
    print(f"N_JOBS        = {N_JOBS}")
    print(f"SECOND_ORDER  = {SECOND_ORDER}")
    print(f"d_C           = {D_C_MIN} to {D_C_MAX}, step={D_C_STEP}")
    print(f"T_INIT        = {T_INIT},  T_STEP = {T_STEP}")
    pv_base = saltelli.sample(PROBLEM_BASE, N_SOBOL, calc_second_order=SECOND_ORDER)
    pv_ps = saltelli.sample(PROBLEM_PS,   N_SOBOL, calc_second_order=SECOND_ORDER)
    print(f"BaseModel runs     : {len(pv_base)}")
    print(f"ProdSpaceModel runs: {len(pv_ps)}")

    result_base = run_sobol("BaseModel", PROBLEM_BASE, _evaluate_base, n=N_SOBOL)
    plot_single_model(result_base, "BaseModel")
    if SECOND_ORDER:
        plot_s2_heatmap(result_base, "BaseModel")
    save_indices_csv(result_base, "BaseModel")

    result_ps = run_sobol("ProductSpaceModel", PROBLEM_PS, _evaluate_ps, n=N_SOBOL)
    plot_single_model(result_ps, "ProductSpaceModel")
    if SECOND_ORDER:
        plot_s2_heatmap(result_ps, "ProductSpaceModel")
    save_indices_csv(result_ps, "ProductSpaceModel")

    plot_comparison(result_base, result_ps)
    plot_ps_only(result_ps)
    plot_output_distributions(result_base, result_ps)

    print(f"All results saved to: {RESULTS_DIR}/")
    print("Done.")