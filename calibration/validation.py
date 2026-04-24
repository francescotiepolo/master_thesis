"""
Phase 3: Validation and diagnostics.

  1. Best-point evaluation -- NSGA-II result point on calibration + validation periods
  2. Trajectory comparison -- observed vs simulated country trajectories across both periods
  3. Fit metrics -- period_metrics.json, yearly_metrics.csv, trajectory_fit.png, trajectory_summary.png
"""

import os
import copy
import json
import numpy as np
import matplotlib.pyplot as plt


try:
    from calibration_config import (
        PARAM_NAMES, N_PARAMS,
        YEAR_START, CALIB_END, CALIB_YEARS, VALID_YEARS,
        NSGA2_DIR, VAL_DIR,
    )
    from calibration_utils import (
        load_data, simulate, _windowed_simulate, compute_stats, aggregate_loss_components,
        trajectory_correlation,
    )
except ModuleNotFoundError as exc:
    if exc.name not in {"calibration_config", "calibration_utils"}:
        raise
    from .calibration_config import (
        PARAM_NAMES, N_PARAMS,
        YEAR_START, CALIB_END, CALIB_YEARS, VALID_YEARS,
        NSGA2_DIR, VAL_DIR,
    )
    from .calibration_utils import (
        load_data, simulate, _windowed_simulate, compute_stats, aggregate_loss_components,
        trajectory_correlation,
    )


# Shared simulation helper

def _make_window_data(data, start_year):
    """
    Return a data dict initialised at start_year when those observations exist.
    """
    if start_year == YEAR_START:
        return data, YEAR_START

    required = ("alpha_obs", "C_obs", "P_obs")
    missing = [name for name in required if start_year not in data[name]]
    if missing:
        print(f"  WARNING: Missing observed initial state for year {start_year}; using {YEAR_START}.")
        return data, YEAR_START

    window_data = copy.deepcopy(data)
    window_data["alpha_init"] = data["alpha_obs"][start_year].copy()
    window_data["C_init"] = data["C_obs"][start_year].copy()
    window_data["P_init"] = data["P_obs"][start_year].copy()
    return window_data, start_year


def _vector_error_metrics(sim_vec, obs_vec):
    """
    Compute RMSE, MAE, and R² between two vectors.
    """
    diff = sim_vec - obs_vec
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    mae = float(np.mean(np.abs(diff)))
    ss_res = float(np.sum(diff ** 2))
    ss_tot = float(np.sum((obs_vec - obs_vec.mean()) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
    }


def _collect_period_metrics(label, years, traj, data, year_start):
    """
    Compute all fit metrics for a period, including per-year stats.
    """
    period_metrics = aggregate_loss_components(
        traj, data, years=years, year_start=year_start
    )
    if period_metrics is None:
        return None, []

    sim_c_all, obs_c_all = [], []
    sim_p_all, obs_p_all = [], []
    yearly_rows = []

    for yr in years:
        if yr not in traj or yr not in data["alpha_obs"]:
            continue
        alpha_s, C_s, P_s = traj[yr]
        alpha_o = data["alpha_obs"][yr]
        C_o = data["C_obs"][yr]
        P_o = data["P_obs"][yr]
        stats = compute_stats(alpha_s, C_s, alpha_o, C_o, P_sim=P_s, P_obs=P_o)
        share_sim = stats["share_sim"]
        share_obs = stats["share_obs"]
        sum_P_s = P_s.sum()
        sum_P_o = P_o.sum()
        share_P_sim = P_s / sum_P_s if sum_P_s > 0 else P_s
        share_P_obs = P_o / sum_P_o if sum_P_o > 0 else P_o
        c_err = _vector_error_metrics(share_sim, share_obs)
        p_err = _vector_error_metrics(share_P_sim, share_P_obs)

        sim_c_all.append(share_sim.ravel())
        obs_c_all.append(share_obs.ravel())
        sim_p_all.append(share_P_sim.ravel())
        obs_p_all.append(share_P_obs.ravel())

        yearly_rows.append({
            "period": label,
            "year": int(yr),
            "nrmse_C": float(stats["nrmse_C"]),
            "nrmse_P": float(stats["nrmse_P"]),
            "rank_products": float(stats["rank_products"]),
            "rho_products": float(stats["rho_products"]),
            "rmse_C": c_err["rmse"],
            "mae_C": c_err["mae"],
            "r2_C": c_err["r2"],
            "rmse_P": p_err["rmse"],
            "mae_P": p_err["mae"],
            "r2_P": p_err["r2"],
        })

    if not yearly_rows:
        return period_metrics, []

    c_metrics = _vector_error_metrics(
        np.concatenate(sim_c_all), np.concatenate(obs_c_all)
    )
    p_metrics = _vector_error_metrics(
        np.concatenate(sim_p_all), np.concatenate(obs_p_all)
    )
    period_metrics = {
        **period_metrics,
        "rmse_C": c_metrics["rmse"],
        "mae_C": c_metrics["mae"],
        "r2_C": c_metrics["r2"],
        "rmse_P": p_metrics["rmse"],
        "mae_P": p_metrics["mae"],
        "r2_P": p_metrics["r2"],
    }
    return period_metrics, yearly_rows


def _write_yearly_metrics_csv(rows, path):
    """
    Write per-year fit metrics (from _collect_period_metrics) to a CSV file.
    """
    headers = [
        "period", "year",
        "nrmse_C", "nrmse_P", "rank_products", "rho_products",
        "rmse_C", "mae_C", "r2_C",
        "rmse_P", "mae_P", "r2_P",
    ]
    with open(path, "w", encoding="ascii") as f:
        f.write(",".join(headers) + "\n")
        for row in rows:
            values = []
            for key in headers:
                val = row[key]
                if isinstance(val, str):
                    values.append(val)
                elif key == "year":
                    values.append(str(int(val)))
                else:
                    values.append(f"{float(val):.8f}")
            f.write(",".join(values) + "\n")


def _stack_series(traj, data, years, which):
    """
    Stack observed and simulated shares for a given period and output type (C or P).
    """
    obs_rows, sim_rows, valid_years = [], [], []
    for yr in years:
        if yr not in traj or yr not in data["alpha_obs"]:
            continue
        alpha_s, C_s, P_s = traj[yr]
        if which == "C":
            s = C_s.sum(); o = data["C_obs"][yr].sum()
            sim_rows.append(C_s / s if s > 0 else C_s)
            obs_rows.append(data["C_obs"][yr] / o if o > 0 else data["C_obs"][yr])
        else:
            s = P_s.sum(); o = data["P_obs"][yr].sum()
            sim_rows.append(P_s / s if s > 0 else P_s)
            obs_rows.append(data["P_obs"][yr] / o if o > 0 else data["P_obs"][yr])
        valid_years.append(yr)
    return np.array(valid_years), np.vstack(obs_rows), np.vstack(sim_rows)


def _load_country_labels():
    """
    Load country short names from extracted_data/countries_index.csv.
    """
    csv_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "extracted_data", "countries_index.csv"
    )
    if not os.path.exists(csv_path):
        return None
    import csv
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        labels = {int(row["position"]): row["country_name_short"] for row in reader}
    return labels


def plot_observed_vs_simulated(traj_calib, traj_valid, data):
    """
    Generate trajectory_fit.png: observed vs simulated market share time series for each country,
    with Spearman correlation annotations for calibration and validation periods.
    Also generates trajectory_summary.png: median trajectory across all countries.
    """
    country_labels = _load_country_labels()
    SC = data["SC"]

    # Collect per-country share time series across both periods
    all_years_calib = sorted(yr for yr in CALIB_YEARS if yr in traj_calib)
    all_years_valid = sorted(yr for yr in VALID_YEARS if yr in traj_valid)

    obs_calib, sim_calib = {}, {}
    for yr in all_years_calib:
        _, C_sim, _ = traj_calib[yr]
        C_obs = data["C_obs"][yr]
        s_sim = C_sim / C_sim.sum() if C_sim.sum() > 0 else C_sim
        s_obs = C_obs / C_obs.sum() if C_obs.sum() > 0 else C_obs
        for j in range(SC):
            obs_calib.setdefault(j, []).append(s_obs[j])
            sim_calib.setdefault(j, []).append(s_sim[j])

    obs_valid, sim_valid = {}, {}
    for yr in all_years_valid:
        _, C_sim, _ = traj_valid[yr]
        C_obs = data["C_obs"][yr]
        s_sim = C_sim / C_sim.sum() if C_sim.sum() > 0 else C_sim
        s_obs = C_obs / C_obs.sum() if C_obs.sum() > 0 else C_obs
        for j in range(SC):
            obs_valid.setdefault(j, []).append(s_obs[j])
            sim_valid.setdefault(j, []).append(s_sim[j])

    # Per-country Spearman for annotation
    from calibration_utils import spearman
    rho_calib = {}
    rho_valid = {}
    for j in range(SC):
        rho_calib[j] = spearman(np.array(sim_calib[j]), np.array(obs_calib[j]))
        rho_valid[j] = spearman(np.array(sim_valid[j]), np.array(obs_valid[j]))

    n_cols = 5
    n_rows = 4
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 12), sharex=True)

    for j in range(SC):
        row, col = divmod(j, n_cols)
        ax = axes[row, col]
        label = country_labels[j] if country_labels and j in country_labels else f"Country {j}"

        # Calibration period
        ax.plot(all_years_calib, obs_calib[j], color="#1F4E79", lw=1.5, label="Observed")
        ax.plot(all_years_calib, sim_calib[j], color="#B5541E", lw=1.5, label="Simulated")

        # Validation period
        ax.plot(all_years_valid, obs_valid[j], color="#1F4E79", lw=1.5, ls="--")
        ax.plot(all_years_valid, sim_valid[j], color="#B5541E", lw=1.5, ls="--")

        # Calibration/validation boundary
        ax.axvline(CALIB_END, color="grey", ls=":", lw=0.8, alpha=0.7)

        ax.set_title(f"{label}", fontsize=9, fontweight="bold")
        ax.annotate(
            f"cal={rho_calib[j]:.2f}  val={rho_valid[j]:.2f}",
            xy=(0.03, 0.95), xycoords="axes fraction", fontsize=7,
            va="top", color="grey",
        )
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=7)

    # Hide unused subplot
    for idx in range(SC, n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row, col].set_visible(False)

    # Shared legend and labels
    axes[0, 0].legend(fontsize=7, frameon=False)
    fig.supxlabel("Year", fontsize=11)
    fig.supylabel("Market share", fontsize=11)
    fig.suptitle("Country trajectories: observed vs simulated", fontsize=13, fontweight="bold")
    plt.tight_layout()
    out_path = os.path.join(VAL_DIR, "trajectory_fit.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: trajectory_fit.png")

    # Summary plot: mean and median across all countries
    fig, ax = plt.subplots(figsize=(10, 5))

    obs_all = np.array([obs_calib[j] + obs_valid[j] for j in range(SC)])  # (SC, T)
    sim_all = np.array([sim_calib[j] + sim_valid[j] for j in range(SC)])  # (SC, T)
    all_years = all_years_calib + all_years_valid

    # Individual country traces
    for j in range(SC):
        ax.plot(all_years, obs_all[j], color="#4C72B0", alpha=0.08, lw=0.7)
        ax.plot(all_years, sim_all[j], color="#DD8452", alpha=0.08, lw=0.7)

    ax.plot(all_years, np.median(obs_all, axis=0), color="#1F4E79", lw=2.5, label="Observed median")
    ax.plot(all_years, np.median(sim_all, axis=0), color="#B5541E", lw=2.5, label="Simulated median")
    ax.axvline(CALIB_END, color="grey", ls=":", lw=1, alpha=0.7, label="Calibration boundary")
    ax.set_xlabel("Year")
    ax.set_ylabel("Market share")
    ax.set_title("Country market shares: all trajectories", fontsize=12, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=9, frameon=False)
    plt.tight_layout()
    out_path = os.path.join(VAL_DIR, "trajectory_summary.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: trajectory_summary.png")


# Best-point evaluation

def evaluate_best_point(data):
    """
    Evaluate the NSGA-II compromise point on calibration and validation periods,
    printing per-component loss breakdowns.
    """
    nsga2_path = os.path.join(NSGA2_DIR, "best_theta.npy")
    if not os.path.exists(nsga2_path):
        print("  No NSGA-II best_theta.npy found -- run `python calibration/run_pipeline.py nsga2` first.")
        return None
    best_path = nsga2_path
    print("  Using NSGA-II compromise point.")
    theta = np.load(best_path)
    if len(theta) != N_PARAMS:
        print(f"  ERROR: best_theta.npy has {len(theta)} parameters but N_PARAMS={N_PARAMS}. "
              f"Stale artifact -- re-run the optimizer to generate a compatible best_theta.npy.")
        return None

    print(f"Best-point evaluation (NSGA-II compromise point):")
    for name, val in zip(PARAM_NAMES, theta):
        print(f"  {name:20s} = {val:.5f}")

    results = {}
    yearly_rows = []

    # Calibration period (1988-2010)
    print(f"\n  Calibration period ({CALIB_YEARS[0]}-{CALIB_YEARS[-1]}):")
    traj_calib = _windowed_simulate(theta, data, CALIB_YEARS)
    if traj_calib is None:
        print("    FAILED: simulation returned None")
    else:
        agg, rows = _collect_period_metrics(
            "calibration", CALIB_YEARS, traj_calib, data, YEAR_START
        )
        if agg is not None:
            results["calibration"] = agg
            yearly_rows.extend(rows)
            print(f"    Total loss   : {agg['loss_total']:.5f}")
            for k, v in agg.items():
                if k != "loss_total":
                    print(f"    {k:15s}: {v:.5f}")

            print(f"\n    Per-year stats:")
            print(f"    {'Year':>6s}  {'nrmse_C':>8s}  {'nrmse_P':>8s}  {'rank_prod':>10s}")
            for yr in CALIB_YEARS:
                if yr not in traj_calib or yr not in data["alpha_obs"]:
                    continue
                alpha_s, C_s, P_s = traj_calib[yr]
                P_o = data["P_obs"].get(yr)
                s = compute_stats(alpha_s, C_s, data["alpha_obs"][yr], data["C_obs"][yr],
                                  P_sim=P_s, P_obs=P_o)
                print(f"    {yr:6d}  {s['nrmse_C']:8.4f}  {s['nrmse_P']:8.4f}  {s['rank_products']:10.4f}")

    # Validation period (2010-2024)
    print(f"\n  Validation period ({VALID_YEARS[0]}-{VALID_YEARS[-1]}):")
    data_oos, start_year = _make_window_data(data, CALIB_END)
    traj_valid = simulate(theta, data_oos, VALID_YEARS, start_year=start_year)
    if traj_valid is None:
        print("    FAILED: simulation returned None")
    else:
        agg_v, rows = _collect_period_metrics(
            "validation", VALID_YEARS, traj_valid, data, start_year
        )
        if agg_v is not None:
            results["validation"] = agg_v
            yearly_rows.extend(rows)
            print(f"    Total loss   : {agg_v['loss_total']:.5f}")
            for k, v in agg_v.items():
                if k != "loss_total":
                    print(f"    {k:15s}: {v:.5f}")

            print(f"\n    Per-year stats:")
            print(f"    {'Year':>6s}  {'nrmse_C':>8s}  {'nrmse_P':>8s}  {'rank_prod':>10s}")
            for yr in VALID_YEARS:
                if yr not in traj_valid or yr not in data["alpha_obs"]:
                    continue
                alpha_s, C_s, P_s = traj_valid[yr]
                P_o = data["P_obs"].get(yr)
                s = compute_stats(alpha_s, C_s, data["alpha_obs"][yr], data["C_obs"][yr],
                                  P_sim=P_s, P_obs=P_o)
                print(f"    {yr:6d}  {s['nrmse_C']:8.4f}  {s['nrmse_P']:8.4f}  {s['rank_products']:10.4f}")

    if traj_calib is not None and traj_valid is not None:
        plot_observed_vs_simulated(traj_calib, traj_valid, data)

    with open(os.path.join(VAL_DIR, "period_metrics.json"), "w") as f:
        json.dump(results, f, indent=2)
    _write_yearly_metrics_csv(yearly_rows, os.path.join(VAL_DIR, "yearly_metrics.csv"))

    print(f"\n  Saved: period_metrics.json")
    print(f"  Saved: yearly_metrics.csv")
    return results


if __name__ == "__main__":
    os.makedirs(VAL_DIR, exist_ok=True)
    print("Loading empirical data...")
    data = load_data()

    print("\n1. Best-point evaluation...")
    evaluate_best_point(data)

    print(f"\nAll outputs saved to: {VAL_DIR}/")