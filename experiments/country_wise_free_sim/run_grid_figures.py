"""
Produce 4 grid figures (19 panels each, 4x5) summarising the alpha-frozen
calibration results:

  1. trajectory_C_conditioned.png    — C(country share) obs vs sim, conditioned mode
  2. trajectory_C_alpha_frozen.png   — C(country share) obs vs sim, alpha-frozen mode
  3. alpha_top10_conditioned.png     — top-10 alpha trajectories, obs vs sim, conditioned
  4. alpha_top10_alpha_frozen.png    — top-10 alpha trajectories, observed only
                                       (sim alpha equals obs by construction)

Outputs land in experiments/country_wise_free_sim/simulation_alpha_frozen_calib/.

Reuses the per-year trajectory cache under experiments/country_wise_free_sim/_cache/.
Pass --fresh to ignore the cache and recompute every country.
"""

import argparse
import os
import sys
import time

import numpy as np
import matplotlib.pyplot as plt

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)

from run_free_sim import (  # noqa: E402
    DEFAULT_COUNTRY_DIR,
    DEFAULT_R_VALUES_PATH,
    PER_COUNTRY_TIMEOUT_S,
    SIM_START_YEAR,
    YEAR_END,
    YEARS,
    _alpha_frozen_worker,
    _purge_cache_family,
    _conditioned_worker,
    _load_traj_cache,
    _run_with_timeout,
    load_country_index,
    load_growth_regression_r,
    load_products_index,
    set_simulation_sources,
    set_year_range,
    load_theta,
    plot_trajectory_fit,
    write_inclusion_stats_csv,
)
import run_free_sim as _rfs  # for reading the live SIM_START_YEAR / YEAR_END after override
from calibration_utils import load_data  # noqa: E402

OUT_DIR = os.path.join(THIS_DIR, "simulation_alpha_frozen_calib")


def _missing_expected_years(traj, expected_years):
    if traj is None:
        return list(expected_years)
    have = set(traj.keys())
    return [year for year in expected_years if year not in have]


def run_per_country(
    mode_label,
    worker,
    rows,
    data,
    code_to_idx,
    fresh=False,
    timeout_s=PER_COUNTRY_TIMEOUT_S,
    expected_years=YEARS,
):
    expected_years = list(expected_years)
    out = {}
    for row in rows:
        code = row["location_code"]
        theta = load_theta(code)
        if theta is None:
            print(f"[skip] {code}: no/invalid best_theta.npy", flush=True)
            continue
        if fresh:
            _purge_cache_family(mode_label, code)
        else:
            cached = _load_traj_cache(mode_label, code, theta)
            if cached is not None:
                missing = _missing_expected_years(cached, expected_years)
                if not missing:
                    print(f"[{code}] using cached {mode_label} ({len(cached)} yrs)",
                          flush=True)
                    out[code] = cached
                    continue
                first_missing = missing[0]
                last_cached = max(cached)
                print(
                    f"[{code}] cached {mode_label} is incomplete "
                    f"({len(cached)}/{len(expected_years)} yrs, "
                    f"last={last_cached}, first missing={first_missing}); recomputing",
                    flush=True,
                )
                _purge_cache_family(mode_label, code)
        t0 = time.time()
        print(f"[{code}] running {mode_label}...", flush=True)
        _run_with_timeout(
            target=worker,
            args=(theta, data, code_to_idx[code], mode_label, code),
            timeout_s=timeout_s,
            label=f"[{code}] {mode_label}",
        )
        traj = _load_traj_cache(mode_label, code, theta)
        if traj is None:
            print(f"[{code}] no trajectory produced", flush=True)
            continue
        missing = _missing_expected_years(traj, expected_years)
        if missing:
            print(
                f"[{code}] WARNING: {mode_label} trajectory still incomplete "
                f"({len(traj)}/{len(expected_years)} yrs, last={max(traj)}). "
                f"Increase --timeout-s or inspect the solver for this country.",
                flush=True,
            )
        out[code] = traj
        print(f"[{code}] done in {time.time()-t0:.1f}s ({len(traj)} yrs)",
              flush=True)
    return out


def plot_alpha_grid(per_country, rows, data, out_path, mode_label,
                    show_sim=True, top_k=10):
    products = load_products_index()

    def prod_label(i):
        r = products.get(i)
        if r is None:
            return str(i)
        return r.get("product_name_short") or r.get("hs_product_code") or str(i)

    obs_years = sorted(data["alpha_obs"].keys())
    last_obs = max(obs_years)

    n_cols, n_rows = 5, 4
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(22, 14), sharex=True)
    cmap = plt.get_cmap("tab10")

    for idx, row in enumerate(rows):
        ax = axes.flat[idx]
        ci = row["position"]
        code = row["location_code"]
        traj = per_country.get(code)

        alpha_last = data["alpha_obs"][last_obs][ci]
        top_idx = np.argsort(alpha_last)[::-1][:top_k]

        sim_years = sorted(traj.keys()) if (show_sim and traj is not None) else None

        for k, pi in enumerate(top_idx):
            color = cmap(k % 10)
            obs_series = [data["alpha_obs"][y][ci, pi] for y in obs_years]
            ax.plot(obs_years, obs_series, color=color, lw=1.2,
                    label=prod_label(pi))
            if sim_years is not None:
                sim_series = [traj[y]["alpha"][ci, pi] for y in sim_years]
                ax.plot(sim_years, sim_series, color=color, lw=1.0, ls="--",
                        alpha=0.85)

        ax.set_title(f"{row['country_name_short']} ({code})", fontsize=9,
                     fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=5, ncol=2, frameon=False, loc="upper left")

    for idx in range(len(rows), n_rows * n_cols):
        axes.flat[idx].set_visible(False)

    if show_sim:
        subtitle = "observed (solid) vs simulated (dashed)"
    else:
        subtitle = "observed only — simulated alpha = observed by construction"
    fig.suptitle(
        f"Top-{top_k} alphas at {last_obs} — {mode_label}\n{subtitle}",
        fontsize=13, fontweight="bold",
    )
    fig.supxlabel("Year", fontsize=11)
    fig.supylabel(r"$\alpha$ (specialisation share)", fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--params-dir",
        default=DEFAULT_COUNTRY_DIR,
        help=f"Directory holding per-country best_theta.npy files. Default: {DEFAULT_COUNTRY_DIR}",
    )
    parser.add_argument(
        "--r-values",
        default=DEFAULT_R_VALUES_PATH,
        help=f"Growth-regression CSV used to seed r_C. Default: {DEFAULT_R_VALUES_PATH}",
    )
    parser.add_argument(
        "--fresh", action="store_true",
        help="Ignore the trajectory cache and recompute every country.",
    )
    parser.add_argument(
        "--timeout-s",
        type=float,
        default=max(PER_COUNTRY_TIMEOUT_S, 600.0),
        help=(
            "Wall-clock budget per country/mode. Partial caches are retried "
            "automatically. Default: 600."
        ),
    )
    parser.add_argument(
        "--out-dir",
        default=OUT_DIR,
        help=f"Where to save the four grid figures. Default: {OUT_DIR}",
    )
    parser.add_argument(
        "--start-year", type=int, default=SIM_START_YEAR,
        help=f"First simulation year (initial state). Default: {SIM_START_YEAR}",
    )
    parser.add_argument(
        "--end-year", type=int, default=YEAR_END,
        help=f"Last simulation year. Default: {YEAR_END}",
    )
    args = parser.parse_args()

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    set_simulation_sources(args.params_dir, args.r_values)
    set_year_range(args.start_year, args.end_year)
    print(f"Params dir: {args.params_dir}", flush=True)
    print(f"R-values:   {args.r_values}", flush=True)
    print(f"Years:      {_rfs.SIM_START_YEAR}..{_rfs.YEAR_END}", flush=True)

    print("Loading empirical data...", flush=True)
    data = load_data()
    rows = load_country_index()
    data["r_C_growth_regression"] = load_growth_regression_r(rows)
    code_to_idx = {r["location_code"]: r["position"] for r in rows}

    print("\n=== conditioned per-country ===", flush=True)
    cond = run_per_country("conditioned", _conditioned_worker, rows, data,
                           code_to_idx, fresh=args.fresh,
                           timeout_s=args.timeout_s)

    print("\n=== alpha-frozen per-country ===", flush=True)
    frozen = run_per_country("alpha_frozen", _alpha_frozen_worker, rows, data,
                             code_to_idx, fresh=args.fresh,
                             timeout_s=args.timeout_s)

    plot_trajectory_fit(
        cond, rows, data,
        os.path.join(out_dir, "trajectory_C_conditioned.png"),
        mode_label="conditioned",
    )
    plot_trajectory_fit(
        frozen, rows, data,
        os.path.join(out_dir, "trajectory_C_alpha_frozen.png"),
        mode_label="alpha-frozen",
    )
    write_inclusion_stats_csv(
        frozen, rows, data,
        os.path.join(out_dir, "inclusion_stats_alpha_frozen.csv"),
    )
    plot_alpha_grid(
        cond, rows, data,
        os.path.join(out_dir, "alpha_top10_conditioned.png"),
        mode_label="conditioned", show_sim=True,
    )
    plot_alpha_grid(
        frozen, rows, data,
        os.path.join(out_dir, "alpha_top10_alpha_frozen.png"),
        mode_label="alpha-frozen", show_sim=False,
    )


if __name__ == "__main__":
    main()
