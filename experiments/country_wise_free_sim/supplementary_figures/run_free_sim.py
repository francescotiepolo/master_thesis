"""
Simulation experiments using the legacy country-wise calibration results
(calibration/calibration_results/country_wise/<CODE>/best_theta.npy).

Two passes are produced:

1. Per-country *conditioned* simulations.
   For each country we build the model with that country's calibrated theta
   and step the ODE annually. Only the target country evolves freely; the
   other 18 are pinned to observed P, C, alpha at every year boundary
   (P is reconstructed as observed_P minus observed-target contribution
   plus simulated-target contribution, mirroring the legacy calibration).

2. A single joint *free* simulation.
   We pick one reference country's calibrated theta (the one with the
   smallest calibration loss) and run a single full ODE from the 1988
   observed state to 2024 with no anchoring. Every country's trajectory
   here comes out of one consistent simulation.

Outputs (written next to this script):
  - trajectory_fit_conditioned.png  : 4x5 grid of obs vs conditioned sim
  - trajectory_fit_free_joint.png   : 4x5 grid of obs vs single joint free sim
  - alpha_top10_conditioned_<CODE>.png : per-country top-10 alphas (conditioned)
  - alpha_top10_free_joint_<CODE>.png  : per-country top-10 alphas (free joint)
"""

import argparse
import csv
import hashlib
import multiprocessing as mp
import os
import sys
import time
import warnings

import numpy as np
import matplotlib.pyplot as plt

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(THIS_DIR))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "calibration"))
sys.path.insert(0, os.path.join(ROOT, "calibration", "country_wise_calibration"))

from product_space_model import ProductSpaceModel  # noqa: E402
from calibration_config import (  # noqa: E402
    CALIB_DIR,
    EXTRACTED_DIR,
    FIXED,
    MAX_SOLVER_STEPS,
    SIM_STEPS_PER_YEAR,
    YEAR_START,
)
from calibration_utils import _patch_model, _update_growth_rates, load_data  # noqa: E402
from calibration_country_wise import (  # noqa: E402
    COUNTRY_PARAM_NAMES as LEGACY_PARAM_NAMES,
    build_country_model as _build_country_model,
)

# Paths default to the alpha-frozen country-wise calibration (6-param schema +
# globals from joint/bootstrap_globals.json + r_C from r_values_fixed.csv).
# Override with --params-dir / --r-values to point at a different calibration.
DEFAULT_COUNTRY_DIR = os.path.join(CALIB_DIR, "country_wise_alpha_frozen")
GROWTH_REG_DIR = os.path.join(CALIB_DIR, "growth_regression")
DEFAULT_R_VALUES_PATH = os.path.join(GROWTH_REG_DIR, "r_values_fixed.csv")

# These are mutated by main() when CLI flags override them.
LEGACY_COUNTRY_DIR = DEFAULT_COUNTRY_DIR
R_VALUES_PATH = DEFAULT_R_VALUES_PATH
SUMMARY_PATH = os.path.join(LEGACY_COUNTRY_DIR, "summary.csv")

SIM_START_YEAR = YEAR_START  # 1988, matches calibration
YEAR_END = 2024
# Skip 1993/1994 — 0% data coverage; calibration ignores them via CALIB_YEARS.
SKIP_YEARS = {1993, 1994}
YEARS = [y for y in range(SIM_START_YEAR, YEAR_END + 1) if y not in SKIP_YEARS]

PER_COUNTRY_TIMEOUT_S = 60.0
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_cache")
CACHE_SCHEMA_VERSION = "v2"


def set_simulation_sources(params_dir, r_values_path):
    """Update module-level input paths used by loaders and cache keys."""
    global LEGACY_COUNTRY_DIR, SUMMARY_PATH, R_VALUES_PATH
    LEGACY_COUNTRY_DIR = params_dir
    SUMMARY_PATH = os.path.join(LEGACY_COUNTRY_DIR, "summary.csv")
    R_VALUES_PATH = r_values_path


def _cache_key(mode_label, code, theta):
    hasher = hashlib.sha1()
    hasher.update(CACHE_SCHEMA_VERSION.encode("utf-8"))
    hasher.update(mode_label.encode("utf-8"))
    hasher.update(code.encode("utf-8"))
    hasher.update(os.path.abspath(LEGACY_COUNTRY_DIR).encode("utf-8"))
    hasher.update(os.path.abspath(R_VALUES_PATH).encode("utf-8"))
    try:
        stat = os.stat(R_VALUES_PATH)
        hasher.update(str(stat.st_mtime_ns).encode("utf-8"))
        hasher.update(str(stat.st_size).encode("utf-8"))
    except OSError:
        pass
    hasher.update(np.asarray(theta, dtype=float).tobytes())
    return hasher.hexdigest()[:12]


def _cache_path(mode_label, code, theta):
    key = _cache_key(mode_label, code, theta)
    return os.path.join(CACHE_DIR, f"{mode_label}_{code}_{key}.npz")


def _purge_cache_family(mode_label, code):
    if not os.path.isdir(CACHE_DIR):
        return
    prefix = f"{mode_label}_{code}"
    for name in os.listdir(CACHE_DIR):
        if name.startswith(prefix) and name.endswith(".npz"):
            os.remove(os.path.join(CACHE_DIR, name))


def _save_traj_cache(mode_label, code, theta, traj):
    os.makedirs(CACHE_DIR, exist_ok=True)
    years = sorted(traj.keys())
    if not years:
        return
    SP = traj[years[0]]["P"].size
    SC = traj[years[0]]["C"].size
    P = np.stack([traj[y]["P"] for y in years])
    C = np.stack([traj[y]["C"] for y in years])
    alpha = np.stack([traj[y]["alpha"] for y in years])
    final_path = _cache_path(mode_label, code, theta)
    tmp_path = final_path + f".tmp.{os.getpid()}"
    np.savez_compressed(
        tmp_path,
        years=np.asarray(years, dtype=int),
        P=P, C=C, alpha=alpha, SP=SP, SC=SC,
    )
    # numpy adds .npz; rename whatever exists to the final path atomically.
    written = tmp_path if os.path.exists(tmp_path) else tmp_path + ".npz"
    os.replace(written, final_path)


def _load_traj_cache(mode_label, code, theta):
    path = _cache_path(mode_label, code, theta)
    if not os.path.exists(path):
        return None
    z = np.load(path)
    years = z["years"].tolist()
    out = {}
    for k, y in enumerate(years):
        out[int(y)] = {
            "P": z["P"][k].copy(),
            "C": z["C"][k].copy(),
            "alpha": z["alpha"][k].copy(),
        }
    return out


def load_country_index():
    rows = []
    with open(os.path.join(EXTRACTED_DIR, "countries_index.csv"), newline="") as fh:
        for r in csv.DictReader(fh):
            r["position"] = int(r["position"])
            rows.append(r)
    return rows


def load_products_index():
    path = os.path.join(EXTRACTED_DIR, "products_index.csv")
    out = {}
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            out[int(r["position"])] = r
    return out


def load_growth_regression_r(country_rows):
    by = {}
    with open(R_VALUES_PATH, newline="") as fh:
        for r in csv.DictReader(fh):
            by[r["country"]] = float(r["r_i"])
    return np.array(
        [by[r["location_code"]] for r in country_rows], dtype=float
    )


def best_loss_country():
    """Return the country code with the smallest best_loss in summary.csv."""
    best_code, best_val = None, np.inf
    with open(SUMMARY_PATH, newline="") as fh:
        for r in csv.DictReader(fh):
            try:
                v = float(r["best_loss"])
            except (TypeError, ValueError):
                continue
            if v < best_val:
                best_val = v
                best_code = r["country_code"]
    return best_code, best_val


def build_model(theta, data, country_idx=0):
    """
    Build a ProductSpaceModel from a 6-param theta using the same recipe as
    the country-wise calibration: per-country params from theta, globals from
    joint/bootstrap_globals.json, r_C from r_values_fixed.csv (loaded into
    data["r_C_growth_regression"]). country_idx is accepted for API symmetry
    with build_country_model but is currently unused by the model construction.
    """
    return _build_country_model(theta, data, country_idx)


def _normalize_row(row, fallback):
    row = np.clip(np.asarray(row, dtype=float), 0.0, None)
    s = row.sum()
    if s > 0:
        return row / s
    fb = np.clip(np.asarray(fallback, dtype=float), 0.0, None)
    fs = fb.sum()
    return fb / fs if fs > 0 else fb


def _solve_one_year(model, y0, n_yr):
    """Step the ODE forward by n_yr years; return (P, C, alpha) at the new year."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.solve(
                t_end=float(n_yr),
                d_C=float(FIXED.get("d_C", 0.0)),
                n_steps=SIM_STEPS_PER_YEAR * n_yr,
                y0=y0,
                max_solver_steps=MAX_SOLVER_STEPS,
                rtol=1e-3,
                atol=1e-6,
            )
    except Exception as exc:
        return None, f"solver exception: {exc}"

    if model.y is None or model.y.shape[1] == 0:
        return None, "empty solution"

    P_sim = np.maximum(model.y[:model.SP, -1], 0.0)
    C_sim = np.maximum(model.y[model.SP:model.N, -1], 0.0)
    alpha_sim = np.clip(
        model.y_partial[:, -1].reshape(model.SC, model.SP), 0.0, 1.0,
    )
    if not (np.all(np.isfinite(P_sim)) and np.all(np.isfinite(C_sim))):
        return None, "non-finite state"
    return (P_sim, C_sim, alpha_sim), None


def simulate_conditioned(theta, data, country_idx, save_cb=None):
    """
    Simulate one country freely while pinning the other 18 to observed values
    at every year boundary.
    """
    model = build_model(theta, data, country_idx)
    P = data["P_obs"][SIM_START_YEAR].astype(float).copy()
    C = data["C_obs"][SIM_START_YEAR].astype(float).copy()
    alpha = data["alpha_obs"][SIM_START_YEAR].astype(float).copy()
    y0 = np.concatenate([P, C, alpha.flatten()])

    out = {SIM_START_YEAR: {"P": P.copy(), "C": C.copy(), "alpha": alpha.copy()}}
    if save_cb is not None:
        save_cb(out)
    prev_year = SIM_START_YEAR
    for year in YEARS[1:]:
        _update_growth_rates(model, data, year)
        result, err = _solve_one_year(model, y0, year - prev_year)
        if result is None:
            print(f"  conditioned solve stopped at {year}: {err}")
            return out
        _, C_sim, alpha_sim = result

        C = data["C_obs"][year].astype(float).copy()
        alpha = data["alpha_obs"][year].astype(float).copy()
        C[country_idx] = C_sim[country_idx]
        alpha[country_idx] = _normalize_row(
            alpha_sim[country_idx],
            data["alpha_obs"][year][country_idx],
        )
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
        if save_cb is not None:
            save_cb(out)
        y0 = np.concatenate([P, C, alpha.flatten()])
        prev_year = year
    return out


def simulate_alpha_frozen(theta, data, country_idx, save_cb=None):
    """
    Like simulate_conditioned, but ALL alpha rows (including the target's) are
    pinned to observed at every year boundary. Only C[target] is taken from
    the ODE; the other 18 countries' C are observed; P is reconstructed.

    P_new = P_obs + (C_sim[target] - C_obs[target]) * alpha_obs[target,:]
    """
    model = build_model(theta, data, country_idx)
    P = data["P_obs"][SIM_START_YEAR].astype(float).copy()
    C = data["C_obs"][SIM_START_YEAR].astype(float).copy()
    alpha = data["alpha_obs"][SIM_START_YEAR].astype(float).copy()
    y0 = np.concatenate([P, C, alpha.flatten()])

    out = {SIM_START_YEAR: {"P": P.copy(), "C": C.copy(), "alpha": alpha.copy()}}
    if save_cb is not None:
        save_cb(out)
    prev_year = SIM_START_YEAR
    for year in YEARS[1:]:
        _update_growth_rates(model, data, year)
        result, err = _solve_one_year(model, y0, year - prev_year)
        if result is None:
            print(f"  alpha-frozen solve stopped at {year}: {err}")
            return out
        _, C_sim, _ = result

        # Pin the entire alpha matrix to observed.
        alpha = data["alpha_obs"][year].astype(float).copy()
        # Other countries' C: observed. Target country's C: simulated.
        C = data["C_obs"][year].astype(float).copy()
        C[country_idx] = C_sim[country_idx]
        # Reconstruct P with target's contribution swapped (alpha is observed
        # for everyone, so the swap simplifies to a delta-C term).
        target_alpha_row = alpha[country_idx]
        delta_C = C[country_idx] - data["C_obs"][year][country_idx]
        P = data["P_obs"][year].astype(float).copy() + delta_C * target_alpha_row
        P = np.clip(P, 1e-12, None)
        model.activate_new_links(P, C, alpha)

        out[year] = {"P": P.copy(), "C": C.copy(), "alpha": alpha.copy()}
        if save_cb is not None:
            save_cb(out)
        y0 = np.concatenate([P, C, alpha.flatten()])
        prev_year = year
    return out


def simulate_free_joint(theta, data, save_cb=None):
    """
    One full free ODE: all 19 countries evolve together from the 1988 observed
    state with no further anchoring.
    """
    model = build_model(theta, data)
    P = data["P_obs"][SIM_START_YEAR].astype(float).copy()
    C = data["C_obs"][SIM_START_YEAR].astype(float).copy()
    alpha = data["alpha_obs"][SIM_START_YEAR].astype(float).copy()
    y0 = np.concatenate([P, C, alpha.flatten()])

    out = {SIM_START_YEAR: {"P": P.copy(), "C": C.copy(), "alpha": alpha.copy()}}
    if save_cb is not None:
        save_cb(out)
    prev_year = SIM_START_YEAR
    for year in YEARS[1:]:
        _update_growth_rates(model, data, year)
        result, err = _solve_one_year(model, y0, year - prev_year)
        if result is None:
            print(f"  free-joint solve stopped at {year}: {err}")
            return out
        P_sim, C_sim, alpha_sim = result

        row_s = alpha_sim.sum(axis=1, keepdims=True)
        row_s[row_s == 0] = 1.0
        alpha = alpha_sim / row_s
        P = P_sim
        C = C_sim
        out[year] = {"P": P.copy(), "C": C.copy(), "alpha": alpha.copy()}
        if save_cb is not None:
            save_cb(out)
        y0 = np.concatenate([P, C, alpha.flatten()])
        prev_year = year
    return out


def plot_trajectory_fit(per_country, country_rows, data, out_path, mode_label):
    n_cols, n_rows = 5, 4
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 12), sharex=True)
    obs_years = [y for y in YEARS if y in data["C_obs"]]

    for idx, row in enumerate(country_rows):
        ax = axes.flat[idx]
        ci = row["position"]
        code = row["location_code"]
        traj = per_country.get(code)
        if traj is None:
            ax.set_visible(False)
            continue

        sim_years = sorted(traj.keys())
        sim_share = [
            traj[y]["C"][ci] / max(traj[y]["C"].sum(), 1e-12) for y in sim_years
        ]
        obs_share = [
            data["C_obs"][y][ci] / data["C_obs"][y].sum() for y in obs_years
        ]
        ax.plot(obs_years, obs_share, color="#1F4E79", lw=1.5, label="Observed")
        ax.plot(sim_years, sim_share, color="#B5541E", lw=1.5,
                label=f"Simulated ({mode_label})")
        ax.set_title(row["country_name_short"], fontsize=9, fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=7)

    for idx in range(len(country_rows), n_rows * n_cols):
        axes.flat[idx].set_visible(False)

    axes[0, 0].legend(fontsize=7, frameon=False)
    fig.supxlabel("Year", fontsize=11)
    fig.supylabel("Market share", fontsize=11)
    fig.suptitle(
        f"Country market shares: observed vs simulated ({mode_label})",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


def plot_top10_alphas(per_country, country_rows, data, out_dir, mode_label, top_k=10):
    products = load_products_index()

    def prod_label(i):
        row = products.get(i)
        if row is None:
            return str(i)
        return row.get("product_name_short") or row.get("hs_product_code") or str(i)

    obs_years = sorted(data["alpha_obs"].keys())
    last_obs_year = max(obs_years)

    for row in country_rows:
        code = row["location_code"]
        ci = row["position"]
        traj = per_country.get(code)
        if traj is None:
            continue

        alpha_last_obs = data["alpha_obs"][last_obs_year][ci]
        top_idx = np.argsort(alpha_last_obs)[::-1][:top_k]
        sim_years = sorted(traj.keys())

        fig, ax = plt.subplots(figsize=(11, 6))
        cmap = plt.get_cmap("tab10")
        for k, pi in enumerate(top_idx):
            color = cmap(k % 10)
            obs_series = [data["alpha_obs"][y][ci, pi] for y in obs_years]
            sim_series = [traj[y]["alpha"][ci, pi] for y in sim_years]
            ax.plot(obs_years, obs_series, color=color, lw=1.6, label=prod_label(pi))
            ax.plot(sim_years, sim_series, color=color, lw=1.3, ls="--", alpha=0.85)

        ax.set_title(
            f"{row['country_name_short']} ({code}) — top {top_k} alphas at {last_obs_year}\n"
            f"observed (solid) vs {mode_label} simulation (dashed)",
            fontsize=11,
            fontweight="bold",
        )
        ax.set_xlabel("Year")
        ax.set_ylabel(r"$\alpha$ (specialisation share)")
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=7, ncol=2, frameon=False, loc="best")
        plt.tight_layout()
        out_path = os.path.join(out_dir, f"alpha_top10_{mode_label}_{code}.png")
        plt.savefig(out_path, dpi=180, bbox_inches="tight")
        plt.close()
        print(f"Saved {out_path}")


def _conditioned_worker(theta, data, country_idx, mode_label, code):
    """Run simulate_conditioned in a child process; cache after every year."""
    cb = lambda traj: _save_traj_cache(mode_label, code, theta, traj)
    simulate_conditioned(theta, data, country_idx, save_cb=cb)


def _alpha_frozen_worker(theta, data, country_idx, mode_label, code):
    """Run simulate_alpha_frozen in a child process; cache after every year."""
    cb = lambda traj: _save_traj_cache(mode_label, code, theta, traj)
    simulate_alpha_frozen(theta, data, country_idx, save_cb=cb)


def _free_joint_worker(theta, data, mode_label, code):
    """Run simulate_free_joint in a child process; cache after every year."""
    cb = lambda traj: _save_traj_cache(mode_label, code, theta, traj)
    simulate_free_joint(theta, data, save_cb=cb)


def _run_with_timeout(target, args, timeout_s, label):
    """
    Run target(*args) in a forked subprocess. Kill it after timeout_s.
    The child is responsible for writing its own cache file before exiting.
    Returns True if the process ended naturally, False if killed.
    """
    methods = mp.get_all_start_methods()
    method = "fork" if "fork" in methods else methods[0]
    ctx = mp.get_context(method)
    proc = ctx.Process(target=target, args=args)
    proc.start()
    proc.join(timeout=timeout_s)
    if proc.is_alive():
        print(f"  {label}: hit {timeout_s:.0f}s budget; killing subprocess",
              flush=True)
        if hasattr(proc, "kill"):
            proc.kill()
        else:
            proc.terminate()
        proc.join(timeout=2.0)
        return False
    return True


def load_theta(code):
    theta_path = os.path.join(LEGACY_COUNTRY_DIR, code, "best_theta.npy")
    if not os.path.exists(theta_path):
        return None
    theta = np.load(theta_path)
    if len(theta) != len(LEGACY_PARAM_NAMES):
        return None
    return theta


def _missing_expected_years(traj, expected_years=YEARS):
    if traj is None:
        return list(expected_years)
    have = set(traj.keys())
    return [year for year in expected_years if year not in have]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["conditioned", "alpha_frozen", "free_joint"],
        choices=["conditioned", "alpha_frozen", "free_joint"],
        help="Which simulation passes to run (default: all three).",
    )
    parser.add_argument(
        "--params-dir",
        default=DEFAULT_COUNTRY_DIR,
        help=(
            "Directory holding per-country best_theta.npy + summary.csv. "
            f"Default: {DEFAULT_COUNTRY_DIR}"
        ),
    )
    parser.add_argument(
        "--r-values",
        default=DEFAULT_R_VALUES_PATH,
        help=(
            "CSV with country growth-regression r-values used to seed model.r_C. "
            f"Default: {DEFAULT_R_VALUES_PATH}"
        ),
    )
    parser.add_argument(
        "--fresh", action="store_true",
        help="Ignore cached trajectories and recompute requested modes.",
    )
    parser.add_argument(
        "--timeout-s",
        type=float,
        default=max(PER_COUNTRY_TIMEOUT_S, 300.0),
        help=(
            "Wall-clock budget per simulation worker. Partial caches are "
            "retried automatically. Default: 300."
        ),
    )
    args = parser.parse_args()
    modes = set(args.modes)

    # Override module-level paths so downstream loaders pick up the CLI choice.
    set_simulation_sources(args.params_dir, args.r_values)
    print(f"Params dir: {LEGACY_COUNTRY_DIR}")
    print(f"R-values:   {R_VALUES_PATH}")

    print("Loading empirical data...", flush=True)
    data = load_data()
    rows = load_country_index()
    data["r_C_growth_regression"] = load_growth_regression_r(rows)
    code_to_idx = {r["location_code"]: r["position"] for r in rows}

    def _run_per_country(mode_label, worker, label_pretty):
        """Run per-country simulation pass. Returns dict {code: traj}."""
        print(f"\n=== {label_pretty} per-country ===", flush=True)
        out = {}
        for row in rows:
            code = row["location_code"]
            theta = load_theta(code)
            if theta is None:
                print(f"[skip] {code}: no/invalid best_theta.npy", flush=True)
                continue
            if args.fresh:
                _purge_cache_family(mode_label, code)
            else:
                cached = _load_traj_cache(mode_label, code, theta)
                if cached is not None:
                    missing = _missing_expected_years(cached)
                    if not missing:
                        print(f"[{code}] using cached {label_pretty} simulation "
                              f"({len(cached)} years)", flush=True)
                        out[code] = cached
                        continue
                    print(
                        f"[{code}] cached {label_pretty} simulation is incomplete "
                        f"({len(cached)}/{len(YEARS)} years, last={max(cached)}, "
                        f"first missing={missing[0]}); recomputing",
                        flush=True,
                    )
                    _purge_cache_family(mode_label, code)
            t0 = time.time()
            print(f"[{code}] {label_pretty} simulation...", flush=True)
            _run_with_timeout(
                target=worker,
                args=(theta, data, code_to_idx[code], mode_label, code),
                timeout_s=args.timeout_s,
                label=f"[{code}] {label_pretty}",
            )
            traj = _load_traj_cache(mode_label, code, theta)
            if traj is None:
                print(f"[{code}] no trajectory produced "
                      f"(killed before any year saved)", flush=True)
                continue
            missing = _missing_expected_years(traj)
            if missing:
                print(
                    f"[{code}] WARNING: {label_pretty} trajectory still incomplete "
                    f"({len(traj)}/{len(YEARS)} years, last={max(traj)}). "
                    f"Increase --timeout-s or inspect the solver for this country.",
                    flush=True,
                )
            out[code] = traj
            print(f"[{code}] done in {time.time()-t0:.1f}s "
                  f"({len(traj)} years cached)", flush=True)
        return out

    # Pass 1: per-country conditioned simulations
    if "conditioned" in modes:
        cond = _run_per_country("conditioned", _conditioned_worker, "conditioned")
        plot_trajectory_fit(
            cond, rows, data,
            os.path.join(THIS_DIR, "trajectory_fit_conditioned.png"),
            mode_label="conditioned",
        )
        plot_top10_alphas(cond, rows, data, THIS_DIR, mode_label="conditioned")

    # Pass 2: per-country alpha-frozen simulations (only C[target] is free)
    if "alpha_frozen" in modes:
        frozen = _run_per_country(
            "alpha_frozen", _alpha_frozen_worker, "alpha-frozen",
        )
        plot_trajectory_fit(
            frozen, rows, data,
            os.path.join(THIS_DIR, "trajectory_fit_alpha_frozen.png"),
            mode_label="alpha-frozen",
        )
        # Skip the alpha plot — alpha is observed by construction.

    # Pass 3: single joint free simulation using the best-loss country's theta
    if "free_joint" in modes:
        ref_code, ref_loss = best_loss_country()
        ref_theta = load_theta(ref_code) if ref_code else None
        if ref_theta is None:
            print("\nSkipping joint free simulation: no usable reference theta.",
                  flush=True)
            return

        print(
            f"\n=== free joint (reference country: {ref_code}, "
            f"best_loss={ref_loss:.4f}) ===", flush=True,
        )
        if args.fresh:
            _purge_cache_family("free_joint", ref_code)
            cached_full = None
        else:
            cached_full = _load_traj_cache("free_joint", ref_code, ref_theta)
        if cached_full is not None and not _missing_expected_years(cached_full):
            print(f"using cached free-joint simulation "
                  f"({len(cached_full)} years)", flush=True)
            full = cached_full
        else:
            if cached_full is not None:
                missing = _missing_expected_years(cached_full)
                print(
                    f"cached free-joint simulation is incomplete "
                    f"({len(cached_full)}/{len(YEARS)} years, "
                    f"last={max(cached_full)}, first missing={missing[0]}); recomputing",
                    flush=True,
                )
                _purge_cache_family("free_joint", ref_code)
            t0 = time.time()
            _run_with_timeout(
                target=_free_joint_worker,
                args=(ref_theta, data, "free_joint", ref_code),
                timeout_s=args.timeout_s,
                label="free-joint",
            )
            full = _load_traj_cache("free_joint", ref_code, ref_theta)
            if full is None:
                print("free-joint produced no trajectory; skipping plots",
                      flush=True)
                return
            missing = _missing_expected_years(full)
            if missing:
                print(
                    f"WARNING: free-joint trajectory still incomplete "
                    f"({len(full)}/{len(YEARS)} years, last={max(full)}). "
                    f"Increase --timeout-s or inspect the solver.",
                    flush=True,
                )
            print(f"free-joint done in {time.time()-t0:.1f}s "
                  f"({len(full)} years cached)", flush=True)
        free_per_country = {row["location_code"]: full for row in rows}
        plot_trajectory_fit(
            free_per_country, rows, data,
            os.path.join(THIS_DIR, "trajectory_fit_free_joint.png"),
            mode_label=f"free joint, theta={ref_code}",
        )
        plot_top10_alphas(
            free_per_country, rows, data, THIS_DIR,
            mode_label="free_joint",
        )


if __name__ == "__main__":
    main()
