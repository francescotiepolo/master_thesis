"""
Shared functions for the calibration files.

  load_data              — loads all extracted arrays into a dict
  build_model            — constructs ProductSpaceModel from empirical data + theta
  simulate               — runs model year-by-year, returns annual snapshots
  spearman               — Spearman rank correlation
  compute_stats          — summary statistics at one time point
  trajectory_loss        — full scalar loss over calibration years
  theta_to_dict          — parameter vector to named dict
  evaluate_batch         — parallel evaluation of multiple parameter vectors
  load_nroy_bounds       — load NROY bounds from initial design results or fall back to prior
"""

import os
import sys
import warnings
import signal
import time
import multiprocessing as mp
from contextlib import contextmanager
import numpy as np
from scipy.stats import spearmanr as _spearmanr

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from product_space_model import ProductSpaceModel
from calibration_config import (
    EXTRACTED_DIR, YEAR_START, YEAR_END, CALIB_YEARS, FIXED, LOSS_WEIGHTS,
    TIME_WEIGHT_SLOPE, SIM_STEPS_PER_YEAR, SOLVE_TIMEOUT_S, TRAJECTORY_TIMEOUT_S,
    MAX_SOLVER_STEPS,
    PARAM_NAMES, PARAM_BOUNDS,
    LHS_DIR, DE_DIR,
    WINDOW_REINIT_YEARS, GROWTH_RATE_PERIODS,
    COUNTRY_GROUPS,
)

# Penalty value for failed simulations
PENALTY = 1e6


def _build_r_C_from_groups(params, SC):
    """
    Build per-country r_C vector from group-level r_C free parameters.
    """
    r_C = np.zeros(SC)
    for group_name, indices in COUNTRY_GROUPS.items():
        key = f"r_C_{group_name}"
        val = float(params.get(key, 0.0))
        for j in indices:
            r_C[j] = val
    return r_C


class _SolveTimeout(RuntimeError):
    """
    Raised when an ODE solve exceeds the configured wall-clock limit.
    """


@contextmanager
def _wall_clock_timeout(timeout_s):
    """
    Hard time limit for ODE solves via SIGALRM.
    simulate() can only check for timeouts between years, not mid-solve.
    """
    # Disable SIGALRM in subprocess workers
    if (
        timeout_s is None
        or timeout_s <= 0
        or os.name != "posix"
        or os.environ.get("_CALIB_SUBPROCESS")
        or mp.current_process().name != "MainProcess"
    ):
        yield
        return

    def _handle_timeout(signum, frame):
        raise _SolveTimeout(f"ODE solve exceeded {timeout_s}s")

    old_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, _handle_timeout)
    signal.setitimer(signal.ITIMER_REAL, timeout_s)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, old_handler)


# Data loading

def _compute_piecewise_growth_rates(series_dict, periods):
    """
    Compute OLS log growth rates for each sub-period.
    """
    from scipy.stats import linregress
    rates = {}
    for start, end in periods:
        years = sorted(yr for yr in series_dict if start <= yr <= end)
        if len(years) < 3:
            continue
        t = np.array(years, dtype=float) - years[0]
        matrix = np.stack([series_dict[yr] for yr in years], axis=0)
        n_series = matrix.shape[1]
        slopes = np.full(n_series, 0.0)
        for i in range(n_series):
            y = matrix[:, i]
            pos = y > 0
            if pos.sum() >= 3:
                slope, *_ = linregress(t[pos], np.log(y[pos]))
                if np.isfinite(slope):
                    slopes[i] = slope
        rates[(start, end)] = slopes
    return rates


def load_data(extracted_dir=EXTRACTED_DIR, years=None):
    """
    Load all arrays from extracted_data/ into a dict.
    """
    if years is None:
        years = list(range(YEAR_START, YEAR_END + 1))

    def npy(name):
        return np.load(os.path.join(extracted_dir, name))

    data = {
        "phi_space": npy("phi_space.npy"),
        "beta_C_ref": npy("beta_C_ref.npy"),
        "alpha_init": npy("alpha_init.npy"),
        "P_init": npy("P_init.npy"),
        "C_init": npy("C_init.npy"),
        "r_P": npy("r_P.npy"),
        "r_C": npy("r_C.npy"),
        "alpha_obs": {},
        "C_obs": {},
        "P_obs": {},
    }
    data["SC"] = data["beta_C_ref"].shape[0]
    data["SP"] = data["phi_space"].shape[0]

    ann = os.path.join(extracted_dir, "annual")
    for yr in years:
        data["alpha_obs"][yr] = np.load(os.path.join(ann, f"alpha_{yr}.npy"))
        data["C_obs"][yr] = np.load(os.path.join(ann, f"C_{yr}.npy"))
        data["P_obs"][yr] = np.load(os.path.join(ann, f"P_{yr}.npy"))

    # Piecewise growth rates
    data["r_P_periods"] = _compute_piecewise_growth_rates(data["P_obs"], GROWTH_RATE_PERIODS)
    data["r_C_periods"] = _compute_piecewise_growth_rates(data["C_obs"], GROWTH_RATE_PERIODS)

    # Country competition matrix from growth regression: only negative pi entries
    # (true competition) are kept; positives are zeroed. The absolute scale of the
    # regression pi is unreliable in ODE units, so it is multiplied by a free scalar
    # s_pi during calibration (C_CC = s_pi * pi_competition).
    pi_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "calibration_results", "growth_regression", "pi_matrix_fixed.npy",
    )
    if os.path.exists(pi_path):
        pi_mat = np.load(pi_path).astype(float)
        data["pi_competition"] = np.maximum(-pi_mat, 0.0)
    else:
        data["pi_competition"] = None

    print(f"Data loaded: SC={data['SC']}, SP={data['SP']}, "
          f"years={min(data['alpha_obs'])}–{max(data['alpha_obs'])}, "
          f"growth_rate_periods={len(data['r_P_periods'])}")
    return data


# Parameter helpers

def theta_to_dict(theta):
    """
    Convert parameter vector to dict.
    """
    return dict(zip(PARAM_NAMES, theta))

def satisfies_constraints(theta):
    """
    Check that all free parameters are within their bounds.
    """
    for val, (lo, hi) in zip(theta, PARAM_BOUNDS):
        if not (lo <= val <= hi):
            return False
    return True


# Model construction

def build_model(theta, data):
    """
    Construct a ProductSpaceModel with empirical data and theta.
    """
    params = theta_to_dict(theta)
    fp = FIXED

    model = ProductSpaceModel(
        N_products = data["SP"],
        n_countries = data["SC"],
        patch_network = True,
        seed = 0,
        phi_space = data["phi_space"],
        s = float(fp["s"]),
        c = float(fp["c"]),
        c_prime = float(fp["c_prime"]),
        gamma = float(fp["gamma"]),
        kappa = float(params["kappa"]),
        sigma = float(params["sigma"]),
        nu = float(params["nu"]),
        G = float(params["G"]),
        q = float(fp["q"]),
        mu = float(fp["mu"]),
        beta_trade_off = float(params["beta_trade_off"]),
        enable_entry = bool(fp.get("enable_entry", False)),
        entry_threshold = float(params["entry_threshold"]),
    )
    _patch_model(model, data, params,
                 h_mean=float(params["h_mean"]),
                 C_diag_mean=float(params["C_diag_mean"]),
                 C_offdiag_mean=float(params["C_offdiag_mean"]))
    return model


def _patch_model(model, data, params=None, h_mean=None, C_diag_mean=None, C_offdiag_mean=None, s_pi=None):
    """
    Overwrite randomly generated parameters with empirical/fixed values.

    If `s_pi` is provided and `data["pi_competition"]` is available, the country
    competition matrix C_CC is set to `s_pi * pi_competition`,
    overriding `C_diag_mean`/`C_offdiag_mean` for C_CC.
    """
    fp = FIXED
    if h_mean is None:
        h_mean = fp.get("h_mean", 0.225)
    if C_diag_mean is None:
        C_diag_mean = fp.get("C_diag_mean", 0.95)
    if C_offdiag_mean is None:
        C_offdiag_mean = fp.get("C_offdiag_mean", 0.03)
    cap = data["beta_C_ref"].astype(float) # Continuous capability weights in [0,1]
    mask = cap > 0

    model.adj_matrix = cap
    model.forbidden_network = np.zeros_like(cap)
    model.KC = mask.sum(axis=1).astype(float) # Degree: count of ever-exported products per country
    model.KP = mask.sum(axis=0).astype(float) # Degree: count of countries per product

    model.r_P = data["r_P"].copy()
    if params is not None and "r_C_declining" in params:
        model.r_C = _build_r_C_from_groups(params, model.SC)
    else:
        model.r_C = data["r_C"].copy()

    alpha = np.where(mask, data["alpha_init"], 0.0)
    row_s = alpha.sum(axis=1)
    # Countries with zero exports in alpha_init but active in the network (thus with some exports over
    # the entire dataset) get uniform allocation over their active products
    for j in np.where(row_s == 0)[0]:
        n_active = mask[j].sum()
        if n_active > 0:
            alpha[j] = mask[j].astype(float) / n_active
    row_s = alpha.sum(axis=1, keepdims=True)
    row_s[row_s == 0] = 1.0
    model.alpha = alpha / row_s

    bt = model.beta_trade_off
    KC_bt = np.maximum(model.KC, 1)[:, np.newaxis] ** bt
    KP_bt = np.maximum(model.KP, 1)[np.newaxis, :] ** bt
    alpha_safe = np.where(mask, np.maximum(model.alpha, 1e-10), 1.0)
    # cap[j,i] replaces the uniform random draw in Terpstra et al.
    model.beta_C = np.where(mask, cap / (KC_bt * alpha_safe), 0.0)
    model.beta_P = np.where(mask, cap / (KP_bt * alpha_safe), 0.0)

    model.h_P = np.full(model.SP, h_mean)
    model.h_C = np.full(model.SC, h_mean)
    model.C_PP = np.full((model.SP, model.SP), C_offdiag_mean)
    np.fill_diagonal(model.C_PP, C_diag_mean)

    if s_pi is not None:
        if data.get("pi_competition") is None:
            raise ValueError(
                "s_pi was provided but data['pi_competition'] is missing. "
                "Expected pi_matrix_fixed.npy in calibration_results/growth_regression/."
            )
        model.C_CC = float(s_pi) * data["pi_competition"].astype(float)
    else:
        model.C_CC = np.full((model.SC, model.SC), C_offdiag_mean)
        np.fill_diagonal(model.C_CC, C_diag_mean)

    model.phi_space = data["phi_space"].astype(float)
    model._build_product_space_matrices()


# Simulation

def _update_growth_rates(model, data, year):
    """
    Switch model product growth rates to the period covering the given year.
    Country growth rates (r_C) are set by group-level free parameters and not updated here.
    """
    r_P_periods = data.get("r_P_periods")
    if r_P_periods is None:
        return
    for (p_start, p_end) in sorted(r_P_periods.keys()):
        if p_start <= year <= p_end:
            model.r_P = r_P_periods[(p_start, p_end)]
            return


def simulate(theta, data, years, max_time=TRAJECTORY_TIMEOUT_S, start_year=None, solve_timeout_s=SOLVE_TIMEOUT_S):
    """
    Simulate from start_year forward, returning (alpha, C, P) at each year.
    Returns None if failure or timeout.
    """
    import time as _time
    t0 = _time.time()
    if start_year is None:
        start_year = YEAR_START
    params = theta_to_dict(theta)
    d_C = float(FIXED.get("d_C", 0.0))
    model = build_model(theta, data)
    y0 = np.concatenate([data["P_init"], data["C_init"], model.alpha.flatten()])
    results = {}
    prev_year = start_year

    try:
        with _wall_clock_timeout(max_time):
            for year in sorted(set(years)):
                if max_time and _time.time() - t0 > max_time:
                    print(f"    [timeout] {_time.time() - t0:.0f}s")
                    return None
                _update_growth_rates(model, data, year)
                n_yr = year - prev_year
                if n_yr <= 0:
                    results[year] = (model.alpha.copy(), data["C_init"].copy(), data["P_init"].copy())
                    continue
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        with _wall_clock_timeout(solve_timeout_s):
                            model.solve(
                                t_end = float(n_yr),
                                d_C = d_C,
                                n_steps = SIM_STEPS_PER_YEAR * n_yr,
                                y0 = y0,
                                max_solver_steps = MAX_SOLVER_STEPS,
                                rtol = 1e-3,
                                atol = 1e-6,
                            )
                except _SolveTimeout:
                    elapsed = _time.time() - t0
                    print(
                        f"    [solve-timeout] year={year} step_years={n_yr} "
                        f"limit={solve_timeout_s}s total_elapsed={elapsed:.0f}s"
                    )
                    return None
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
    except _SolveTimeout:
        print(f"    [trajectory-timeout] limit={max_time}s total_elapsed={_time.time() - t0:.0f}s")
        return None

    return results


# Summary statistics

def spearman(a, b):
    """
    Spearman rank correlation, returning 0.0 on invalid inputs.
    Scale-invariant: used for product allocations (alpha) which already sum to 1,
    and for delta_rank_C where we care about direction of change regardless of units.
    """
    a_flat, b_flat = a.flatten(), b.flatten()
    if len(a_flat) < 3:
        return 0.0
    mask = (a_flat != 0) | (b_flat != 0)
    if mask.sum() < 3:
        return 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r, _ = _spearmanr(a_flat[mask], b_flat[mask])
    return float(r) if np.isfinite(r) else 0.0



def compute_stats(alpha_sim, C_sim, alpha_obs, C_obs, P_sim=None, P_obs=None):
    """
    Compute calibration targets for a single year.
    - nrmse_C:        RMSE of log country market shares / naive baseline RMSE.
    - nrmse_P:        RMSE of log product market shares / naive baseline RMSE.
    - rank_products:  1 - mean per-country Spearman of product allocations.
    """
    sum_sim = C_sim.sum()
    sum_obs = C_obs.sum()
    share_sim = C_sim / sum_sim if sum_sim > 0 else C_sim
    share_obs = C_obs / sum_obs if sum_obs > 0 else C_obs

    eps = 1e-10
    log_sim = np.log(share_sim + eps)
    log_obs = np.log(share_obs + eps)
    n = float(len(share_obs))
    rmse = float(np.sqrt(np.mean((log_sim - log_obs) ** 2)))
    naive_rmse = float(np.sqrt(np.mean((np.log(1.0 / n + eps) - log_obs) ** 2)))
    nrmse_C = rmse / naive_rmse if naive_rmse > 0 else rmse

    # Product-level NRMSE (analogous to nrmse_C)
    nrmse_P = 0.0
    if P_sim is not None and P_obs is not None:
        sum_P_sim = P_sim.sum()
        sum_P_obs = P_obs.sum()
        share_P_sim = P_sim / sum_P_sim if sum_P_sim > 0 else P_sim
        share_P_obs = P_obs / sum_P_obs if sum_P_obs > 0 else P_obs
        log_P_sim = np.log(share_P_sim + eps)
        log_P_obs = np.log(share_P_obs + eps)
        n_P = float(len(share_P_obs))
        rmse_P = float(np.sqrt(np.mean((log_P_sim - log_P_obs) ** 2)))
        naive_rmse_P = float(np.sqrt(np.mean((np.log(1.0 / n_P + eps) - log_P_obs) ** 2)))
        nrmse_P = rmse_P / naive_rmse_P if naive_rmse_P > 0 else rmse_P

    SC = alpha_sim.shape[0]
    rho_products_per_country = np.array([
        spearman(alpha_sim[j, :], alpha_obs[j, :]) for j in range(SC)
    ])
    rho_products = float(np.mean(rho_products_per_country))

    return {
        "nrmse_C": nrmse_C,
        "nrmse_P": nrmse_P,
        "rank_products": 1.0 - rho_products,
        "rho_products": rho_products,
        "share_sim": share_sim,
        "share_obs": share_obs,
    }


def trajectory_correlation(traj, data, years, uniformity_penalty=1.0):
    """
    Per-country Spearman correlation of share time series.
    Returns 1 - mean(rho) + uniformity_penalty * std(rho):
      0 = perfect trajectory tracking, penalises uneven fit across countries.
    """
    sorted_years = sorted(yr for yr in years if yr in traj and yr in data["C_obs"])
    if len(sorted_years) < 3:
        return 1.0

    sim_list, obs_list = [], []
    for yr in sorted_years:
        _, C_sim, _ = traj[yr]
        C_obs = data["C_obs"][yr]
        s_sim = C_sim / C_sim.sum() if C_sim.sum() > 0 else C_sim
        s_obs = C_obs / C_obs.sum() if C_obs.sum() > 0 else C_obs
        sim_list.append(s_sim)
        obs_list.append(s_obs)

    sim_mat = np.column_stack(sim_list)  # (SC, T)
    obs_mat = np.column_stack(obs_list)  # (SC, T)
    SC = sim_mat.shape[0]

    corrs = np.array([spearman(sim_mat[j], obs_mat[j]) for j in range(SC)])
    return 1.0 - float(np.mean(corrs)) + uniformity_penalty * float(np.std(corrs))


def aggregate_loss_components(traj, data, years=None, loss_weights=None,
                              time_weight_slope=TIME_WEIGHT_SLOPE,
                              year_start=YEAR_START):
    """
    Aggregate calibration targets into weighted means and total loss.

    Per-year components (nrmse_C, rank_products) are time-weighted averages.
    Trajectory-level components (traj_corr_C) are computed over the full
    time series. Returns dict with loss_total plus one key per component,
    or None if there are no valid year observations.
    """
    if years is None:
        years = CALIB_YEARS
    if loss_weights is None:
        loss_weights = LOSS_WEIGHTS

    # Per-year components (aggregated with time weighting)
    per_year_keys = [k for k in loss_weights if k in ("nrmse_C", "rank_products", "nrmse_P")]
    totals = {k: 0.0 for k in per_year_keys}
    total_w = 0.0

    for yr in sorted(years):
        if yr not in traj or yr not in data["alpha_obs"]:
            continue
        alpha_sim, C_sim, P_sim = traj[yr]
        P_obs = data["P_obs"].get(yr) if "P_obs" in data else None
        s = compute_stats(alpha_sim, C_sim, data["alpha_obs"][yr], data["C_obs"][yr],
                          P_sim=P_sim, P_obs=P_obs)
        wt = 1.0 + time_weight_slope * (yr - year_start)
        for k in per_year_keys:
            totals[k] += wt * s[k]
        total_w += wt

    if total_w == 0:
        return None

    result = {"loss_total": 0.0}
    for k in per_year_keys:
        result[k] = float(totals[k] / total_w)
        result["loss_total"] += loss_weights[k] * result[k]

    # Trajectory-level components (computed over the full time series)
    if "traj_corr_C" in loss_weights:
        tc = trajectory_correlation(traj, data, years)
        result["traj_corr_C"] = tc
        result["loss_total"] += loss_weights["traj_corr_C"] * tc

    result["loss_total"] = float(result["loss_total"])
    return result


# Trajectory loss

def _windowed_simulate(theta, data, years, start_year=None):
    """
    Simulate with periodic initialisation from observed data.
    Prevents error accumulation by resetting state at window boundaries.
    """
    if start_year is None:
        start_year = YEAR_START
    sorted_years = sorted(set(years))

    # Build window boundaries
    boundaries = [start_year]
    for ry in WINDOW_REINIT_YEARS:
        if start_year < ry <= max(sorted_years):
            boundaries.append(ry)
    boundaries = sorted(set(boundaries))

    all_results = {}
    for idx, w_start in enumerate(boundaries):
        w_end = boundaries[idx + 1] if idx + 1 < len(boundaries) else max(sorted_years) + 1
        w_years = [y for y in sorted_years if w_start <= y < w_end]
        if not w_years:
            continue

        # Re-init from observed data at window start (except first window)
        if w_start != start_year and w_start in data.get("alpha_obs", {}):
            w_data = dict(data)
            w_data["alpha_init"] = data["alpha_obs"][w_start].copy()
            w_data["C_init"] = data["C_obs"][w_start].copy()
            w_data["P_init"] = data["P_obs"][w_start].copy()
        else:
            w_data = data

        traj = simulate(theta, w_data, w_years, start_year=w_start)
        if traj is None:
            return None
        all_results.update(traj)

    return all_results


def trajectory_loss(theta, data, years=None, start_year=None):
    """
    Weighted scalar loss over the requested years.
    Returns PENALTY if failure or constraint violation.
    """
    if years is None:
        years = CALIB_YEARS

    if not satisfies_constraints(theta):
        return PENALTY

    traj = _windowed_simulate(theta, data, years, start_year=start_year)
    if traj is None:
        return PENALTY

    agg = aggregate_loss_components(
        traj,
        data,
        years=years,
        year_start=YEAR_START if start_year is None else start_year,
    )
    return agg["loss_total"] if agg is not None else PENALTY


def _batch_eval_worker(theta, data, years, start_year, conn):
    """
    Evaluate one scalar candidate in an isolated subprocess.
    """
    os.environ["_CALIB_SUBPROCESS"] = "1"
    try:
        loss = float(
            trajectory_loss(
                theta,
                data,
                years=years,
                start_year=start_year,
            )
        )
        conn.send(("ok", loss))
    except Exception as exc:
        conn.send(("err", repr(exc)))
    finally:
        conn.close()


class _TimedBatchEvaluator:
    """
    Evaluate scalar-loss candidates in killable subprocesses.
    """

    def __init__(self, max_workers, timeout_s, penalty):
        methods = mp.get_all_start_methods()
        method = "fork" if "fork" in methods else methods[0]
        self.ctx = mp.get_context(method)
        self.max_workers = max(1, int(max_workers))
        self.timeout_s = float(timeout_s)
        self.penalty = float(penalty)

    def evaluate(self, candidates, data, years=None, start_year=None, progress_label=None):
        """
        Evaluate candidates in parallel subprocesses, returning array of losses.
        """
        thetas = [np.asarray(theta, dtype=np.float64).copy() for theta in candidates]
        if not thetas:
            return np.empty(0, dtype=np.float64)

        results = np.full(len(thetas), self.penalty, dtype=np.float64)
        active = {}
        next_idx = 0
        completed = 0
        last_report = time.time()

        while completed < len(thetas):
            while next_idx < len(thetas) and len(active) < self.max_workers:
                parent_conn, child_conn = self.ctx.Pipe(duplex=False)
                proc = self.ctx.Process(
                    target=_batch_eval_worker,
                    args=(thetas[next_idx], data, years, start_year, child_conn),
                )
                proc.start()
                child_conn.close()
                active[next_idx] = {
                    "proc": proc,
                    "conn": parent_conn,
                    "start": time.time(),
                }
                next_idx += 1

            progress = False
            now = time.time()

            for idx, state in list(active.items()):
                proc = state["proc"]
                conn = state["conn"]

                if conn.poll():
                    try:
                        status, payload = conn.recv()
                    except EOFError:
                        status, payload = ("err", "worker closed pipe before sending a result")
                    conn.close()
                    proc.join(timeout=0.1)
                    active.pop(idx)
                    completed += 1
                    progress = True
                    if status == "ok" and np.isfinite(payload):
                        results[idx] = float(payload)
                    else:
                        print(f"    [eval-error] candidate={idx} err={payload}", flush=True)
                    continue

                if now - state["start"] > self.timeout_s:
                    print(
                        f"    [eval-timeout] candidate={idx} limit={self.timeout_s:.0f}s",
                        flush=True,
                    )
                    if hasattr(proc, "kill"):
                        proc.kill()
                    else:
                        proc.terminate()
                    proc.join(timeout=1.0)
                    conn.close()
                    active.pop(idx)
                    completed += 1
                    progress = True
                    continue

                if not proc.is_alive():
                    proc.join(timeout=0.1)
                    conn.close()
                    active.pop(idx)
                    completed += 1
                    progress = True
                    print(
                        f"    [eval-error] candidate={idx} worker exited without result",
                        flush=True,
                    )

            if (
                progress_label is not None
                and (completed == len(thetas) or now - last_report >= 30.0)
            ):
                pct = 100 * completed / len(thetas)
                print(
                    f"    [{progress_label}] progress: {completed}/{len(thetas)} done ({pct:.0f}%) "
                    f"| {len(active)} still running",
                    flush=True,
                )
                last_report = now

            if not progress:
                time.sleep(0.05)

        return results


def multi_objective_loss(theta, data, years=None, start_year=None):
    """
    Return individual objective values as a numpy array for multi-objective optimization.
    """
    from calibration_config import LOSS_OBJECTIVES

    n_obj = len(LOSS_OBJECTIVES)
    penalty_vec = np.full(n_obj, PENALTY)

    if years is None:
        years = CALIB_YEARS
    if not satisfies_constraints(theta):
        return penalty_vec

    traj = _windowed_simulate(theta, data, years, start_year=start_year)
    if traj is None:
        return penalty_vec

    agg = aggregate_loss_components(
        traj, data, years=years,
        year_start=YEAR_START if start_year is None else start_year,
    )
    if agg is None:
        return penalty_vec

    return np.array([agg.get(k, PENALTY) for k in LOSS_OBJECTIVES])


def load_nroy_bounds():
    """
    Load NROY bounds from initial-design results, or return prior bounds if not found.
    """
    import json
    lhs_path = os.path.join(LHS_DIR, "nroy_bounds.json")
    if os.path.exists(lhs_path):
        with open(lhs_path) as f:
            bounds = json.load(f)["bounds"]
        print(f"Loaded NROY bounds from {lhs_path}")
    else:
        bounds = list(PARAM_BOUNDS)
        print("No initial-design results found -- using prior bounds.")
    return bounds


def evaluate_batch(candidates, data, years=None, start_year=None,
                   n_jobs=-1, max_workers=128, chunk_size=200):
    """
    Evaluate a batch of parameter vectors in parallel.
    Processes in isolated subprocesses so ODE solves can be killed after a timeout.
    """
    if n_jobs == -1 or n_jobs > max_workers:
        n_jobs = max_workers
    evaluator = _TimedBatchEvaluator(
        max_workers=n_jobs,
        timeout_s=TRAJECTORY_TIMEOUT_S,
        penalty=PENALTY,
    )
    n = len(candidates)
    results = []
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        if n > chunk_size:
            print(f"  Chunk {start}-{end} / {n}", flush=True)
        chunk_res = evaluator.evaluate(
            candidates[start:end],
            data,
            years=years,
            start_year=start_year,
            progress_label=f"chunk {start}-{end}",
        )
        results.extend(chunk_res)
    return np.array(results)