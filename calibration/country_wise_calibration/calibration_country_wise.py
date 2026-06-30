"""
Country-wise calibration.

Calibrates one target country at a time while conditioning on the observed
exports of the other 18 countries.

The workflow:

1. Latin hypercube survey
2. Differential evolution

Country-wise split used here:
  calibration: 1988-2014 (excluding 1993 and 1994 due to zero coverage)
  validation : 2015-2024
"""

import argparse
import csv
import json
import os
import sys
import time
import warnings
import multiprocessing as mp

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution
from scipy.stats.qmc import LatinHypercube

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    from product_space_model import ProductSpaceModel
    from calibration_config import (
        CALIB_DIR,
        CALIB_END,
        CALIB_YEARS,
        EXTRACTED_DIR,
        FIXED,
        LOSS_WEIGHTS,
        MAX_PARALLEL_JOBS,
        MAX_SOLVER_STEPS,
        SEED,
        SIM_STEPS_PER_YEAR,
        SOLVE_TIMEOUT_S,
        TIME_WEIGHT_SLOPE,
        TRAJECTORY_TIMEOUT_S,
        WINDOW_REINIT_YEARS,
        YEAR_START,
    )
    from calibration_utils import (
        PENALTY,
        _SolveTimeout,
        _patch_model,
        _update_growth_rates,
        _wall_clock_timeout,
        load_data,
        spearman,
    )
except ModuleNotFoundError as exc:
    if exc.name not in {
        "product_space_model",
        "calibration_config",
        "calibration_utils",
    }:
        raise
    from ..product_space_model import ProductSpaceModel
    from .calibration_config import (
        CALIB_DIR,
        CALIB_END,
        CALIB_YEARS,
        EXTRACTED_DIR,
        FIXED,
        LOSS_WEIGHTS,
        MAX_PARALLEL_JOBS,
        MAX_SOLVER_STEPS,
        SEED,
        SIM_STEPS_PER_YEAR,
        SOLVE_TIMEOUT_S,
        TIME_WEIGHT_SLOPE,
        TRAJECTORY_TIMEOUT_S,
        WINDOW_REINIT_YEARS,
        YEAR_START,
    )
    from .calibration_utils import (
        PENALTY,
        _SolveTimeout,
        _patch_model,
        _update_growth_rates,
        _wall_clock_timeout,
        load_data,
        spearman,
    )


CONDITIONED_DIR = os.path.join(CALIB_DIR, "country_wise_restricted")
ALPHA_FROZEN_DIR = os.path.join(CALIB_DIR, "country_wise_alpha_frozen")
COUNTRY_WISE_DIR = CONDITIONED_DIR  # set definitively in main() based on --mode
GROWTH_REGRESSION_DIR = os.path.join(CALIB_DIR, "growth_regression")

# Simulation mode for the calibration loop. Set by main() before any worker
# subprocess is forked; child processes inherit it via fork.
#   "conditioned": only the target country's C and alpha row come from the
#                  ODE; the other 18 countries' state is pinned to observed.
#   "alpha_frozen": all alpha rows (target included) are pinned to observed
#                   every year; only target country's C comes from the ODE.
_MODE = "conditioned"

BOOTSTRAP_PATH = os.path.join(CALIB_DIR, "joint", "bootstrap_globals.json")


def _load_bootstrap():
    if not os.path.exists(BOOTSTRAP_PATH):
        raise FileNotFoundError(
            f"Missing bootstrap pack: {BOOTSTRAP_PATH}. "
            "Run calibration/country_wise_calibration/bootstrap_globals.py first."
        )
    with open(BOOTSTRAP_PATH) as f:
        return json.load(f)


# Bootstrap globals are no longer consulted here — every Stage-3 global is now
# a free per-country parameter. _load_bootstrap is still defined above because
# shared_global_calibration imports it for Stage 3's seed/centre point.
# Use the same regression variant as pi_matrix_fixed.npy (loaded for C_CC in
# calibration_utils.load_data) so r_C and the country competition matrix come
# from a single coherent fit.
R_VALUES_PATH = os.path.join(GROWTH_REGRESSION_DIR, "r_values_fixed.csv")
LEGACY_COUNTRY_WISE_SUMMARY = os.environ.get(
    "COUNTRY_SEED_SUMMARY",
    os.path.join(CALIB_DIR, "country_wise", "summary.csv"),
)

# Set COUNTRY_START_FROM to a country code to resume from that country onward.
# Leave unset for a fresh all-country run.
START_FROM_COUNTRY = os.environ.get("COUNTRY_START_FROM") or None

COUNTRY_CALIB_END = 2014
COUNTRY_CALIB_YEARS = [
    year for year in range(YEAR_START, COUNTRY_CALIB_END + 1)
    if year not in (1993, 1994)
]
COUNTRY_VALID_YEARS = list(range(COUNTRY_CALIB_END + 1, 2024 + 1))

COUNTRY_PARAM_NAMES = [
    # Original per-country params
    "s_pi",
    "beta_trade_off",
    "G",
    "nu",
    "h_C",
    "entry_threshold",
    # Globals lifted to per-country
    "kappa",
    "sigma",
    "h_P",
    "s_pi_P",
    "r_C_target",
    "a_P",
    "b_P",
]
COUNTRY_PARAM_BOUNDS = [
    (1e-4, 1.0),    # s_pi
    (0.0, 1.0),     # beta_trade_off
    (0.01, 2.0),    # G
    (0.0, 1.0),     # nu
    (0.05, 5.0),    # h_C
    (0.01, 50.0),   # entry_threshold
    (0.0, 0.2),     # kappa
    (0.01, 4.0),    # sigma
    (0.05, 5.0),    # h_P
    (1e-4, 50.0),   # s_pi_P
    (-5.0, 10.0),   # r_C_target (regression r_C ranges roughly -3.8..+7.8)
    (-0.5, 0.5),    # a_P
    (0.3, 1.0),     # b_P
]
N_COUNTRY_PARAMS = len(COUNTRY_PARAM_NAMES)
# In alpha-frozen mode, alpha is pinned to observed every year, so the alpha-
# only parameters G, nu, entry_threshold have no effect on the loss. We pin
# them per-country to the values from the legacy country_wise/ run (the one
# that calibrated all 6 original params freely per country) and drop them
# from the search.
ALPHA_FROZEN_FIXED_PARAM_NAMES = ("G", "nu", "entry_threshold")
# Stage 3's beta_{declining,rising,stable}_delta are NOT lifted here: they exist
# in Stage 3 only to shift per-country beta_trade_off values that are pinned
# from a prior fit. Here beta_trade_off is already free per country, so a delta
# would be exactly degenerate with it.
# r_C: only the target country's growth rate is free (r_C_target). The other
# 18 countries take their value straight from the growth regression — they get
# overwritten by observed C at every year boundary, so a free a_C/b_C affine
# map would have been near-degenerate.
# r_P: rebuilt each candidate from the affine map
# a_P + b_P * (r_P_regression - mean(r_P_regression)). b_P is kept because
# relative cross-product growth structure propagates to the target via xi_i.

FAST_LHS_N_SAMPLES = 350
FAST_LHS_ELITE_FRACTION = 0.25
FAST_LHS_PADDING = 0.10
FAST_DE_POPSIZE = int(os.environ.get("FAST_DE_POPSIZE", "6"))
FAST_DE_MAXITER = 300
FAST_DE_TOL = 0.005
FAST_DE_MUTATION = (0.5, 1.0)


def _get_mode_aware_loss_weights(mode):
    """
    Return loss weights adjusted for the simulation mode.
    
    In "alpha_frozen" mode, alpha is pinned to observed every year, so:
      - rank_products is always ~1.0 (meaningless)
      - nrmse_P is always ~0 (meaningless)
    
    In "conditioned" mode, use standard weights.
    """
    if mode == "alpha_frozen":
        # Redistribute rank_products and nrmse_P weights to C and traj
        # Maintain relative ratio: nrmse_C:traj_corr_C = 0.25:0.30 = 5:6
        return {
            "nrmse_C": 0.45,
            "traj_corr_C": 0.55,
            "rank_products": 0.0,
            "nrmse_P": 0.0,
        }
    else:  # "conditioned"
        return LOSS_WEIGHTS


def _resolve_mode(mode):
    """
    Resolve an explicit simulation mode or fall back to the module default.
    """
    if mode is None:
        mode = _MODE
    if mode not in ("conditioned", "alpha_frozen"):
        raise ValueError(f"unknown mode: {mode}")
    return mode


def country_theta_to_dict(theta):
    """
    Convert a flat parameter vector to a dictionary for the country model.
    """
    return dict(zip(COUNTRY_PARAM_NAMES, theta))


def satisfies_country_constraints(theta):
    """
    Check if theta satisfies constraints for the country model.
    """
    for value, (lower, upper) in zip(theta, COUNTRY_PARAM_BOUNDS):
        if not (lower <= value <= upper):
            return False
    return True


def _theta_key(theta):
    """
    Create a key from a theta vector for saving results.
    """
    return tuple(np.round(np.asarray(theta, dtype=np.float64), 12))


def _default_seed_for_extra_params():
    """
    Per-country defaults for the 8 globals lifted to per-country.
    Numeric defaults come from bootstrap_globals.json when present (kappa,
    sigma, h_P, s_pi_P, a_P, b_P); r_C_target is filled per country from the
    growth regression (see _load_legacy_country_seed_map). Falls back to mid-
    bound for any value missing from the bootstrap pack.
    """
    try:
        boot = _load_bootstrap()
    except FileNotFoundError:
        boot = {}
    midpoints = {
        name: 0.5 * (lo + hi)
        for name, (lo, hi) in zip(COUNTRY_PARAM_NAMES, COUNTRY_PARAM_BOUNDS)
    }
    extras = {
        "kappa": float(boot.get("kappa", midpoints["kappa"])),
        "sigma": float(boot.get("sigma", midpoints["sigma"])),
        "h_P": float(boot.get("h_P", midpoints["h_P"])),
        "s_pi_P": float(boot.get("s_pi_P", midpoints["s_pi_P"])),
        "a_P": float(boot.get("a_P", 0.0)),
        "b_P": float(boot.get("b_P", 1.0)),
    }
    return extras


def _load_legacy_country_seed_map():
    """
    Build per-country seed vectors of length N_COUNTRY_PARAMS.

    If the seed CSV already contains the current country-wise parameter schema,
    use it directly. Otherwise, fall back to the older 6-parameter summary:
    the legacy CSV contributes the original per-country params; bootstrap
    globals contribute kappa/sigma/h_P/s_pi_P/a_P/b_P; r_C_target is seeded
    from the country's own regression r_C value.
    """
    if not os.path.exists(LEGACY_COUNTRY_WISE_SUMMARY):
        return {}
    extras = _default_seed_for_extra_params()
    # Per-country r_C_target seed read directly from the regression CSV.
    # (load_country_index / load_growth_regression_r are defined later in
    # this file, so we can't call them here at module import time.)
    r_C_seed_by_code = {}
    if os.path.exists(R_VALUES_PATH):
        with open(R_VALUES_PATH, newline="") as handle:
            for row in csv.DictReader(handle):
                code = row.get("country")
                if code is None:
                    continue
                try:
                    r_C_seed_by_code[code] = float(row["r_i"])
                except (KeyError, TypeError, ValueError):
                    continue
    out = {}
    with open(LEGACY_COUNTRY_WISE_SUMMARY, newline="") as handle:
        for row in csv.DictReader(handle):
            code = row.get("country_code")
            if not code:
                continue
            try:
                if all(name in row and row[name] != "" for name in COUNTRY_PARAM_NAMES):
                    out[code] = np.array(
                        [float(row[name]) for name in COUNTRY_PARAM_NAMES],
                        dtype=float,
                    )
                    continue
            except (TypeError, ValueError):
                pass
            try:
                base = [
                    float(row["s_pi"]),
                    float(row["beta_trade_off"]),
                    float(row["G"]),
                    float(row["nu"]),
                    float(row["h"] if "h" in row and row["h"] != "" else row["h_C"]),
                    float(row["entry_threshold"]),
                ]
            except (KeyError, TypeError, ValueError):
                continue
            extra_map = {
                **extras,
                "r_C_target": r_C_seed_by_code.get(code, 0.0),
            }
            extras_vec = [extra_map[name] for name in COUNTRY_PARAM_NAMES[6:]]
            out[code] = np.array(base + extras_vec, dtype=float)
    return out


_LEGACY_COUNTRY_SEEDS = _load_legacy_country_seed_map()


def _load_idx_to_code_map():
    """
    Map country position (idx) -> location code, read directly from
    countries_index.csv so we don't need load_country_index() (defined later
    in this file) at import time.
    """
    path = os.path.join(EXTRACTED_DIR, "countries_index.csv")
    if not os.path.exists(path):
        return {}
    out = {}
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                out[int(row["position"])] = row["location_code"]
            except (KeyError, TypeError, ValueError):
                continue
    return out


_IDX_TO_CODE = _load_idx_to_code_map()

# Indices of ALPHA_FROZEN_FIXED_PARAM_NAMES inside COUNTRY_PARAM_NAMES.
_ALPHA_FROZEN_FIXED_INDICES = tuple(
    COUNTRY_PARAM_NAMES.index(name) for name in ALPHA_FROZEN_FIXED_PARAM_NAMES
)
# Indices of free (searched) params in alpha-frozen mode.
_ALPHA_FROZEN_FREE_INDICES = tuple(
    i for i in range(N_COUNTRY_PARAMS) if i not in _ALPHA_FROZEN_FIXED_INDICES
)


def _free_indices_for_mode(mode):
    """
    Indices of COUNTRY_PARAM_NAMES that the optimiser sees in this mode.
    """
    if mode == "alpha_frozen":
        return _ALPHA_FROZEN_FREE_INDICES
    return tuple(range(N_COUNTRY_PARAMS))


def active_param_names(mode):
    """
    Names searched by LHS/DE in this mode.
    """
    return [COUNTRY_PARAM_NAMES[i] for i in _free_indices_for_mode(mode)]


def active_param_bounds(mode):
    """
    Bounds searched by LHS/DE in this mode.
    """
    return [COUNTRY_PARAM_BOUNDS[i] for i in _free_indices_for_mode(mode)]


def _legacy_fixed_for_idx(country_idx):
    """
    Return {G, nu, entry_threshold} for the given country idx pulled from the
    legacy country_wise/ summary, or None if unavailable. Used to pin alpha-
    only params in alpha-frozen mode.
    """
    code = _IDX_TO_CODE.get(int(country_idx))
    if code is None:
        return None
    seed = _LEGACY_COUNTRY_SEEDS.get(code)
    if seed is None:
        return None
    return {
        name: float(seed[COUNTRY_PARAM_NAMES.index(name)])
        for name in ALPHA_FROZEN_FIXED_PARAM_NAMES
    }


def expand_theta(reduced_theta, country_idx, mode):
    """
    Take an optimiser-side theta (length len(active_param_names(mode))) and
    return a full-length theta of size N_COUNTRY_PARAMS, filling alpha-only
    params from the legacy country_wise/ values when in alpha-frozen mode.
    """
    reduced = np.asarray(reduced_theta, dtype=float)
    free_idx = _free_indices_for_mode(mode)
    if reduced.shape[-1] == N_COUNTRY_PARAMS and mode != "alpha_frozen":
        return reduced.astype(float).copy()
    if reduced.shape[-1] != len(free_idx):
        raise ValueError(
            f"expand_theta: reduced theta has length {reduced.shape[-1]} but "
            f"mode {mode!r} expects {len(free_idx)}."
        )
    full = np.zeros(N_COUNTRY_PARAMS, dtype=float)
    for slot, theta_idx in enumerate(free_idx):
        full[theta_idx] = reduced[slot]
    if mode == "alpha_frozen":
        legacy = _legacy_fixed_for_idx(country_idx)
        if legacy is None:
            raise RuntimeError(
                f"alpha-frozen mode requires legacy country_wise/ values for "
                f"country_idx={country_idx} but none were found in "
                f"{LEGACY_COUNTRY_WISE_SUMMARY}."
            )
        for name, value in legacy.items():
            full[COUNTRY_PARAM_NAMES.index(name)] = value
    return full


def reduce_theta(full_theta, mode):
    """
    Inverse of expand_theta — slice out the optimiser-visible entries.
    """
    full = np.asarray(full_theta, dtype=float)
    free_idx = _free_indices_for_mode(mode)
    return full[list(free_idx)].astype(float).copy()


def _country_seed_theta(country_code, mode=None):
    """
    Return a clipped seed theta for the given country drawn from the legacy
    summary, or None if unavailable. In alpha-frozen mode the seed is
    reduced to the optimiser-visible dimensions.
    """
    base = _LEGACY_COUNTRY_SEEDS.get(country_code)
    if base is None:
        return None
    lower = np.array([b[0] for b in COUNTRY_PARAM_BOUNDS], dtype=float)
    upper = np.array([b[1] for b in COUNTRY_PARAM_BOUNDS], dtype=float)
    full = np.clip(base.astype(float), lower, upper)
    if mode is None:
        mode = _MODE
    return reduce_theta(full, mode)


def _timed_eval_worker(func, theta, conn):
    """
    Evaluate func(theta) and send result or exception back through conn.
    """
    os.environ["_CALIB_SUBPROCESS"] = "1"
    try:
        conn.send(("ok", float(func(theta))))
    except Exception as exc:
        conn.send(("err", repr(exc)))
    finally:
        conn.close()


class _TimedProcessMap:
    """
    Timeout safe for SciPy differential evolution.
    Each candidate is evaluated in an isolated subprocess so a stuck ODE solve
    can be killed without blocking the whole optimizer.
    """

    def __init__(self, max_workers, timeout_s, penalty):
        self.max_workers = max(1, int(max_workers))
        self.timeout_s = float(timeout_s)
        self.penalty = float(penalty)
        methods = mp.get_all_start_methods()
        method = "fork" if "fork" in methods else methods[0]
        self.ctx = mp.get_context(method)
        self.last_results = {}

    def __call__(self, func, iterable):
        """
        Evaluate func on each element of iterable with a timeout.
        """
        thetas = [np.asarray(theta, dtype=np.float64).copy() for theta in iterable]
        if not thetas:
            return []

        results = [self.penalty] * len(thetas)
        active = {}
        next_idx = 0
        completed = 0
        last_report = time.time()

        while completed < len(thetas):
            while next_idx < len(thetas) and len(active) < self.max_workers:
                parent_conn, child_conn = self.ctx.Pipe(duplex=False)
                proc = self.ctx.Process(
                    target=_timed_eval_worker,
                    args=(func, thetas[next_idx], child_conn),
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
                        print(f"    [de-eval-error] candidate={idx} err={payload}")
                    continue

                if now - state["start"] > self.timeout_s:
                    print(f"    [de-eval-timeout] candidate={idx} limit={self.timeout_s:.0f}s")
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
                    print(f"    [de-eval-error] candidate={idx} worker exited without result")

            if completed == len(thetas) or now - last_report >= 30.0:
                pct = 100 * completed / len(thetas)
                print(
                    f"    [DE batch] progress: {completed}/{len(thetas)} done ({pct:.0f}%) "
                    f"| {len(active)} running",
                    flush=True,
                )
                last_report = now

            if not progress:
                time.sleep(0.05)

        self.last_results = {
            _theta_key(theta): value for theta, value in zip(thetas, results)
        }
        return results


def load_country_index(extracted_dir=EXTRACTED_DIR):
    """
    Load country index from extracted data.
    """
    path = os.path.join(extracted_dir, "countries_index.csv")
    rows = []
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            row["position"] = int(row["position"])
            rows.append(row)
    return rows


def load_growth_regression_r(country_rows, r_values_path=R_VALUES_PATH):
    """
    Load growth-regression r-values for each country.
    """
    if not os.path.exists(r_values_path):
        raise FileNotFoundError(
            f"Growth-regression r-values not found: {r_values_path}"
        )

    by_code = {}
    with open(r_values_path, newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            by_code[row["country"]] = float(row["r_i"])

    missing = [
        row["location_code"] for row in country_rows
        if row["location_code"] not in by_code
    ]
    if missing:
        raise ValueError(
            "Missing growth-regression r-values for: " + ", ".join(sorted(missing))
        )

    return np.array(
        [by_code[row["location_code"]] for row in country_rows],
        dtype=float,
    )


def _target_row_normalized(alpha_row, fallback_row):
    """
    Normalize a row of alpha to sum to 1.
    """
    alpha_row = np.clip(np.asarray(alpha_row, dtype=float), 0.0, None)
    total = alpha_row.sum()
    if total > 0:
        return alpha_row / total
    fallback = np.clip(np.asarray(fallback_row, dtype=float), 0.0, None)
    fallback_total = fallback.sum()
    if fallback_total > 0:
        return fallback / fallback_total
    return fallback


def _r_vec_from_affine(reg_vec, a, b):
    """
    Apply the affine map r = a + b * (reg - mean(reg)) used by Stage 3 to
    rebuild model.r_C and model.r_P from the regression priors.
    """
    arr = np.asarray(reg_vec, dtype=float)
    centered = arr - float(np.mean(arr))
    return float(a) + float(b) * centered


def build_country_model(theta, data, country_idx):
    """
    Build a ProductSpaceModel where every Stage-3 global is also a free per-country parameter.
    """
    params = country_theta_to_dict(theta)

    model = ProductSpaceModel(
        N_products=data["SP"],
        n_countries=data["SC"],
        patch_network=True,
        seed=0,
        phi_space=data["phi_space"],
        s=float(FIXED["s"]),
        c=float(FIXED["c"]),
        c_prime=float(FIXED["c_prime"]),
        gamma=float(FIXED["gamma"]),
        kappa=float(params["kappa"]),
        sigma=float(params["sigma"]),
        nu=float(params["nu"]),
        G=float(params["G"]),
        q=float(FIXED["q"]),
        mu=float(FIXED["mu"]),
        beta_trade_off=float(params["beta_trade_off"]),
        enable_entry=bool(FIXED.get("enable_entry", False)),
        entry_threshold=float(params["entry_threshold"]),
    )
    _patch_model(
        model,
        data,
        params=None,
        h_C_mean=float(params["h_C"]),
        h_P_mean=float(params["h_P"]),
        s_pi=float(params["s_pi"]),
        s_pi_P=float(params["s_pi_P"]),
    )
    # r_C: regression vector for non-target countries; target's value is the
    # free per-country parameter r_C_target. Free a_C/b_C were dropped because
    # only the target's r_C reaches the loss (others get reset every year), so
    # a 2-parameter affine map was near-degenerate.
    r_C_vec = np.asarray(data["r_C_growth_regression"], dtype=float).copy()
    r_C_vec[int(country_idx)] = float(params["r_C_target"])
    model.r_C = r_C_vec
    # r_P from the per-country affine map a_P + b_P * (regression - mean).
    # _update_growth_rates short-circuits when data["r_P_regression"] is set,
    # so this value persists across years inside the simulator.
    r_P_reg = data.get("r_P_regression")
    if r_P_reg is None:
        raise RuntimeError(
            "Country-wise calibration requires data['r_P_regression'] now that "
            "a_P, b_P are free per-country parameters. Ensure "
            "calibration_results/growth_regression/r_P_regression.npy exists."
        )
    model.r_P = _r_vec_from_affine(r_P_reg, params["a_P"], params["b_P"])
    return model


def simulate_country_conditioned(
    theta,
    data,
    country_idx,
    years,
    start_year=None,
    max_time=TRAJECTORY_TIMEOUT_S,
    solve_timeout_s=SOLVE_TIMEOUT_S,
    mode=None,
):
    """
    Simulate the target country with the rest of the world conditioned on
    observed values.

    mode="conditioned" (default): the other 18 countries' C and alpha rows
        are pinned to observed each year; the target country's C and alpha
        come from the ODE.
    mode="alpha_frozen": every country's alpha row (target included) is
        pinned to observed each year; only the target country's C comes
        from the ODE. P is reconstructed as P_obs + delta_C * alpha_obs[target,:].
    If `mode` is None, the module-level `_MODE` is used (set by main()).
    """
    mode = _resolve_mode(mode)
    if start_year is None:
        start_year = YEAR_START

    model = build_country_model(theta, data, country_idx)
    alpha_start = data["alpha_obs"].get(start_year, model.alpha).astype(float).copy()
    C_start = data["C_obs"].get(start_year, data["C_init"]).astype(float).copy()
    P_start = data["P_obs"].get(start_year, data["P_init"]).astype(float).copy()
    y0 = np.concatenate([P_start, C_start, alpha_start.flatten()])
    results = {}
    prev_year = start_year
    t0 = time.time()

    try:
        with _wall_clock_timeout(max_time):
            for year in sorted(set(years)):
                if max_time and time.time() - t0 > max_time:
                    return None

                _update_growth_rates(model, data, year)
                n_yr = year - prev_year
                if n_yr <= 0:
                    C_cond = data["C_obs"][year].astype(float).copy()
                    alpha_cond = data["alpha_obs"][year].astype(float).copy()
                    P_cond = data["P_obs"][year].astype(float).copy()
                    results[year] = {
                        "country_share": float(C_cond[country_idx] / C_cond.sum()),
                        "country_total": float(C_cond[country_idx]),
                        "alpha_target": alpha_cond[country_idx].copy(),
                    }
                    y0 = np.concatenate([P_cond, C_cond, alpha_cond.flatten()])
                    prev_year = year
                    continue

                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        with _wall_clock_timeout(solve_timeout_s):
                            model.solve(
                                t_end=float(n_yr),
                                d_C=float(FIXED.get("d_C", 0.0)),
                                n_steps=SIM_STEPS_PER_YEAR * n_yr,
                                y0=y0,
                                max_solver_steps=MAX_SOLVER_STEPS,
                                rtol=1e-3,
                                atol=1e-6,
                            )
                except _SolveTimeout:
                    return None
                except Exception:
                    return None

                if model.y is None or model.y.shape[1] == 0:
                    return None
                if model.y_partial is None or model.y_partial.shape[1] == 0:
                    return None

                P_sim = np.maximum(model.y[:model.SP, -1], 0.0)
                C_sim = np.maximum(model.y[model.SP:model.N, -1], 0.0)
                alpha_sim = np.clip(
                    model.y_partial[:, -1].reshape(model.SC, model.SP),
                    0.0,
                    1.0,
                )

                if (
                    not np.all(np.isfinite(P_sim))
                    or not np.all(np.isfinite(C_sim))
                    or np.max(P_sim) > 1e6
                    or np.max(C_sim) > 1e6
                ):
                    return None

                C_cond = data["C_obs"][year].astype(float).copy()
                alpha_cond = data["alpha_obs"][year].astype(float).copy()
                C_cond[country_idx] = C_sim[country_idx]

                if mode == "conditioned":
                    alpha_cond[country_idx] = _target_row_normalized(
                        alpha_sim[country_idx],
                        data["alpha_obs"][year][country_idx],
                    )
                    exports_obs_target = (
                        data["C_obs"][year][country_idx]
                        * data["alpha_obs"][year][country_idx]
                    )
                    exports_sim_target = C_cond[country_idx] * alpha_cond[country_idx]
                    P_cond = (
                        data["P_obs"][year].astype(float).copy()
                        - exports_obs_target
                        + exports_sim_target
                    )
                else:  # alpha_frozen
                    # Target country's alpha row stays observed.
                    target_row = alpha_cond[country_idx]
                    delta_C = C_cond[country_idx] - data["C_obs"][year][country_idx]
                    P_cond = (
                        data["P_obs"][year].astype(float).copy()
                        + delta_C * target_row
                    )
                P_cond = np.clip(P_cond, 1e-12, None)

                model.activate_new_links(P_cond, C_cond, alpha_cond)

                results[year] = {
                    "country_share": float(C_cond[country_idx] / C_cond.sum()),
                    "country_total": float(C_cond[country_idx]),
                    "alpha_target": alpha_cond[country_idx].copy(),
                }

                y0 = np.concatenate([P_cond, C_cond, alpha_cond.flatten()])
                prev_year = year
    except _SolveTimeout:
        return None

    return results


def windowed_simulate_country_conditioned(theta, data, country_idx, years, start_year=None, mode=None):
    """
    Legacy compatibility wrapper for the country-wise trajectory simulator.

    Country-wise calibration now uses one continuous run with no intermediate
    re-initialisation, even though several call sites still use the historical
    "windowed" function name.
    """
    return simulate_country_conditioned(
        theta,
        data,
        country_idx,
        years,
        start_year=start_year,
        mode=mode,
    )


def compute_country_year_stats(year_result, data, country_idx, year):
    """
    Compute loss components for a single country-year.
    """
    share_sim = float(year_result["country_share"])
    C_obs = data["C_obs"][year]
    share_obs = float(C_obs[country_idx] / C_obs.sum())

    eps = 1e-10
    rmse_C = abs(np.log(share_sim + eps) - np.log(share_obs + eps))
    naive_C = abs(np.log(1.0 / data["SC"] + eps) - np.log(share_obs + eps))
    nrmse_C = rmse_C / naive_C if naive_C > 0 else rmse_C

    alpha_sim = np.asarray(year_result["alpha_target"], dtype=float)
    alpha_obs = np.asarray(data["alpha_obs"][year][country_idx], dtype=float)
    alpha_sim = _target_row_normalized(alpha_sim, alpha_obs)
    alpha_obs = _target_row_normalized(alpha_obs, alpha_obs)

    rmse_P = float(np.sqrt(np.mean((np.log(alpha_sim + eps) - np.log(alpha_obs + eps)) ** 2)))
    n_products = float(len(alpha_obs))
    naive_P = float(
        np.sqrt(np.mean((np.log(1.0 / n_products + eps) - np.log(alpha_obs + eps)) ** 2))
    )
    nrmse_P = rmse_P / naive_P if naive_P > 0 else rmse_P
    rank_products = 1.0 - spearman(alpha_sim, alpha_obs)

    return {
        "nrmse_C": float(nrmse_C),
        "nrmse_P": nrmse_P,
        "rank_products": float(rank_products),
        "share_sim": share_sim,
        "share_obs": share_obs,
    }


def aggregate_country_loss(results, data, country_idx, years=None, mode=None):
    """
    Aggregate loss components across years for a single country.
    """
    mode = _resolve_mode(mode)
    if years is None:
        years = COUNTRY_CALIB_YEARS

    totals = {"nrmse_C": 0.0, "nrmse_P": 0.0, "rank_products": 0.0}
    total_weight = 0.0
    share_sim_series = []
    share_obs_series = []
    alpha_sim_series = []
    alpha_obs_series = []

    for year in sorted(years):
        if year not in results:
            continue
        stats = compute_country_year_stats(results[year], data, country_idx, year)
        weight = 1.0 + TIME_WEIGHT_SLOPE * (year - YEAR_START)
        for key in totals:
            totals[key] += weight * stats[key]
        total_weight += weight
        share_sim_series.append(stats["share_sim"])
        share_obs_series.append(stats["share_obs"])
        alpha_obs_year = np.asarray(data["alpha_obs"][year][country_idx], dtype=float)
        alpha_sim_year = _target_row_normalized(
            np.asarray(results[year]["alpha_target"], dtype=float),
            alpha_obs_year,
        )
        alpha_obs_year = _target_row_normalized(alpha_obs_year, alpha_obs_year)
        alpha_sim_series.append(alpha_sim_year)
        alpha_obs_series.append(alpha_obs_year)

    if total_weight == 0:
        return None

    aggregated = {}
    for key in totals:
        aggregated[key] = float(totals[key] / total_weight)

    if len(share_sim_series) >= 3:
        aggregated["traj_corr_C"] = 1.0 - spearman(
            np.asarray(share_sim_series, dtype=float),
            np.asarray(share_obs_series, dtype=float),
        )
    else:
        aggregated["traj_corr_C"] = 1.0

    if len(alpha_sim_series) >= 3:
        alpha_sim_mat = np.asarray(alpha_sim_series, dtype=float)
        alpha_obs_mat = np.asarray(alpha_obs_series, dtype=float)
        obs_std = alpha_obs_mat.std(axis=0)
        obs_mean = alpha_obs_mat.mean(axis=0)
        valid = obs_std > 1e-12
        if np.any(valid):
            corrs = np.array(
                [spearman(alpha_sim_mat[:, i], alpha_obs_mat[:, i]) for i in np.where(valid)[0]],
                dtype=float,
            )
            weights = obs_mean[valid]
            weight_sum = float(weights.sum())
            if weight_sum > 0:
                aggregated["traj_corr_alpha"] = 1.0 - float((corrs * weights).sum() / weight_sum)
            else:
                aggregated["traj_corr_alpha"] = 1.0 - float(corrs.mean())
        else:
            aggregated["traj_corr_alpha"] = 1.0
    else:
        aggregated["traj_corr_alpha"] = 1.0

    # Use mode-aware loss weights
    mode_weights = _get_mode_aware_loss_weights(mode)
    aggregated["loss_total"] = float(
        mode_weights["nrmse_C"] * aggregated["nrmse_C"]
        + mode_weights["traj_corr_C"] * aggregated["traj_corr_C"]
        + mode_weights["rank_products"] * aggregated["rank_products"]
        + mode_weights["nrmse_P"] * aggregated["nrmse_P"]
    )
    return aggregated


def country_trajectory_loss(theta, data, country_idx, years=None, mode=None):
    """
    Run one continuous no-reinit simulation for one country and return its
    scalar loss.

    `theta` is reduced (length len(active_param_names(mode))) — alpha-only
    params are pinned per-country to the legacy country_wise/ values. The
    full N_COUNTRY_PARAMS-length theta is reconstructed before simulation.
    """
    mode = _resolve_mode(mode)
    if years is None:
        years = COUNTRY_CALIB_YEARS
    full_theta = expand_theta(theta, country_idx, mode)
    if not satisfies_country_constraints(full_theta):
        return PENALTY
    traj = windowed_simulate_country_conditioned(
        full_theta,
        data,
        country_idx,
        years,
        mode=mode,
    )
    if traj is None:
        return PENALTY
    aggregated = aggregate_country_loss(
        traj,
        data,
        country_idx,
        years=years,
        mode=mode,
    )
    if aggregated is None:
        return PENALTY
    return aggregated["loss_total"]


def _country_eval_worker(theta, data, country_idx, years, mode, conn):
    """
    Evaluate country_trajectory_loss in a subprocess and send result back through conn.
    """
    os.environ["_CALIB_SUBPROCESS"] = "1"
    try:
        loss = float(
            country_trajectory_loss(
                theta,
                data,
                country_idx,
                years=years,
                mode=mode,
            )
        )
        conn.send(("ok", loss))
    except Exception as exc:
        conn.send(("err", repr(exc)))
    finally:
        conn.close()


class TimedCountryBatchEvaluator:
    """
    Evaluate a batch of candidate thetas for a country with timeouts.
    """
    def __init__(self, max_workers, timeout_s, penalty):
        methods = mp.get_all_start_methods()
        method = "fork" if "fork" in methods else methods[0]
        self.ctx = mp.get_context(method)
        self.max_workers = max(1, int(max_workers))
        self.timeout_s = float(timeout_s)
        self.penalty = float(penalty)

    def evaluate(self, candidates, data, country_idx, years=None, mode=None):
        thetas = [np.asarray(theta, dtype=np.float64).copy() for theta in candidates]
        if not thetas:
            return np.empty(0, dtype=np.float64)
        mode = _resolve_mode(mode)

        results = np.full(len(thetas), self.penalty, dtype=np.float64)
        active = {}
        next_idx = 0
        completed = 0
        last_report = time.time()

        while completed < len(thetas):
            while next_idx < len(thetas) and len(active) < self.max_workers:
                parent_conn, child_conn = self.ctx.Pipe(duplex=False)
                proc = self.ctx.Process(
                    target=_country_eval_worker,
                    args=(thetas[next_idx], data, country_idx, years, mode, child_conn),
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
                    continue

                if now - state["start"] > self.timeout_s:
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

            if completed == len(thetas) or now - last_report >= 30.0:
                pct = 100 * completed / len(thetas)
                print(
                    f"    [LHS batch] progress: {completed}/{len(thetas)} done ({pct:.0f}%) "
                    f"| {len(active)} running",
                    flush=True,
                )
                last_report = now

            if not progress:
                time.sleep(0.05)

        return results


def evaluate_country_batch(candidates, data, country_idx, years=None, n_jobs=1, mode=None):
    evaluator = TimedCountryBatchEvaluator(
        max_workers=n_jobs,
        timeout_s=TRAJECTORY_TIMEOUT_S,
        penalty=PENALTY,
    )
    return evaluator.evaluate(candidates, data, country_idx, years=years, mode=mode)


def nroy_bounding_box(points, bounds, padding):
    """
    Compute a bounding box around the given points, clipped to the specified bounds
    and expanded by a padding fraction.
    """
    lower = points.min(axis=0)
    upper = points.max(axis=0)
    width = upper - lower
    lower = lower - padding * width
    upper = upper + padding * width
    clipped = []
    for idx, (base_lower, base_upper) in enumerate(bounds):
        clipped.append(
            (
                float(max(lower[idx], base_lower)),
                float(min(upper[idx], base_upper)),
            )
        )
    return clipped


def lhs_maximin(bounds, n_samples, seed):
    """
    Generate a Latin hypercube sample.
    """
    sampler = LatinHypercube(d=len(bounds), seed=seed, optimization="random-cd")
    unit = sampler.random(n=n_samples)
    lower = np.array([b[0] for b in bounds], dtype=float)
    upper = np.array([b[1] for b in bounds], dtype=float)
    return lower + unit * (upper - lower)


def run_country_lhs(country_dir, country_code, country_idx, data, n_jobs, lhs_samples):
    """
    Run LHS for a specific country in the optimiser-visible (mode-aware) space.
    In alpha-frozen mode, G/nu/entry_threshold are pinned per-country (legacy
    country_wise/ values) and not searched.
    """
    os.makedirs(country_dir, exist_ok=True)

    bounds_active = active_param_bounds(_MODE)
    names_active = active_param_names(_MODE)

    print(f"  LHS: evaluating {lhs_samples} samples with {n_jobs} workers "
          f"(d={len(bounds_active)}, mode={_MODE})", flush=True)
    theta_all = lhs_maximin(
        bounds_active,
        lhs_samples,
        SEED + 17 * country_idx,
    )
    seed_theta = _country_seed_theta(country_code, mode=_MODE)
    if seed_theta is not None and len(theta_all) > 0:
        theta_all[0] = seed_theta
    loss_all = evaluate_country_batch(
        theta_all,
        data,
        country_idx,
        years=COUNTRY_CALIB_YEARS,
        n_jobs=n_jobs,
    )

    finite = loss_all < PENALTY * 0.1
    if not np.any(finite):
        raise RuntimeError(f"{country_code}: LHS produced no finite evaluations.")

    finite_theta = theta_all[finite]
    finite_loss = loss_all[finite]
    n_elite = max(1, int(np.ceil(FAST_LHS_ELITE_FRACTION * len(finite_theta))))
    elite = finite_theta[np.argsort(finite_loss)[:n_elite]]
    bounds = nroy_bounding_box(elite, bounds_active, FAST_LHS_PADDING)

    print(
        f"  LHS complete: {len(finite_theta)}/{len(theta_all)} finite | "
        f"best={finite_loss.min():.5f}",
        flush=True,
    )

    np.save(os.path.join(country_dir, "lhs_theta_all.npy"), theta_all)
    np.save(os.path.join(country_dir, "lhs_loss_all.npy"), loss_all)
    with open(os.path.join(country_dir, "lhs_nroy_bounds.json"), "w") as handle:
        json.dump({"params": names_active, "bounds": bounds}, handle, indent=2)

    return {
        "theta_all": theta_all,
        "loss_all": loss_all,
        "finite_theta": finite_theta,
        "finite_loss": finite_loss,
        "nroy_bounds": bounds,
    }


def run_country_de(country_dir, country_idx, data, lhs_result, n_jobs):
    """
    Run DE for a specific country, seeding it with LHS results.
    """
    bounds = lhs_result["nroy_bounds"]
    finite_theta = lhs_result["finite_theta"]
    finite_loss = lhs_result["finite_loss"]

    n_active_params = len(active_param_names(_MODE))
    population_size = FAST_DE_POPSIZE * n_active_params
    de_workers = min(max(1, int(n_jobs)), population_size)
    seed = SEED + 101 * country_idx
    rng = np.random.default_rng(seed)
    timed_map = _TimedProcessMap(
        max_workers=de_workers,
        timeout_s=TRAJECTORY_TIMEOUT_S,
        penalty=PENALTY,
    )

    if len(finite_theta) >= population_size:
        n_best = population_size // 2
        best_idx = np.argsort(finite_loss)[:n_best]
        rand_idx = rng.choice(len(finite_theta), population_size - n_best, replace=False)
        init = np.vstack([finite_theta[best_idx], finite_theta[rand_idx]])
        rows = load_country_index()
        seed_theta = _country_seed_theta(rows[country_idx]["location_code"], mode=_MODE)
        if seed_theta is not None:
            init[0] = seed_theta
    else:
        init = "latinhypercube"

    print(
        f"  DE: population={population_size} workers={de_workers} "
        f"maxiter={FAST_DE_MAXITER} tol={FAST_DE_TOL}",
        flush=True,
    )
    de_log = []
    de_gen = [0]
    de_t0 = [time.time()]
    de_best = [np.inf]

    def callback(xk, convergence):
        """
        Callback to log DE progress after each generation.
        """
        de_gen[0] += 1
        elapsed = time.time() - de_t0[0]
        loss = timed_map.last_results.get(_theta_key(xk))
        if loss is not None and loss < de_best[0]:
            de_best[0] = float(loss)
        print(
            f"  DE gen {de_gen[0]:3d} | best={de_best[0]:.5f} | "
            f"conv={float(convergence):.4f} | {elapsed:.0f}s",
            flush=True,
        )
        de_log.append(
            {
                "generation": int(de_gen[0]),
                "best_loss": float(de_best[0]),
                "convergence": float(convergence),
                "time_s": float(elapsed),
            }
        )

    result = differential_evolution(
        country_trajectory_loss,
        bounds=bounds,
        args=(data, country_idx, COUNTRY_CALIB_YEARS),
        seed=seed,
        init=init,
        popsize=FAST_DE_POPSIZE,
        maxiter=FAST_DE_MAXITER,
        tol=FAST_DE_TOL,
        atol=0.0,
        mutation=FAST_DE_MUTATION,
        recombination=0.7,
        polish=False,
        updating="deferred",
        workers=timed_map,
        disp=False,
        callback=callback,
    )

    # result.x lives in optimiser-visible space; expand to the full
    # N_COUNTRY_PARAMS-length vector (filling alpha-only params from the
    # legacy fit when in alpha-frozen mode) before reporting and simulating.
    reduced_x = np.asarray(result.x, dtype=float)
    best_theta = expand_theta(reduced_x, country_idx, _MODE)
    best_loss = float(result.fun)
    best_traj = windowed_simulate_country_conditioned(best_theta, data, country_idx, COUNTRY_CALIB_YEARS)
    aggregated = None
    if best_traj is not None:
        aggregated = aggregate_country_loss(
            best_traj,
            data,
            country_idx,
            years=COUNTRY_CALIB_YEARS,
        )

    np.save(os.path.join(country_dir, "best_theta.npy"), best_theta)
    with open(os.path.join(country_dir, "de_result.json"), "w") as handle:
        json.dump(
            {
                "mode": _MODE,
                "country_idx": country_idx,
                "best_loss": best_loss,
                "message": str(result.message),
                "nfev": int(result.nfev),
                "nit": int(result.nit),
                "params": {
                    name: float(value) for name, value in zip(COUNTRY_PARAM_NAMES, best_theta)
                },
                "loss_components": aggregated,
                "de_log": de_log,
            },
            handle,
            indent=2,
        )

    return {
        "best_theta": best_theta,
        "best_loss": best_loss,
        "message": str(result.message),
        "nfev": int(result.nfev),
        "nit": int(result.nit),
        "loss_components": aggregated,
    }


def calibrate_country(country_row, data, n_jobs, lhs_samples):
    """
    Calibrate the model for a single country, running LHS followed by DE.
    """
    country_idx = int(country_row["position"])
    country_code = country_row["location_code"]
    country_name = country_row["country_name_short"]
    country_dir = os.path.join(COUNTRY_WISE_DIR, country_code)

    print(f"\n{'=' * 64}")
    print(f"Country-wise calibration: {country_code} ({country_name})")
    print(f"{'=' * 64}")

    lhs_result = run_country_lhs(
        country_dir=country_dir,
        country_code=country_code,
        country_idx=country_idx,
        data=data,
        n_jobs=n_jobs,
        lhs_samples=lhs_samples,
    )
    de_result = run_country_de(country_dir, country_idx, data, lhs_result, n_jobs=n_jobs)

    summary = {
        "country_idx": country_idx,
        "country_code": country_code,
        "country_name": country_name,
        "best_loss": de_result["best_loss"],
        "nfev": de_result["nfev"],
        "nit": de_result["nit"],
        "message": de_result["message"],
    }
    for name, value in zip(COUNTRY_PARAM_NAMES, de_result["best_theta"]):
        summary[name] = float(value)
    if de_result["loss_components"] is not None:
        for key, value in de_result["loss_components"].items():
            summary[key] = float(value)

    print(f"Best loss: {de_result['best_loss']:.5f}")
    for name, value in zip(COUNTRY_PARAM_NAMES, de_result["best_theta"]):
        print(f"  {name:18s} = {value:.5f}")
    return summary, de_result["best_theta"]


def write_summary(summary_rows):
    """
    Write a summary CSV of the calibration results for all countries.
    """
    os.makedirs(COUNTRY_WISE_DIR, exist_ok=True)
    path = os.path.join(COUNTRY_WISE_DIR, "summary.csv")
    fieldnames = [
        "country_idx",
        "country_code",
        "country_name",
        "best_loss",
        "nfev",
        "nit",
        "message",
        *COUNTRY_PARAM_NAMES,
        "nrmse_C",
        "traj_corr_C",
        "rank_products",
        "nrmse_P",
        "loss_total",
    ]
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)
    return path


def load_saved_country_result(country_row):
    """
    Load saved calibration result for a country.
    """
    country_idx = int(country_row["position"])
    country_code = country_row["location_code"]
    country_name = country_row["country_name_short"]
    country_dir = os.path.join(COUNTRY_WISE_DIR, country_code)
    de_result_path = os.path.join(country_dir, "de_result.json")
    best_theta_path = os.path.join(country_dir, "best_theta.npy")

    if not (os.path.exists(de_result_path) and os.path.exists(best_theta_path)):
        return None

    with open(de_result_path) as handle:
        de_result = json.load(handle)
    best_theta = np.load(best_theta_path)

    summary = {
        "country_idx": country_idx,
        "country_code": country_code,
        "country_name": country_name,
        "best_loss": float(de_result["best_loss"]),
        "nfev": int(de_result["nfev"]),
        "nit": int(de_result["nit"]),
        "message": str(de_result["message"]),
    }
    params = de_result.get("params", {})
    for idx, name in enumerate(COUNTRY_PARAM_NAMES):
        if name in params:
            summary[name] = float(params[name])
        else:
            summary[name] = float(best_theta[idx])

    loss_components = de_result.get("loss_components")
    if loss_components is not None:
        for key, value in loss_components.items():
            summary[key] = float(value)

    return summary, best_theta


def _collect_country_trajectories(best_thetas, country_rows, data):
    """
    Simulate every country with its own calibrated theta and return a dict
    of per-country observed/simulated shares for calibration and validation
    windows. Used by both the per-country plot and the summary plot.
    """
    collected = {}
    for row in country_rows:
        country_idx = int(row["position"])
        country_code = row["location_code"]
        if country_code not in best_thetas:
            continue

        traj_calib = windowed_simulate_country_conditioned(
            best_thetas[country_code],
            data,
            country_idx,
            COUNTRY_CALIB_YEARS,
        )
        traj_valid = simulate_country_conditioned(
            best_thetas[country_code],
            data,
            country_idx,
            COUNTRY_VALID_YEARS,
            start_year=COUNTRY_VALID_YEARS[0],
        )

        calib_years = [yr for yr in COUNTRY_CALIB_YEARS if traj_calib is not None and yr in traj_calib]
        valid_years = [yr for yr in COUNTRY_VALID_YEARS if traj_valid is not None and yr in traj_valid]

        obs_calib = [
            data["C_obs"][yr][country_idx] / data["C_obs"][yr].sum()
            for yr in calib_years
        ]
        sim_calib = [traj_calib[yr]["country_share"] for yr in calib_years] if traj_calib is not None else []
        obs_valid = [
            data["C_obs"][yr][country_idx] / data["C_obs"][yr].sum()
            for yr in valid_years
        ]
        sim_valid = [traj_valid[yr]["country_share"] for yr in valid_years] if traj_valid is not None else []

        collected[country_code] = {
            "country_idx": country_idx,
            "label": row["country_name_short"],
            "calib_years": calib_years,
            "valid_years": valid_years,
            "obs_calib": obs_calib,
            "sim_calib": sim_calib,
            "obs_valid": obs_valid,
            "sim_valid": sim_valid,
        }

    return collected


def plot_country_wise_trajectory_fit(collected, country_rows):
    """
    Plot observed vs simulated trajectories for each country in its own subplot.
    """
    n_cols = 5
    n_rows = 4
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 12), sharex=True)

    for idx, row in enumerate(country_rows):
        ax = axes.flat[idx]
        country_code = row["location_code"]
        entry = collected.get(country_code)
        if entry is None:
            ax.set_visible(False)
            continue

        calib_years = entry["calib_years"]
        valid_years = entry["valid_years"]
        obs_calib = entry["obs_calib"]
        sim_calib = entry["sim_calib"]
        obs_valid = entry["obs_valid"]
        sim_valid = entry["sim_valid"]

        rho_calib = (
            spearman(np.asarray(sim_calib, dtype=float), np.asarray(obs_calib, dtype=float))
            if len(calib_years) >= 3 else 0.0
        )
        rho_valid = (
            spearman(np.asarray(sim_valid, dtype=float), np.asarray(obs_valid, dtype=float))
            if len(valid_years) >= 3 else 0.0
        )

        # Observed: one continuous line across both periods
        all_years = calib_years + valid_years
        all_obs = obs_calib + obs_valid
        if all_years:
            ax.plot(all_years, all_obs, color="#1F4E79", lw=1.5, label="Observed")
        # Simulated: solid for calibration, dashed for validation
        if calib_years:
            ax.plot(calib_years, sim_calib, color="#B5541E", lw=1.5, label="Simulated")
        if valid_years:
            ax.plot(valid_years, sim_valid, color="#B5541E", lw=1.5, ls="--")

        ax.axvline(COUNTRY_CALIB_END, color="grey", ls=":", lw=0.8, alpha=0.7)
        ax.set_title(entry["label"], fontsize=9, fontweight="bold")
        ax.annotate(
            f"cal={rho_calib:.2f}  val={rho_valid:.2f}",
            xy=(0.03, 0.95),
            xycoords="axes fraction",
            fontsize=7,
            va="top",
            color="grey",
        )
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=7)

    for idx in range(len(country_rows), n_rows * n_cols):
        axes.flat[idx].set_visible(False)

    axes[0, 0].legend(fontsize=7, frameon=False)
    fig.supxlabel("Year", fontsize=11)
    fig.supylabel("Market share", fontsize=11)
    fig.suptitle(
        "Country-wise calibration: observed vs simulated trajectories",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    out_path = os.path.join(COUNTRY_WISE_DIR, "trajectory_fit.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved trajectory plot to: {out_path}")
    return out_path


def plot_country_wise_trajectory_summary(collected):
    """
    Summary plot mirroring validation/trajectory_summary.png.
    """
    all_years = sorted(set(COUNTRY_CALIB_YEARS + COUNTRY_VALID_YEARS))
    if not all_years:
        return None

    year_to_col = {yr: k for k, yr in enumerate(all_years)}
    n_countries = len(collected)
    if n_countries == 0:
        return None

    obs_mat = np.full((n_countries, len(all_years)), np.nan)
    sim_mat = np.full((n_countries, len(all_years)), np.nan)

    for i, (_, entry) in enumerate(collected.items()):
        for yr, v in zip(entry["calib_years"], entry["obs_calib"]):
            obs_mat[i, year_to_col[yr]] = v
        for yr, v in zip(entry["valid_years"], entry["obs_valid"]):
            obs_mat[i, year_to_col[yr]] = v
        for yr, v in zip(entry["calib_years"], entry["sim_calib"]):
            sim_mat[i, year_to_col[yr]] = v
        for yr, v in zip(entry["valid_years"], entry["sim_valid"]):
            sim_mat[i, year_to_col[yr]] = v

    fig, ax = plt.subplots(figsize=(10, 5))

    years_arr = np.asarray(all_years)
    for i in range(n_countries):
        ax.plot(years_arr, obs_mat[i], color="#4C72B0", alpha=0.08, lw=0.7)
        ax.plot(years_arr, sim_mat[i], color="#DD8452", alpha=0.08, lw=0.7)

    obs_median = np.nanmedian(obs_mat, axis=0)
    sim_median = np.nanmedian(sim_mat, axis=0)
    ax.plot(years_arr, obs_median, color="#1F4E79", lw=2.5, label="Observed median")
    ax.plot(years_arr, sim_median, color="#B5541E", lw=2.5, label="Simulated median")
    ax.axvline(COUNTRY_CALIB_END, color="grey", ls=":", lw=1, alpha=0.7, label="Calibration boundary")

    ax.set_xlabel("Year")
    ax.set_ylabel("Market share")
    ax.set_title(
        "Country market shares: all trajectories (country-wise calibration)",
        fontsize=12,
        fontweight="bold",
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=9, frameon=False)
    plt.tight_layout()
    out_path = os.path.join(COUNTRY_WISE_DIR, "trajectory_summary.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved trajectory summary plot to: {out_path}")
    return out_path


def parse_args():
    """
    Parser to pass arguments.
    """
    parser = argparse.ArgumentParser(description="Fast country-wise calibration")
    parser.add_argument(
        "--countries",
        nargs="*",
        help="Country codes to calibrate. Default: all countries in countries_index.csv",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=min(8, MAX_PARALLEL_JOBS),
        help="Parallel workers for LHS evaluation",
    )
    parser.add_argument(
        "--lhs-samples",
        type=int,
        default=FAST_LHS_N_SAMPLES,
        help="Number of LHS samples per country",
    )
    parser.add_argument(
        "--countries-in-parallel",
        type=int,
        default=1,
        help=(
            "Number of countries to calibrate concurrently. Each concurrent "
            "country gets n_jobs // countries_in_parallel inner workers."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=["conditioned", "alpha_frozen"],
        default="conditioned",
        help=(
            "conditioned: target country's C and alpha row come from the ODE; "
            "alpha_frozen: every country's alpha is pinned to observed each year, "
            "only target country's C comes from the ODE."
        ),
    )
    return parser.parse_args()


# Loaded once and shared across all parallel country workers.
_SHARED_DATA = None


class _PrefixedStream:
    """
    Prefix each line printed with the country code.
    """

    def __init__(self, stream, prefix):
        self._stream = stream
        self._prefix = prefix
        self._buf = ""

    def write(self, text):
        if not text:
            return 0
        self._buf += text
        if "\n" in self._buf:
            lines = self._buf.split("\n")
            self._buf = lines[-1]
            for line in lines[:-1]:
                self._stream.write(f"{self._prefix}{line}\n")
            self._stream.flush()
        return len(text)

    def flush(self):
        if self._buf:
            self._stream.write(f"{self._prefix}{self._buf}")
            self._buf = ""
        self._stream.flush()

    def isatty(self):
        return False

    def __getattr__(self, name):
        return getattr(self._stream, name)


def _country_pool_worker(country_row, inner_n_jobs, lhs_samples, conn):
    """
    Worker function to calibrate a single country in a subprocess, sending results
    back through conn.
    """
    tag = f"[{country_row['location_code']}] "
    sys.stdout = _PrefixedStream(sys.stdout, tag)
    sys.stderr = _PrefixedStream(sys.stderr, tag)
    try:
        result = calibrate_country(
            country_row=country_row,
            data=_SHARED_DATA,
            n_jobs=inner_n_jobs,
            lhs_samples=lhs_samples,
        )
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception:
            pass
        conn.send(("ok", result))
    except Exception as exc:
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception:
            pass
        conn.send(("err", f"{type(exc).__name__}: {exc}"))
    finally:
        conn.close()


def main():
    """
    Main function to run country-wise calibration.
    """
    args = parse_args()
    global _MODE, COUNTRY_WISE_DIR
    _MODE = args.mode
    COUNTRY_WISE_DIR = ALPHA_FROZEN_DIR if _MODE == "alpha_frozen" else CONDITIONED_DIR
    print(f"Mode: {_MODE}")
    print(f"Output dir: {COUNTRY_WISE_DIR}")

    n_jobs = max(1, min(args.n_jobs, MAX_PARALLEL_JOBS))
    countries_in_parallel = max(1, int(args.countries_in_parallel))
    if countries_in_parallel > 1:
        inner_n_jobs = max(1, n_jobs // countries_in_parallel)
        print(
            f"Inter-country parallelism: {countries_in_parallel} concurrent "
            f"countries x {inner_n_jobs} inner workers (total budget: {n_jobs})"
        )
    else:
        inner_n_jobs = n_jobs

    print("Loading empirical data...")
    data = load_data()
    country_rows = load_country_index()
    data["r_C_growth_regression"] = load_growth_regression_r(country_rows)

    if args.countries:
        wanted = {code.upper() for code in args.countries}
        country_rows = [row for row in country_rows if row["location_code"] in wanted]
        missing = sorted(wanted - {row["location_code"] for row in country_rows})
        if missing:
            raise ValueError(f"Unknown country codes: {', '.join(missing)}")

    os.makedirs(COUNTRY_WISE_DIR, exist_ok=True)

    summary_rows = []
    best_thetas = {}
    start_idx = 0
    if START_FROM_COUNTRY is not None:
        start_code = START_FROM_COUNTRY.upper()
        codes = [row["location_code"] for row in country_rows]
        if start_code not in codes:
            raise ValueError(f"START_FROM_COUNTRY not found in selected countries: {start_code}")
        start_idx = codes.index(start_code)
        print(f"Restarting from country: {start_code}", flush=True)

    for row in country_rows[:start_idx]:
        loaded = load_saved_country_result(row)
        if loaded is None:
            raise FileNotFoundError(
                f"Missing saved results for skipped country {row['location_code']}. "
                "Either restore those files or move START_FROM_COUNTRY earlier."
            )
        summary_row, best_theta = loaded
        summary_rows.append(summary_row)
        best_thetas[row["location_code"]] = best_theta

    pending_rows = country_rows[start_idx:]
    if countries_in_parallel > 1 and len(pending_rows) > 1:
        global _SHARED_DATA
        _SHARED_DATA = data
        ctx = mp.get_context("fork")

        def _run_one(row):
            """
            Start a subprocess to calibrate one country and return its state dict.
            """
            parent_conn, child_conn = ctx.Pipe(duplex=False)
            proc = ctx.Process(
                target=_country_pool_worker,
                args=(row, inner_n_jobs, args.lhs_samples, child_conn),
            )
            proc.start()
            child_conn.close()
            return {"row": row, "proc": proc, "conn": parent_conn}

        pending_iter = iter(pending_rows)
        active = []
        results_by_code = {}
        order = [row["location_code"] for row in pending_rows]

        while True:
            while len(active) < countries_in_parallel:
                try:
                    nxt = next(pending_iter)
                except StopIteration:
                    break
                active.append(_run_one(nxt))

            if not active:
                break

            progressed = False
            for slot in list(active):
                if slot["conn"].poll():
                    try:
                        status, payload = slot["conn"].recv()
                    except EOFError:
                        status, payload = ("err", "worker closed pipe without result")
                    slot["conn"].close()
                    slot["proc"].join(timeout=5.0)
                    active.remove(slot)
                    progressed = True
                    if status == "ok":
                        results_by_code[slot["row"]["location_code"]] = payload
                    else:
                        raise RuntimeError(
                            f"Country worker {slot['row']['location_code']} failed: {payload}"
                        )
                    continue
                if not slot["proc"].is_alive():
                    slot["conn"].close()
                    slot["proc"].join(timeout=1.0)
                    active.remove(slot)
                    progressed = True
                    raise RuntimeError(
                        f"Country worker {slot['row']['location_code']} exited without result"
                    )

            if not progressed:
                time.sleep(0.1)

        for code in order:
            summary_row, best_theta = results_by_code[code]
            summary_rows.append(summary_row)
            best_thetas[code] = best_theta
    else:
        for row in pending_rows:
            summary_row, best_theta = calibrate_country(
                country_row=row,
                data=data,
                n_jobs=inner_n_jobs,
                lhs_samples=args.lhs_samples,
            )
            summary_rows.append(summary_row)
            best_thetas[row["location_code"]] = best_theta

    summary_path = write_summary(summary_rows)
    collected = _collect_country_trajectories(best_thetas, country_rows, data)
    plot_country_wise_trajectory_fit(collected, country_rows)
    plot_country_wise_trajectory_summary(collected)
    print(f"\nSaved country-wise summary to: {summary_path}")


if __name__ == "__main__":
    main()
