"""
Stage 3: shared-global calibration on a windowed all-country simulation loss.

Free parameters (length 11):
    kappa, sigma, h_P, s_pi_P, a_C, b_C, a_P, b_P,
    beta_declining_delta, beta_rising_delta, beta_stable_delta

The product-product competition matrix C_PP is taken from
calibration_results/growth_regression/pi_competition_P.npy and scaled by
s_pi_P, replacing the legacy mean-field C_diag_mean / C_offdiag_mean
parameters on the product side.

Product growth rates are no longer the raw regression vector. Instead Stage 3
calibrates a global affine map:
    r_P_centered = r_P_regression - mean(r_P_regression)
    model.r_P    = a_P + b_P * r_P_centered
This keeps the relative cross-product structure from the regression while
letting the optimiser scale and shift the absolute growth level (the raw
regression has values up to ~+24 per year that destabilise the ODE).

Country growth rates also use a global affine map:
    r_C_centered = r_C_regression - mean(r_C_regression)
    model.r_C    = a_C + b_C * r_C_centered

Per-country vectors (s_pi, G, nu, h_C, entry_threshold, beta_trade_off)
are loaded from the existing independent country-wise summary and held fixed in
Stage 3. beta_trade_off was promoted from a
global shared parameter to per-country because (a) the independent country-wise
fits showed it spanning ~0.008–0.95 across countries, and (b) it acts on the
country-side degree K_j when building beta_C / beta_P, so it is naturally a
country-level capability scaling.

Optimisation target: Stage 3 minimises a weighted combination of:
  - a *windowed* all-country free-run loss (P, C, alpha re-seeded at
    WINDOW_REINIT_YEARS) — tractable, prevents error accumulation from
    dominating the optimisation
  - a *fully free* 26-year rollout loss — penalises candidates that only
    survive thanks to window resets (the previous pure-windowed run found a
    "uniform decay + strong competition" corner that fit windowed drift well
    but ruined free integration)
Default weights are STAGE3_W_WINDOW=0.7 and STAGE3_W_FREE=0.3. Set
STAGE3_W_FREE=0 (or STAGE3_WINDOWED=0) to recover the pure objectives for
diagnostics. Stage 4 (joint_simulation.py) always uses the fully free run for
final evaluation.

Per-candidate cost: two simulations (one windowed, one free). Stage 3 wall
clock roughly doubles relative to pure-windowed; STAGE3_TIMEOUT_S should be
sized accordingly.

Grouped coupling correction:
    Stage 3 additionally calibrates 3 offsets on country beta_trade_off, one
    per COUNTRY_GROUPS bucket (declining/rising/stable). This gives the joint
    run room to correct cross-country coupling with only 3 extra dimensions,
    and an L2 prior (STAGE3_BETA_REG_LAMBDA) keeps the corrected vector near
    the independent country-wise baseline.
"""
import csv
import functools
import json
import multiprocessing as mp
import os
import sys
import time
import warnings
import numpy as np
from scipy.optimize import differential_evolution

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from product_space_model import ProductSpaceModel
from calibration.calibration_config import (
    CALIB_DIR, EXTRACTED_DIR, FIXED, LOSS_WEIGHTS, MAX_PARALLEL_JOBS, MAX_SOLVER_STEPS,
    SEED, SIM_STEPS_PER_YEAR, SOLVE_TIMEOUT_S, TRAJECTORY_TIMEOUT_S,
    COUNTRY_GROUPS,
    WINDOW_REINIT_YEARS, YEAR_START,
)
from calibration.calibration_utils import (
    PENALTY, _patch_model, _wall_clock_timeout, _update_growth_rates,
    aggregate_loss_components, load_data,
)
from calibration.country_wise_calibration.calibration_country_wise import (
    COUNTRY_CALIB_END, COUNTRY_CALIB_YEARS, _load_bootstrap, load_country_index,
    load_growth_regression_r,
)

GLOBAL_PARAM_NAMES = [
    "kappa", "sigma", "h_P", "s_pi_P", "a_C", "b_C", "a_P", "b_P",
    "beta_declining_delta", "beta_rising_delta", "beta_stable_delta",
]
GLOBAL_PARAM_BOUNDS = [
    (0.0, 0.2),     # kappa  (widened from 0.05 — previous run reached 0.029)
    (0.01, 4.0),    # sigma
    (0.05, 5.0),    # h_P
    (1e-4, 50.0),   # s_pi_P (widened from 10.0 — previous run pegged at 9.37)
    # a_*/b_* bounds: tightened after the previous run pegged a_C=a_P at -2 and
    # collapsed b_C, b_P toward 0 (the optimiser found that uniform negative
    # growth + strong competition fits the windowed objective better than using
    # the regression's cross-country / cross-product structure). The new bounds
    # bracket the regression means (mean(r_C)≈0.14, mean(r_P)≈-0.22) so a_*
    # cannot be used as a "uniform decay" escape, and b_* is floored at 0.3 so
    # at least 30% of the regression's relative structure is always preserved.
    (-0.5, 0.5),    # a_C  (baseline country growth shift)
    (0.3, 1.0),     # b_C  (slope on centered r_C regression; floored to keep regression structure alive)
    (-0.5, 0.5),    # a_P  (baseline product growth shift)
    (0.3, 1.0),     # b_P  (slope on centered r_P regression; floored to keep regression structure alive)
    (-0.35, 0.35),  # beta_declining_delta
    (-0.35, 0.35),  # beta_rising_delta
    (-0.35, 0.35),  # beta_stable_delta
]

STAGE3_MODE = os.environ.get("STAGE3_MODE", "free").lower()
if STAGE3_MODE not in ("free", "alpha_frozen"):
    raise ValueError(
        f"STAGE3_MODE must be 'free' or 'alpha_frozen', got {STAGE3_MODE!r}"
    )
_DEFAULT_OUTPUT_DIR = os.path.join(
    CALIB_DIR, "joint_alpha_frozen" if STAGE3_MODE == "alpha_frozen" else "joint"
)
OUTPUT_DIR = os.environ.get("STAGE3_OUTPUT_DIR", _DEFAULT_OUTPUT_DIR)
GLOBAL_PARAMS_PATH = os.path.join(OUTPUT_DIR, "global_params.json")
DE_LOG_PATH = os.path.join(OUTPUT_DIR, "de_log.json")
if STAGE3_MODE == "alpha_frozen":
    _DEFAULT_COUNTRY_SUMMARY = os.path.join(
        CALIB_DIR, "country_wise_alpha_frozen", "summary.csv"
    )
else:
    _DEFAULT_COUNTRY_SUMMARY = os.path.join(
        CALIB_DIR, "country_wise_restricted", "summary.csv"
    )
RESTRICTED_SUMMARY = os.environ.get("STAGE3_COUNTRY_SUMMARY", _DEFAULT_COUNTRY_SUMMARY)

STAGE3_YEARS = COUNTRY_CALIB_YEARS

DE_N_CANDIDATES = int(os.environ.get("DE_N_CANDIDATES", "32"))
DE_MAXITER = int(os.environ.get("DE_MAXITER", "200"))
DE_TOL = float(os.environ.get("DE_TOL", "0.005"))
DE_WORKERS = int(os.environ.get("DE_WORKERS", "8"))
# Stage 3 objective. When STAGE3_W_FREE > 0, the loss becomes a weighted
# combination of the windowed (re-seeded every WINDOW_REINIT_YEARS) and the
# fully free 26-year rollout, so candidates that only score well under the
# windowed reset can't escape long-horizon drift. Stage 4 (joint_simulation.py)
# always uses the fully free run for final evaluation.
#
# Defaults: 0.7 / 0.3 — mostly windowed for tractable optimisation, with a
# 30% free-run penalty to penalise candidates that decay between resets.
# Set STAGE3_W_FREE=0 (and optionally STAGE3_WINDOWED=0) to revert to the old
# pure-windowed or pure-free objectives for diagnostics.
STAGE3_WINDOWED = os.environ.get("STAGE3_WINDOWED", "1") not in ("0", "false", "False")
STAGE3_W_WINDOW = float(os.environ.get("STAGE3_W_WINDOW", "0.7"))
STAGE3_W_FREE = float(os.environ.get("STAGE3_W_FREE", "0.3"))
STAGE3_BETA_REG_LAMBDA = float(os.environ.get("STAGE3_BETA_REG_LAMBDA", "0.1"))
if STAGE3_W_WINDOW < 0 or STAGE3_W_FREE < 0:
    raise ValueError("STAGE3_W_WINDOW and STAGE3_W_FREE must be non-negative")
if STAGE3_W_WINDOW + STAGE3_W_FREE <= 0:
    raise ValueError("At least one of STAGE3_W_WINDOW / STAGE3_W_FREE must be > 0")


def apply_beta_group_correction(base_beta, global_param_map):
    """
    Apply 3 grouped offsets to per-country beta_trade_off and clip to [0, 1].
    Group indices come from COUNTRY_GROUPS in calibration_config.py.
    """
    beta = np.asarray(base_beta, dtype=float).copy()
    deltas = {
        "declining": float(global_param_map.get("beta_declining_delta", 0.0)),
        "rising": float(global_param_map.get("beta_rising_delta", 0.0)),
        "stable": float(global_param_map.get("beta_stable_delta", 0.0)),
    }
    for group_name, idxs in COUNTRY_GROUPS.items():
        if idxs:
            beta[np.asarray(idxs, dtype=int)] += deltas.get(group_name, 0.0)
    return np.clip(beta, 0.0, 1.0)


def load_country_vectors():
    """
    Read country summary in countries_index order. Returns a dict of (SC,) arrays
    keyed by ('s_pi', 'G', 'nu', 'h_C', 'entry_threshold', 'beta_trade_off').

    Accepts the existing independent-run summary schema (with `h`) and the newer
    variant (with `h_C`). r_C is not expected here: Stage 3 derives model.r_C
    from the regression through the global affine map (a_C, b_C).
    Controlled by STAGE3_COUNTRY_SUMMARY env var.
    """
    rows = load_country_index()
    by_code = {}
    with open(RESTRICTED_SUMMARY, newline="") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []
        for row in reader:
            by_code[row["country_code"]] = row
    # Independent-run schema uses "h" for country-side saturation; restricted uses "h_C"
    h_col = "h_C" if "h_C" in headers else "h"
    required = ("beta_trade_off",)
    missing = [c for c in required if c not in headers]
    if missing:
        raise KeyError(
            f"{RESTRICTED_SUMMARY} is missing required column(s) {missing}. "
            "Stage 3 expects the independent country-wise summary with "
            "per-country beta_trade_off. Point STAGE3_COUNTRY_SUMMARY at a "
            "schema-compatible CSV."
        )
    SC = len(rows)
    out = {name: np.zeros(SC) for name in
           ("s_pi", "G", "nu", "h_C", "entry_threshold", "beta_trade_off")}
    for r in rows:
        code = r["location_code"]
        idx = int(r["position"])
        if code not in by_code:
            raise KeyError(f"Missing {code} in {RESTRICTED_SUMMARY}")
        row = by_code[code]
        out["s_pi"][idx] = float(row["s_pi"])
        out["G"][idx] = float(row["G"])
        out["nu"][idx] = float(row["nu"])
        out["h_C"][idx] = float(row[h_col])
        out["entry_threshold"][idx] = float(row["entry_threshold"])
        out["beta_trade_off"][idx] = float(row["beta_trade_off"])
    return out


def _r_C_from_affine(data, a_C, b_C):
    """
    Compute model.r_C from the global affine map a_C + b_C * centered regression.
    Errors loudly if the country regression vector is missing — Stage 3/4 require it.
    """
    r_C_reg = data.get("r_C_growth_regression")
    if r_C_reg is None:
        raise RuntimeError(
            "Stage 3 requires data['r_C_growth_regression'] to build the "
            "centered r_C prior. Ensure load_growth_regression_r(...) has been "
            "run and attached to the data dict before model construction."
        )
    r_C_centered = np.asarray(r_C_reg, dtype=float) - float(np.mean(r_C_reg))
    return float(a_C) + float(b_C) * r_C_centered


def _r_P_from_affine(data, a_P, b_P):
    """
    Compute model.r_P from the global affine map a_P + b_P * centered regression.
    Errors loudly if r_P_regression is missing — Stage 3 requires it.
    """
    r_P_reg = data.get("r_P_regression")
    if r_P_reg is None:
        raise RuntimeError(
            "Stage 3 requires data['r_P_regression'] (loaded from "
            "calibration_results/growth_regression/r_P_regression.npy) to build "
            "the centered r_P prior. The file appears to be missing."
        )
    r_P_centered = np.asarray(r_P_reg, dtype=float) - float(np.mean(r_P_reg))
    return float(a_P) + float(b_P) * r_P_centered


def build_joint_model(global_theta, data, country_vecs):
    g = dict(zip(GLOBAL_PARAM_NAMES, global_theta))
    if "beta_trade_off_corrected" in country_vecs:
        beta_trade_off = np.asarray(country_vecs["beta_trade_off_corrected"], dtype=float)
    else:
        beta_trade_off = apply_beta_group_correction(country_vecs["beta_trade_off"], g)
    model = ProductSpaceModel(
        N_products=data["SP"], n_countries=data["SC"], patch_network=True, seed=0,
        phi_space=data["phi_space"],
        s=float(FIXED["s"]), c=float(FIXED["c"]), c_prime=float(FIXED["c_prime"]),
        gamma=float(FIXED["gamma"]),
        kappa=float(g["kappa"]), sigma=float(g["sigma"]),
        nu=country_vecs["nu"], G=country_vecs["G"],
        q=float(FIXED["q"]), mu=float(FIXED["mu"]),
        beta_trade_off=beta_trade_off,
        enable_entry=bool(FIXED.get("enable_entry", False)),
        entry_threshold=country_vecs["entry_threshold"],
    )
    _patch_model(
        model, data, params=None,
        h_C_mean=country_vecs["h_C"], h_P_mean=float(g["h_P"]),
        s_pi=country_vecs["s_pi"], s_pi_P=float(g["s_pi_P"]),
    )
    model.r_C = _r_C_from_affine(data, g["a_C"], g["b_C"])
    # Global affine product growth: model.r_P = a_P + b_P * centered regression.
    # _update_growth_rates short-circuits whenever data['r_P_regression'] is
    # set, so this value persists across years inside the simulator.
    model.r_P = _r_P_from_affine(data, g["a_P"], g["b_P"])
    return model


def free_simulate_all_countries(model, data, years, start_year=YEAR_START,
                                solve_timeout_s=SOLVE_TIMEOUT_S,
                                max_time=TRAJECTORY_TIMEOUT_S, y0=None):
    """
    Free all-country forward simulation. No yearly conditioning on observed data.
    If y0 is None, builds initial state from data["P_init"], data["C_init"],
    model.alpha. Otherwise uses the provided y0 (length SP+SC+SC*SP).
    Returns dict {year: (alpha, C, P)} or None on failure/timeout.
    """
    t0 = time.time()
    if y0 is None:
        y0 = np.concatenate([data["P_init"], data["C_init"], model.alpha.flatten()])
    results = {}
    prev_year = start_year
    try:
        with _wall_clock_timeout(max_time):
            for year in sorted(set(years)):
                _update_growth_rates(model, data, year)
                n_yr = year - prev_year
                if n_yr <= 0:
                    P_t = np.maximum(y0[:model.SP], 0.0)
                    C_t = np.maximum(y0[model.SP:model.N], 0.0)
                    alpha_t = np.clip(
                        y0[model.N:].reshape(model.SC, model.SP), 0.0, 1.0,
                    )
                    results[year] = (alpha_t, C_t, P_t)
                    continue
                with warnings.catch_warnings():
                    # Stiff/exploding DE candidates trigger LSODA
                    # convergence-failure warnings; suppress so the slurm log
                    # stays readable. The candidate's loss is already gated by
                    # the finite/overflow checks below.
                    warnings.simplefilter("ignore")
                    with _wall_clock_timeout(solve_timeout_s):
                        model.solve(
                            t_end=float(n_yr), d_C=float(FIXED.get("d_C", 0.0)),
                            n_steps=SIM_STEPS_PER_YEAR * n_yr, y0=y0,
                            max_solver_steps=MAX_SOLVER_STEPS, rtol=1e-3, atol=1e-6,
                        )
                if model.y is None or model.y.shape[1] == 0:
                    return None
                if model.y_partial is None or model.y_partial.shape[1] == 0:
                    return None
                P_t = np.maximum(model.y[:model.SP, -1], 0.0)
                C_t = np.maximum(model.y[model.SP:model.N, -1], 0.0)
                alpha_t = np.clip(
                    model.y_partial[:, -1].reshape(model.SC, model.SP), 0.0, 1.0,
                )
                if not np.all(np.isfinite(P_t)) or not np.all(np.isfinite(C_t)):
                    return None
                if np.max(P_t) > 1e6 or np.max(C_t) > 1e6:
                    return None
                model.activate_new_links(P_t, C_t, alpha_t)
                results[year] = (alpha_t, C_t, P_t)
                y0 = np.concatenate([P_t, C_t, alpha_t.flatten()])
                prev_year = year
    except Exception:
        return None
    return results


def alpha_frozen_simulate_all_countries(model, data, years,
                                        start_year=YEAR_START,
                                        solve_timeout_s=SOLVE_TIMEOUT_S,
                                        max_time=TRAJECTORY_TIMEOUT_S):
    """
    Joint alpha-frozen simulation. Mirrors free_simulate_all_countries but
    re-pins every country's alpha row to data['alpha_obs'][year] at each
    year boundary. C and P continue from the ODE (no reset). This is the
    multi-country generalisation of the alpha-frozen mode used in
    calibration_country_wise.py / sim_injection.py.

    Returns dict {year: (alpha, C, P)} or None on failure.
    """
    alpha_obs = data.get("alpha_obs")
    if alpha_obs is None or start_year not in alpha_obs:
        raise RuntimeError(
            "alpha-frozen joint sim requires data['alpha_obs'] with entries at "
            "every calibration year"
        )
    alpha = np.asarray(alpha_obs[start_year], dtype=float).copy()
    y0 = np.concatenate([
        np.asarray(data["P_init"], dtype=float),
        np.asarray(data["C_init"], dtype=float),
        alpha.flatten(),
    ])
    results = {}
    prev_year = start_year
    t0 = time.time()
    try:
        with _wall_clock_timeout(max_time):
            for year in sorted(set(years)):
                _update_growth_rates(model, data, year)
                n_yr = year - prev_year
                if n_yr <= 0:
                    P_t = np.maximum(y0[:model.SP], 0.0)
                    C_t = np.maximum(y0[model.SP:model.N], 0.0)
                    alpha_t = np.clip(
                        y0[model.N:].reshape(model.SC, model.SP), 0.0, 1.0,
                    )
                    results[year] = (alpha_t, C_t, P_t)
                    continue
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    with _wall_clock_timeout(solve_timeout_s):
                        model.solve(
                            t_end=float(n_yr), d_C=float(FIXED.get("d_C", 0.0)),
                            n_steps=SIM_STEPS_PER_YEAR * n_yr, y0=y0,
                            max_solver_steps=MAX_SOLVER_STEPS, rtol=1e-3, atol=1e-6,
                        )
                if model.y is None or model.y.shape[1] == 0:
                    return None
                if model.y_partial is None or model.y_partial.shape[1] == 0:
                    return None
                P_t = np.maximum(model.y[:model.SP, -1], 0.0)
                C_t = np.maximum(model.y[model.SP:model.N, -1], 0.0)
                if not np.all(np.isfinite(P_t)) or not np.all(np.isfinite(C_t)):
                    return None
                if np.max(P_t) > 1e6 or np.max(C_t) > 1e6:
                    return None
                if year not in alpha_obs:
                    return None
                alpha_pinned = np.asarray(alpha_obs[year], dtype=float).copy()
                model.activate_new_links(P_t, C_t, alpha_pinned)
                results[year] = (alpha_pinned, C_t, P_t)
                y0 = np.concatenate([P_t, C_t, alpha_pinned.flatten()])
                prev_year = year
    except Exception:
        return None
    return results


def _alpha_frozen_loss_component(global_theta, data, country_vecs, years):
    try:
        model = build_joint_model(global_theta, data, country_vecs)
        traj = alpha_frozen_simulate_all_countries(model, data, years)
    except Exception:
        return PENALTY
    if traj is None:
        return PENALTY
    agg = aggregate_loss_components(traj, data, years=years)
    return float(agg["loss_total"]) if agg is not None else PENALTY


def windowed_free_simulate_all_countries(global_theta, data, country_vecs,
                                         years, start_year=YEAR_START,
                                         solve_timeout_s=SOLVE_TIMEOUT_S,
                                         max_time=TRAJECTORY_TIMEOUT_S,
                                         reinit_years=None):
    """
    Joint free simulation that resets P, C, alpha to observed values at
    WINDOW_REINIT_YEARS boundaries. This mirrors the windowed objective used by
    the country-wise calibration (see calibration_country_wise.windowed_simulate_country_conditioned)
    so error doesn't accumulate over the full 26-year rollout — Stage 3 needs a
    tractable optimisation target, not a fully free 1988→2014 trajectory.

    Returns dict {year: (alpha, C, P)} concatenated across windows, or None on
    failure/timeout.
    """
    if reinit_years is None:
        reinit_years = WINDOW_REINIT_YEARS
    sorted_years = sorted(set(years))
    if not sorted_years:
        return {}
    boundaries = [start_year]
    for ry in reinit_years:
        if start_year < ry <= sorted_years[-1]:
            boundaries.append(ry)
    boundaries = sorted(set(boundaries))

    t0 = time.time()
    all_results = {}
    try:
        with _wall_clock_timeout(max_time):
            for idx, w_start in enumerate(boundaries):
                w_end = boundaries[idx + 1] if idx + 1 < len(boundaries) else sorted_years[-1] + 1
                w_years = [y for y in sorted_years if w_start <= y < w_end]
                if not w_years:
                    continue

                # Re-seed from observed state at the start of each later window
                # and rebuild the model so mutable state (activated links, alpha-
                # dependent beta matrices) does not leak across windows.
                if (
                    w_start != start_year
                    and w_start in data.get("alpha_obs", {})
                    and w_start in data.get("C_obs", {})
                    and w_start in data.get("P_obs", {})
                ):
                    w_data = dict(data)
                    w_data["alpha_init"] = data["alpha_obs"][w_start].copy()
                    w_data["C_init"] = data["C_obs"][w_start].copy()
                    w_data["P_init"] = data["P_obs"][w_start].copy()
                else:
                    w_data = data

                remaining = max_time - (time.time() - t0)
                if remaining <= 0:
                    return None
                model = build_joint_model(global_theta, w_data, country_vecs)
                y0 = np.concatenate([
                    np.asarray(w_data["P_init"], dtype=float),
                    np.asarray(w_data["C_init"], dtype=float),
                    np.asarray(w_data["alpha_init"], dtype=float).flatten(),
                ])
                w_traj = free_simulate_all_countries(
                    model, w_data, w_years, start_year=w_start,
                    solve_timeout_s=solve_timeout_s,
                    max_time=remaining,
                    y0=y0,
                )
                if w_traj is None:
                    return None
                all_results.update(w_traj)
    except Exception:
        return None
    return all_results


def _windowed_loss_component(global_theta, data, country_vecs, years):
    try:
        traj = windowed_free_simulate_all_countries(
            global_theta, data, country_vecs, years,
        )
    except Exception:
        return PENALTY
    if traj is None:
        return PENALTY
    agg = aggregate_loss_components(traj, data, years=years)
    return float(agg["loss_total"]) if agg is not None else PENALTY


def _free_loss_component(global_theta, data, country_vecs, years):
    try:
        model = build_joint_model(global_theta, data, country_vecs)
        traj = free_simulate_all_countries(model, data, years)
    except Exception:
        return PENALTY
    if traj is None:
        return PENALTY
    agg = aggregate_loss_components(traj, data, years=years)
    return float(agg["loss_total"]) if agg is not None else PENALTY


def shared_global_loss(global_theta, data, country_vecs, years=None,
                       windowed=True, w_window=None, w_free=None):
    """
    Stage 3 objective. By default a weighted combination:
        loss = w_window * windowed_loss + w_free * free_loss
    where windowed_loss re-seeds P/C/alpha from observation every
    WINDOW_REINIT_YEARS, and free_loss is the fully-free 26-year rollout. The
    free term penalises candidates that only score well under window resets
    (the previous run found a degenerate "uniform decay + strong competition"
    corner that the windowed objective rewarded but free integration ruins).

    When `windowed=False` and w_free is None, falls back to a pure free run
    for diagnostics (equivalent to STAGE3_W_WINDOW=0, STAGE3_W_FREE=1).
    """
    if years is None:
        years = STAGE3_YEARS
    for v, (lo, hi) in zip(global_theta, GLOBAL_PARAM_BOUNDS):
        if not (lo <= v <= hi):
            return PENALTY

    if STAGE3_MODE == "alpha_frozen":
        # Alpha-frozen joint sim: alpha pinned to observed every year for all
        # countries, C and P evolve continuously from the ODE. No
        # windowed/free mix — every year boundary already re-pins alpha, so
        # the windowed reset would be redundant noise.
        loss_af = _alpha_frozen_loss_component(global_theta, data, country_vecs, years)
        if loss_af >= PENALTY:
            return PENALTY
        objective = loss_af
    else:
        # Resolve weights. Default to module-level STAGE3_W_WINDOW / STAGE3_W_FREE,
        # but if the legacy `windowed` flag is False and weights weren't given,
        # collapse to a pure free objective for backward-compatible diagnostics.
        if w_window is None and w_free is None:
            if windowed:
                w_window, w_free = STAGE3_W_WINDOW, STAGE3_W_FREE
            else:
                w_window, w_free = 0.0, 1.0
        elif w_window is None:
            w_window = 0.0
        elif w_free is None:
            w_free = 0.0

        total = 0.0
        if w_window > 0:
            loss_w = _windowed_loss_component(global_theta, data, country_vecs, years)
            if loss_w >= PENALTY:
                return PENALTY
            total += w_window * loss_w
        if w_free > 0:
            loss_f = _free_loss_component(global_theta, data, country_vecs, years)
            if loss_f >= PENALTY:
                return PENALTY
            total += w_free * loss_f
        norm = w_window + w_free
        objective = total / norm if norm > 0 else PENALTY

    # Keep grouped beta corrections near the independent per-country prior.
    # This allows structural coupling adjustment without drifting too far.
    if STAGE3_BETA_REG_LAMBDA > 0:
        g = dict(zip(GLOBAL_PARAM_NAMES, global_theta))
        beta_base = np.asarray(country_vecs["beta_trade_off"], dtype=float)
        beta_corr = apply_beta_group_correction(beta_base, g)
        beta_reg = float(np.mean((beta_corr - beta_base) ** 2))
        objective += STAGE3_BETA_REG_LAMBDA * beta_reg

    return objective


def _global_eval_worker(func, theta, conn):
    """Evaluate func(theta) in a subprocess and return result through conn."""
    os.environ["_CALIB_SUBPROCESS"] = "1"
    try:
        conn.send(("ok", float(func(theta))))
    except Exception as exc:
        conn.send(("err", repr(exc)))
    finally:
        conn.close()


class _TimedGlobalMap:
    """
    Subprocess-per-candidate evaluator for Stage 3 DE.
    Mirrors _TimedProcessMap from calibration_country_wise.py so that
    TRAJECTORY_TIMEOUT_S is enforced via process kill rather than SIGALRM
    (which is disabled in pool workers and therefore never fires).
    """

    def __init__(self, max_workers, timeout_s, penalty):
        self.max_workers = max(1, int(max_workers))
        self.timeout_s = float(timeout_s)
        self.penalty = float(penalty)
        methods = mp.get_all_start_methods()
        method = "fork" if "fork" in methods else methods[0]
        self.ctx = mp.get_context(method)

    def __call__(self, func, iterable):
        thetas = [np.asarray(t, dtype=np.float64).copy() for t in iterable]
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
                    target=_global_eval_worker,
                    args=(func, thetas[next_idx], child_conn),
                )
                proc.start()
                child_conn.close()
                active[next_idx] = {"proc": proc, "conn": parent_conn,
                                     "start": time.time()}
                next_idx += 1

            progress = False
            now = time.time()
            for idx, state in list(active.items()):
                proc, conn = state["proc"], state["conn"]
                if conn.poll():
                    try:
                        status, payload = conn.recv()
                    except EOFError:
                        status, payload = "err", "pipe closed"
                    conn.close()
                    proc.join(timeout=0.1)
                    active.pop(idx)
                    completed += 1
                    progress = True
                    if status == "ok" and np.isfinite(payload):
                        results[idx] = float(payload)
                    else:
                        print(f"    [stage3-eval] candidate={idx} err={payload}",
                              flush=True)
                    continue
                if now - state["start"] > self.timeout_s:
                    print(f"    [stage3-timeout] candidate={idx} "
                          f"limit={self.timeout_s:.0f}s", flush=True)
                    proc.kill() if hasattr(proc, "kill") else proc.terminate()
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
                print(f"    [stage3 batch] {completed}/{len(thetas)} done "
                      f"({pct:.0f}%) | {len(active)} running", flush=True)
                last_report = now

            if not progress:
                time.sleep(0.05)

        return results


def _build_seeded_init_population(rng):
    """
    Build the DE initial population: row 0 = bootstrap pack point (clipped to
    bounds), remaining rows = Latin-hypercube draw within bounds. Gives DE one
    known-good seed without committing the whole population to the bootstrap
    region (Finding 6: pure-random LHS init wastes the bootstrap pack;
    seeding the entire population from it would lock DE into the legacy basin).
    """
    boot = _load_bootstrap()
    pop_size = DE_N_CANDIDATES
    n_params = len(GLOBAL_PARAM_NAMES)
    if pop_size < 5:
        raise ValueError(f"DE_N_CANDIDATES must be >= 5, got {pop_size}")
    lo = np.array([b[0] for b in GLOBAL_PARAM_BOUNDS])
    hi = np.array([b[1] for b in GLOBAL_PARAM_BOUNDS])

    pack_point = np.array([
        float(boot[name]) if name in boot else 0.0
        for name in GLOBAL_PARAM_NAMES
    ])
    pack_point = np.clip(pack_point, lo, hi)

    # Latin hypercube fill for the remaining rows.
    n_lhs = pop_size - 1
    cuts = np.linspace(0.0, 1.0, n_lhs + 1)
    u = rng.uniform(size=(n_lhs, n_params))
    lhs = np.empty_like(u)
    for d in range(n_params):
        strata_lo = cuts[:-1]
        perm = rng.permutation(n_lhs)
        lhs[:, d] = strata_lo[perm] + u[:, d] * (1.0 / n_lhs)
    lhs_scaled = lo + lhs * (hi - lo)

    init = np.vstack([pack_point[None, :], lhs_scaled])
    return init, pack_point


def main():
    print(f"Stage 3 ({STAGE3_MODE} mode): shared-global calibration "
          f"on years {STAGE3_YEARS[0]}-{STAGE3_YEARS[-1]}")
    print(f"  country vectors from: {RESTRICTED_SUMMARY}")
    print(f"  output dir:           {OUTPUT_DIR}")
    data = load_data()
    rows = load_country_index()
    data["r_C_growth_regression"] = load_growth_regression_r(rows)
    country_vecs = load_country_vectors()
    if data.get("r_P_regression") is None:
        raise RuntimeError(
            "Stage 3 requires r_P_regression.npy in "
            "calibration_results/growth_regression/ to build the centered "
            "r_P prior used by the global a_P, b_P calibration."
        )
    print(f"  DE config: workers={DE_WORKERS} candidates={DE_N_CANDIDATES} "
          f"maxiter={DE_MAXITER} tol={DE_TOL}")
    print(f"  grouped beta regularizer: lambda={STAGE3_BETA_REG_LAMBDA}")
    if STAGE3_MODE == "alpha_frozen":
        print("  objective: alpha-frozen joint sim "
              f"(alpha pinned to observed every year, C+P evolve from ODE) "
              f"on {STAGE3_YEARS[0]}-{STAGE3_YEARS[-1]}")
    else:
        if STAGE3_WINDOWED:
            eff_w_w = STAGE3_W_WINDOW
            eff_w_f = STAGE3_W_FREE
        else:
            eff_w_w, eff_w_f = 0.0, 1.0
        norm = eff_w_w + eff_w_f
        eff_w_w_n = eff_w_w / norm
        eff_w_f_n = eff_w_f / norm
        if eff_w_f_n == 0.0:
            print(f"  objective: pure windowed (re-init at {WINDOW_REINIT_YEARS})")
        elif eff_w_w_n == 0.0:
            print("  objective: pure fully-free 1988→2014 rollout")
        else:
            print(f"  objective: mixed loss "
                  f"({eff_w_w_n:.2f} windowed + {eff_w_f_n:.2f} free 26yr); "
                  f"window re-init at {WINDOW_REINIT_YEARS}")

    rng = np.random.default_rng(SEED)
    init_pop, pack_point = _build_seeded_init_population(rng)
    print(f"  DE init: {init_pop.shape[0]} candidates "
          f"(1 bootstrap pack + {init_pop.shape[0]-1} LHS)")
    print(f"  bootstrap seed: " + ", ".join(
        f"{n}={v:.4f}" for n, v in zip(GLOBAL_PARAM_NAMES, pack_point)))

    de_log = []
    t0 = time.time()

    def callback(xk, convergence):
        de_log.append({"x": xk.tolist(), "conv": float(convergence),
                       "time_s": time.time() - t0})
        print(f"  gen {len(de_log):3d} | conv={float(convergence):.4f} | "
              f"{time.time()-t0:.0f}s", flush=True)

    stage3_timeout = float(os.environ.get("STAGE3_TIMEOUT_S", TRAJECTORY_TIMEOUT_S))
    print(f"  candidate timeout: {stage3_timeout:.0f}s")
    timed_map = _TimedGlobalMap(
        max_workers=DE_WORKERS,
        timeout_s=stage3_timeout,
        penalty=PENALTY,
    )
    loss_func = functools.partial(
        shared_global_loss, data=data, country_vecs=country_vecs,
        years=STAGE3_YEARS, windowed=STAGE3_WINDOWED,
        w_window=STAGE3_W_WINDOW, w_free=STAGE3_W_FREE,
    )
    result = differential_evolution(
        loss_func, bounds=GLOBAL_PARAM_BOUNDS,
        init=init_pop,
        seed=SEED, maxiter=DE_MAXITER, tol=DE_TOL,
        mutation=(0.5, 1.0), recombination=0.7, polish=False,
        updating="deferred", workers=timed_map, disp=False, callback=callback,
    )

    pack = dict(zip(GLOBAL_PARAM_NAMES, result.x.tolist()))
    pack["_mode"] = STAGE3_MODE
    pack["_country_summary"] = RESTRICTED_SUMMARY
    pack["_years"] = [int(STAGE3_YEARS[0]), int(STAGE3_YEARS[-1])]
    pack["_loss"] = float(result.fun)
    pack["_nfev"] = int(result.nfev)
    pack["_message"] = str(result.message)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(GLOBAL_PARAMS_PATH, "w") as f:
        json.dump(pack, f, indent=2)
    with open(DE_LOG_PATH, "w") as f:
        json.dump(de_log, f, indent=2)
    print(f"Wrote {GLOBAL_PARAMS_PATH} (loss={result.fun:.5f})")


if __name__ == "__main__":
    main()
