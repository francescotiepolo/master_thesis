"""
Shared functions for the calibration files.

  load_data              — loads all extracted arrays into a dict
  build_model            — constructs ProductSpaceModel from empirical data + theta
  simulate               — runs model year-by-year, returns annual snapshots
  diversification_entropy — per-country Shannon entropy of alpha
  spearman               — Spearman rank correlation
  compute_stats          — summary statistics at one time point
  trajectory_loss        — full scalar loss over calibration years
  theta_to_dict          — parameter vector to named dict
  evaluate_batch         — parallel evaluation of multiple parameter vectors
  load_nroy_bounds       — load NROY bounds from HM results or fall back to prior
"""

import os
import sys
import warnings
import numpy as np
from scipy.stats import spearmanr as _spearmanr

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from product_space_model import ProductSpaceModel
from calibration_config import (
    EXTRACTED_DIR, YEAR_START, CALIB_YEARS, FIXED, LOSS_WEIGHTS,
    TIME_WEIGHT_SLOPE, SIM_STEPS_PER_YEAR, PARAM_NAMES, PARAM_BOUNDS,
    N_PARAMS, HM_DIR, BO_DIR
)

# Penalty value for failed simulations
PENALTY = 1e6


# Sampling

def lhc_sample(bounds, n, seed):
    """
    Latin Hypercube sample mapped to parameter bounds.
    """
    from scipy.stats.qmc import LatinHypercube
    sampler = LatinHypercube(d=len(bounds), seed=seed)
    unit = sampler.random(n=n)
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])
    return lo + unit * (hi - lo)


# Data loading

def load_data(extracted_dir=EXTRACTED_DIR, years=None):
    """
    Load all arrays from extracted_data/ into a dict.
    """
    if years is None:
        years = list(range(1988, 2025))

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

    print(f"Data loaded: SC={data['SC']}, SP={data['SP']}, "
          f"years={min(data['alpha_obs'])}–{max(data['alpha_obs'])}")
    return data


# Parameter helpers

def theta_to_dict(theta):
    """
    Convert parameter vector to dict.
    """
    return dict(zip(PARAM_NAMES, theta))


def in_bounds(theta, bounds=None):
    """
    Check if theta is within parameter bounds.
    """
    if bounds is None:
        bounds = PARAM_BOUNDS
    for val, (lo, hi) in zip(theta, bounds):
        if val < lo or val > hi:
            return False
    return True


def satisfies_constraints(theta):
    """
    Check that all free parameters are within their bounds and c_prime <= c.
    """
    params = theta_to_dict(theta)
    for name, (lo, hi) in zip(PARAM_NAMES, PARAM_BOUNDS):
        if not (lo <= params[name] <= hi):
            return False
    if "c_prime" in params and "c" in params:
        if params["c_prime"] > params["c"]:
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
        s = float(params["s"]),
        c = float(params["c"]),
        c_prime = float(params["c_prime"]),
        gamma = float(params["gamma"]),
        kappa = float(fp["kappa"]),
        sigma = float(params["sigma"]),
        nu = float(params["nu"]),
        G = float(fp["G"]),
        q = float(fp["q"]),
        mu = float(fp["mu"]),
        beta_trade_off = float(params["beta_trade_off"]),
        enable_entry = bool(fp.get("enable_entry", False)),
        entry_threshold = float(params["entry_threshold"]),
    )
    _patch_model(model, data, h_mean=float(params["h_mean"]),
                 C_diag_mean=float(params["C_diag_mean"]),
                 C_offdiag_mean=float(params["C_offdiag_mean"]))
    return model


def _patch_model(model, data, h_mean=None, C_diag_mean=None, C_offdiag_mean=None):
    """
    Overwrite randomly generated parameters with empirical/fixed values.
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
    model.C_CC = np.full((model.SC, model.SC), C_offdiag_mean)
    np.fill_diagonal(model.C_CC, C_diag_mean)

    model.phi_space = data["phi_space"].astype(float)
    model._build_product_space_matrices()


# Simulation

def simulate(theta, data, years):
    """
    Simulate from YEAR_START forward, returning (alpha, C, P) at each year.
    Returns None if failure.
    """
    model = build_model(theta, data)
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
                    n_steps = SIM_STEPS_PER_YEAR * n_yr,
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


# Summary statistics

def diversification_entropy(alpha):
    """
    Per-country Shannon entropy of the alpha distribution.
    entropy_j = -sum_i alpha_ij * log(alpha_ij + eps)
    Returns a (SC,) vector — higher values mean less specialised.
    """
    eps = 1e-12
    return -np.sum(alpha * np.log(alpha + eps), axis=1)


def spearman(a, b):
    """
    Spearman rank correlation, returning 0.0 invalid inputs.
    Used because (1) it is scale-invariant, so the model is not penalised 
    for normalisation differences between simulated and observed values;
    (2) it is robust to outliers (e.g. USA, CHN), treating all rank
    differences equally regardless of magnitude.
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


def compute_stats(alpha_sim, C_sim, alpha_obs, C_obs):
    """
    Compute calibration targets for a single year.
    - rank_C: 1 - Spearman rank correlation of country exports C
    - alpha: 1 - Spearman rank correlation of per-country diversification
                 entropy, capturing whether the model reproduces the right
                 ordering of countries by how diversified their export basket is
    """
    rho_C = spearman(C_sim, C_obs)
    ent_sim = diversification_entropy(alpha_sim)
    ent_obs = diversification_entropy(alpha_obs)
    rho_alpha = spearman(ent_sim, ent_obs)

    return {
        "rank_C": 1.0 - rho_C,
        "alpha": 1.0 - rho_alpha,
        "rho_C": rho_C,
        "rho_alpha": rho_alpha,
    }


# Trajectory loss

def trajectory_loss(theta, data, years=None):
    """
    Weighted scalar loss over the calibration window.
    Returns PENALTY if failure or constraint violation.
    """
    if years is None:
        years = CALIB_YEARS

    if not satisfies_constraints(theta):
        return PENALTY

    traj = simulate(theta, data, years)
    if traj is None:
        return PENALTY

    w = LOSS_WEIGHTS
    total, total_w = 0.0, 0.0

    for yr in years:
        if yr not in traj or yr not in data["alpha_obs"]:
            continue
        alpha_sim, C_sim, _ = traj[yr]
        s = compute_stats(alpha_sim, C_sim, data["alpha_obs"][yr], data["C_obs"][yr])
        loss_t = w["rank_C"] * s["rank_C"] + w["alpha"] * s["alpha"]
        wt = 1.0 + TIME_WEIGHT_SLOPE * (yr - YEAR_START) # Linear time weighting: later years are more important
        total += wt * loss_t
        total_w += wt

    return float(total / total_w) if total_w > 0 else PENALTY


# GP helpers (shared across HM, BO, MCMC, validation)

def fit_botorch_gp(X_raw, y_raw, bounds):
    """
    Train a GP (fit a BoTorch SingleTaskGP) on raw data.
    Normalises inputs to [0,1]^d (so no parameter dominates just because of its scale).
    Returns (gp, bounds_tensor).
    """
    # Importing BoTorch and PyTorch here to avoid heavy dependency for non-GP users.
    import torch
    from botorch.models import SingleTaskGP
    from botorch.fit import fit_gpytorch_mll
    from botorch.utils.transforms import normalize
    from gpytorch.mlls import ExactMarginalLogLikelihood

    bt = torch.tensor(bounds, dtype=torch.float64).T  # (2, d) bounds tensor (containing lower and upper bounds for each parameter): row 0 = lower bound, row 1 = upper bound
    X = normalize(torch.tensor(X_raw, dtype=torch.float64), bt) # Rescale each parameter to [0,1]
    Y = torch.tensor(y_raw, dtype=torch.float64).unsqueeze(-1) # BoTorch expects shape (n, 1)

    gp = SingleTaskGP(X, Y) # GP with default homoscedastic noise
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp) # Marginal log-likelihood: objective for hyperparameter fitting
    fit_gpytorch_mll(mll) # Optimise GP hyperparameters
    gp.eval() # Switch to prediction mode
    return gp, bt


def gp_predict(gp, theta_raw, bt):
    """
    Predict mean and std from GP on raw inputs.
    """
    import torch
    from botorch.utils.transforms import normalize

    X = normalize(torch.tensor(theta_raw, dtype=torch.float64), bt) # Rescale inputs to match training
    with torch.no_grad(): # No gradients needed for prediction
        posterior = gp.posterior(X)
        mu = posterior.mean.squeeze(-1).numpy() # Predicted mean loss
        sigma = posterior.variance.sqrt().squeeze(-1).numpy() # Predicted std (uncertainty)
    return mu, sigma


def load_nroy_bounds():
    """
    Load NROY bounds from history matching results, or return prior bounds if not found.
    """
    import json
    hm_path = os.path.join(HM_DIR, "nroy_bounds.json")
    if os.path.exists(hm_path):
        with open(hm_path) as f:
            bounds = json.load(f)["bounds"]
        print(f"Loaded NROY bounds from {hm_path}")
    else:
        bounds = list(PARAM_BOUNDS)
        print("No HM results found -- using prior bounds.")
    return bounds


def evaluate_batch(candidates, data, n_jobs=-1):
    """
    Evaluate a batch of parameter vectors in parallel.
    """
    from joblib import Parallel, delayed
    return np.array(
        Parallel(n_jobs=n_jobs, verbose=0)(
            delayed(trajectory_loss)(candidates[i], data)
            for i in range(len(candidates))
        )
    )


def load_bo_gp():
    """
    Load the BoTorch GP saved.
    Returns (gp, lo, hi, bounds_tensor) or raises FileNotFoundError.
    """
    import torch
    from botorch.models import SingleTaskGP

    # Use the tighter NROY bounds from history matching if available, otherwise use full prior bounds from calibration_config.
    # NROY bounds represent the best guess of where good parameters lie according to HM,
    # so using them can improve GP predictions and BO efficiency.
    if os.path.exists(nroy_path):
        b = np.load(nroy_path)
        lo = b[:, 0]
        hi = b[:, 1]
    else:
        lo = np.array([b[0] for b in PARAM_BOUNDS])
        hi = np.array([b[1] for b in PARAM_BOUNDS])

    bt = torch.tensor(np.stack([lo, hi]), dtype=torch.float64) # (2, d) bounds tensor

    gp_state_path = os.path.join(BO_DIR, "gp_state.pth") # Saved GP hyperparameters
    train_X_path = os.path.join(BO_DIR, "gp_train_X.npy") # Normalised training inputs
    train_Y_path = os.path.join(BO_DIR, "gp_train_Y.npy") # Training targets (negative loss)

    if not os.path.exists(gp_state_path) or not os.path.exists(train_X_path):
        raise FileNotFoundError(
            f"GP state not found at {gp_state_path}. Run the BO phase first."
        )

    # Reconstruct the GP and load saved weights
    train_X = torch.tensor(np.load(train_X_path), dtype=torch.float64)
    train_Y = torch.tensor(np.load(train_Y_path), dtype=torch.float64)
    gp = SingleTaskGP(train_X, train_Y)                              # build skeleton
    gp.load_state_dict(torch.load(gp_state_path, weights_only=True)) # fill in hyperparameters
    gp.eval()
    print(f"Loaded BoTorch GP from {gp_state_path} ({len(train_X)} training points)")

    return gp, lo, hi, bt