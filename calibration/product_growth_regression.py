"""
Infer r_k and product-product competition (a_k, b_k) from trade data via
Ridge CV regression, parameterized to match the model's competition form
(self + proximity-weighted), so the resulting C_PP[k,l] = max(-b_k,0) * phi[k,l]
with C_PP[k,k] = max(-a_k,0).

Per product k the regression is:
    g_k(t) = r_k + a_k * P_k(t) + b_k * (phi_off @ P)_k(t) + eta_k * mut_k(t)

with phi_off = phi_space - diag(diag(phi_space)) (here phi has zero diagonal
already). The mutualism proxy is aligned with the model's product mutualism
term: in ProductSpaceModel, rho_P_k = sum_j alpha[j,k] * beta_P[j,k] * C_j,
and beta_P[j,k] = cap[j,k] / (KP[k]^bt * alpha[j,k]) on the network mask, so
the alpha factors cancel and the model term equals

    rho_P_k = (1 / KP[k]^bt) * sum_j beta_C_ref[j,k] * C_j.

The proxy uses exactly that alpha-free form with a neutral bt_proxy=0.5; the
'free' variant fits eta_k so any residual scale mismatch is absorbed and only
'fixed' depends on the bt_proxy choice.

Three variants follow growth_regression.py: raw (no mut), free (mut as free
regressor), fixed (subtract mut, eta=1).
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from calibration_config import EXTRACTED_DIR, YEAR_START, YEAR_END, MAX_PARALLEL_JOBS
from growth_regression import (
    load_panel, load_alpha_panel, _ridge_fit, _bootstrap_se, RIDGE_ALPHAS,
    N_BOOTSTRAP,
)

N_JOBS = int(os.environ.get("REGRESSION_N_JOBS", str(MAX_PARALLEL_JOBS)))

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "calibration_results", "growth_regression")


def compute_product_mut_proxy(C, P, alpha_panel, beta_C_ref, bt_proxy=0.5):
    """
    Alpha-free data proxy for rho_P_k = (1/KP_k^bt) * sum_j beta_C_ref[j,k] * C_j.
    bt_proxy=0.5 is the midpoint of the calibrated range; the 'free' variant
    absorbs residual scale into eta_k. Returns dict {year: (SP,)}.
    """
    mask = (beta_C_ref > 0).astype(float)
    KP = mask.sum(axis=0)
    KP_factor = 1.0 / np.maximum(KP, 1.0) ** bt_proxy  # shape (SP,)

    mut = {}
    for yr in alpha_panel:
        if yr not in C:
            continue
        mut[yr] = KP_factor * (beta_C_ref.T @ C[yr])
    return mut


def build_product_data(P, k, phi_off, mut_panel=None, dt=1):
    """
    Build regression arrays for product k.

    Returns y (growth rates of P_k), X (T, 2) with columns [P_k, (phi_off P)_k],
    mut_vals (T,) or None, ts.
    """
    sorted_years = sorted(P.keys())
    y, x_self, x_phi, mut_vals, ts = [], [], [], [], []
    for yr in sorted_years:
        if yr + dt not in P:
            continue
        pk = P[yr][k]
        pk_next = P[yr + dt][k]
        if pk <= 0 or pk_next <= 0:
            continue
        g = (pk_next - pk) / (dt * pk)
        if not np.isfinite(g):
            continue
        if mut_panel is not None:
            if yr not in mut_panel:
                continue
            mut_vals.append(mut_panel[yr][k])
        phi_neighbors = float(phi_off[k] @ P[yr])
        y.append(g)
        x_self.append(pk)
        x_phi.append(phi_neighbors)
        ts.append(yr)
    if not y:
        return np.array([]), np.empty((0, 2)), None, np.array([])
    X = np.column_stack([np.array(x_self), np.array(x_phi)])
    mut_arr = np.array(mut_vals) if mut_panel is not None else None
    return np.array(y), X, mut_arr, np.array(ts)


def estimate_product(y, X, mut_k=None, variant="fixed", n_folds=5):
    """
    Ridge fit for a single product. Returns r_k, a_k, b_k, eta_k, R^2, SEs.
    """
    T = len(y)
    ss_tot = np.sum((y - y.mean()) ** 2)

    if variant == "raw":
        X_fit = X
        y_fit = y
    elif variant == "free":
        if mut_k is None:
            raise ValueError("variant='free' requires mut_k")
        X_fit = np.column_stack([mut_k.reshape(-1, 1), X])
        y_fit = y
    elif variant == "fixed":
        if mut_k is None:
            raise ValueError("variant='fixed' requires mut_k")
        X_fit = X
        y_fit = y - mut_k
    else:
        raise ValueError(variant)

    coef, intercept, model, scaler = _ridge_fit(X_fit, y_fit, RIDGE_ALPHAS, n_folds)
    y_hat = model.predict(scaler.transform(X_fit))
    ss_res = np.sum((y_fit - y_hat) ** 2)
    ss_tot_fit = np.sum((y_fit - y_fit.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot_fit if ss_tot_fit > 1e-20 else 0.0

    y_hat_g = y_hat + mut_k if variant == "fixed" else y_hat
    ss_res_g = np.sum((y - y_hat_g) ** 2)
    r2_g = 1.0 - ss_res_g / ss_tot if ss_tot > 1e-20 else 0.0

    if variant == "free":
        eta_k = float(coef[0])
        a_k = float(coef[1])
        b_k = float(coef[2])
    else:
        eta_k = None
        a_k = float(coef[0])
        b_k = float(coef[1])

    se_all, se_r = _bootstrap_se(X_fit, y_fit, coef, intercept, n_folds, N_BOOTSTRAP)
    if variant == "free":
        se_eta = float(se_all[0])
        se_a, se_b = float(se_all[1]), float(se_all[2])
    else:
        se_eta = None
        se_a, se_b = float(se_all[0]), float(se_all[1])

    return {
        "r_k": float(intercept),
        "a_k": a_k, "b_k": b_k,
        "eta_k": eta_k,
        "se_r": float(se_r), "se_a": se_a, "se_b": se_b, "se_eta": se_eta,
        "r2": float(r2), "r2_g": float(r2_g),
        "alpha_cv": float(model.alpha_),
        "n_obs": T,
    }


def assemble_C_PP(a_vec, b_vec, phi_space):
    """
    Build product-product competition matrix from regression coefficients.
    Negative a_k/b_k = competition; C_PP[k,k] = max(-a_k,0),
    C_PP[k,l] = max(-b_k,0)*phi[k,l] for k≠l.
    """
    SP = len(a_vec)
    a_neg = np.maximum(-a_vec, 0.0)
    b_neg = np.maximum(-b_vec, 0.0)
    phi_off = phi_space.copy()
    np.fill_diagonal(phi_off, 0.0)
    C_PP = (b_neg[:, np.newaxis]) * phi_off
    np.fill_diagonal(C_PP, a_neg)
    return C_PP


def save_variant(product_codes, all_res, suffix, phi_space):
    os.makedirs(OUT_DIR, exist_ok=True)

    SP = len(product_codes)
    r_vec = np.array([all_res[c]["r_k"] for c in product_codes])
    a_vec = np.array([all_res[c]["a_k"] for c in product_codes])
    b_vec = np.array([all_res[c]["b_k"] for c in product_codes])
    se_r = np.array([all_res[c]["se_r"] for c in product_codes])
    se_a = np.array([all_res[c]["se_a"] for c in product_codes])
    se_b = np.array([all_res[c]["se_b"] for c in product_codes])
    r2 = np.array([all_res[c]["r2"] for c in product_codes])
    r2_g = np.array([all_res[c]["r2_g"] for c in product_codes])
    eta = np.array([
        all_res[c]["eta_k"] if all_res[c]["eta_k"] is not None else np.nan
        for c in product_codes
    ])
    se_eta = np.array([
        all_res[c]["se_eta"] if all_res[c]["se_eta"] is not None else np.nan
        for c in product_codes
    ])

    df = pd.DataFrame({
        "product": product_codes, "r_k": r_vec, "se_r": se_r,
        "a_k": a_vec, "se_a": se_a, "b_k": b_vec, "se_b": se_b,
        "eta_k": eta, "se_eta": se_eta, "r2": r2, "r2_on_g": r2_g,
    })
    df.to_csv(os.path.join(OUT_DIR, f"r_values_P{suffix}.csv"), index=False)

    np.save(os.path.join(OUT_DIR, f"r_P_regression{suffix}.npy"), r_vec)
    np.save(os.path.join(OUT_DIR, f"a_vec_P{suffix}.npy"), a_vec)
    np.save(os.path.join(OUT_DIR, f"b_vec_P{suffix}.npy"), b_vec)

    C_PP = assemble_C_PP(a_vec, b_vec, phi_space)
    np.save(os.path.join(OUT_DIR, f"pi_competition_P{suffix}.npy"), C_PP)
    return r_vec, a_vec, b_vec, C_PP, r2_g


def plot_summary(product_codes, r_vec, a_vec, b_vec, C_PP, r2_g, suffix=""):
    SP = len(product_codes)
    fig, axes = plt.subplots(2, 2, figsize=(13, 11))

    ax = axes[0, 0]
    vmax = np.percentile(np.abs(C_PP), 99) if np.any(C_PP) else 1.0
    if vmax < 1e-10:
        vmax = 1.0
    im = ax.imshow(C_PP, cmap="Reds", vmin=0, vmax=vmax, aspect="equal")
    ax.set_title(r"$C_{PP}^{data}$" + f" ({suffix.strip('_') or 'raw'})")
    plt.colorbar(im, ax=ax, shrink=0.8)

    ax = axes[0, 1]
    ax.hist(r_vec, bins=40, color="steelblue")
    ax.axvline(0, color="k", lw=0.5)
    ax.set_xlabel(r"$r_k$")
    ax.set_title(f"intercepts ({suffix.strip('_') or 'raw'})")

    ax = axes[1, 0]
    ax.hist(a_vec, bins=40, color="indianred", alpha=0.7, label=r"$a_k$ (self)")
    ax.hist(b_vec, bins=40, color="seagreen", alpha=0.7, label=r"$b_k$ (proximity)")
    ax.axvline(0, color="k", lw=0.5)
    ax.legend()
    ax.set_title("competition coefficients")

    ax = axes[1, 1]
    ax.hist(r2_g, bins=30, color="steelblue")
    ax.set_xlabel(r"$R^2$ on $g$")
    ax.set_title("model fit")

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"product_regression_summary{suffix}.png"), dpi=150)
    plt.close()


def main():
    C, P, country_codes, product_codes = load_panel()
    SP = len(product_codes)
    SC = len(country_codes)
    print(f"Panel: {len(P)} years, SP={SP}, SC={SC}")

    phi_space = np.load(os.path.join(EXTRACTED_DIR, "phi_space.npy")).astype(float)
    assert phi_space.shape == (SP, SP), f"phi_space shape mismatch: {phi_space.shape}"
    phi_off = phi_space.copy()
    np.fill_diagonal(phi_off, 0.0)

    alpha_panel = load_alpha_panel(sorted(P.keys()))
    beta_C_ref = np.load(os.path.join(EXTRACTED_DIR, "beta_C_ref.npy")).astype(float)
    mut_panel = compute_product_mut_proxy(C, P, alpha_panel, beta_C_ref)

    if mut_panel:
        all_mut = np.concatenate([v for v in mut_panel.values()])
        print(f"  product mut_proxy: mean={all_mut.mean():.3f}, std={all_mut.std():.3f}")
    np.save(os.path.join(OUT_DIR, "mut_proxy_panel_P.npy"),
            np.row_stack([mut_panel[yr] for yr in sorted(mut_panel)]))

    variants = ["raw", "free", "fixed"]
    results_by_variant = {v: {} for v in variants}

    def _process_product(k, code):
        y, X, mut_k, _ = build_product_data(P, k, phi_off, mut_panel=mut_panel)
        if len(y) < 10:
            stub = {"r_k": 0.0, "a_k": 0.0, "b_k": 0.0, "eta_k": None,
                    "se_r": 0.0, "se_a": 0.0, "se_b": 0.0, "se_eta": None,
                    "r2": 0.0, "r2_g": 0.0, "alpha_cv": 0.0, "n_obs": len(y)}
            return code, len(y), {v: stub for v in variants}, True
        per_variant = {v: estimate_product(y, X, mut_k=mut_k, variant=v)
                       for v in variants}
        return code, len(y), per_variant, False

    print(f"Fitting {SP} products in parallel (n_jobs={N_JOBS}) ...", flush=True)
    parallel_out = Parallel(n_jobs=N_JOBS, backend="loky", verbose=5)(
        delayed(_process_product)(k, product_codes[k]) for k in range(SP)
    )

    n_skipped = 0
    for code, n_obs, per_variant, was_skipped in parallel_out:
        for variant, res in per_variant.items():
            results_by_variant[variant][code] = res
        if was_skipped:
            n_skipped += 1
    if n_skipped:
        print(f"  skipped {n_skipped}/{SP} products (insufficient data)")

    suffixes = {"raw": "_raw", "free": "_free", "fixed": "_fixed"}
    for variant in variants:
        suffix = suffixes[variant]
        r_vec, a_vec, b_vec, C_PP, r2_g = save_variant(
            product_codes, results_by_variant[variant], suffix, phi_space)
        plot_summary(product_codes, r_vec, a_vec, b_vec, C_PP, r2_g, suffix=suffix)
        diag = np.diag(C_PP)
        off = C_PP[~np.eye(SP, dtype=bool)]
        print(f"\n--- variant: {variant} ---")
        print(f"  r_k    mean={r_vec.mean():+.3f}  range=[{r_vec.min():+.3f}, {r_vec.max():+.3f}]")
        print(f"  a_k    mean={a_vec.mean():+.3f}  %neg={np.mean(a_vec < 0)*100:.0f}")
        print(f"  b_k    mean={b_vec.mean():+.3f}  %neg={np.mean(b_vec < 0)*100:.0f}")
        print(f"  C_PP   diag mean={diag.mean():.4f}  off mean={off.mean():.4f}")
        print(f"  R^2 on g  mean={r2_g.mean():.3f}")

    # Default pi_competition_P (no suffix) = fixed variant
    fixed_path = os.path.join(OUT_DIR, "pi_competition_P_fixed.npy")
    canonical = np.load(fixed_path)
    np.save(os.path.join(OUT_DIR, "pi_competition_P.npy"), canonical)
    np.save(os.path.join(OUT_DIR, "r_P_regression.npy"),
            np.load(os.path.join(OUT_DIR, "r_P_regression_fixed.npy")))
    print(f"\nCanonical outputs: pi_competition_P.npy, r_P_regression.npy "
          f"(from 'fixed' variant) saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
