"""
Infer r_i and pi_ij (country competition) from trade data via Ridge CV regression.

Directly regressing g_i on C_j mixes competition with mutualism-mediated effects,
so a data proxy mut_proxy_i(t) = sum_k alpha_obs * beta_C_ref * (P_k / E_k) is
constructed and either included as a free regressor or subtracted from the target.
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from calibration_config import (
    EXTRACTED_DIR, YEAR_START, YEAR_END,
)

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "calibration_results", "growth_regression")

RIDGE_ALPHAS = np.logspace(-4, 6, 50)
N_BOOTSTRAP = 200


# Data — normalized export values matching the calibration pipeline

def load_panel():
    """
    Build annual C_j(t) and P_k(t) in the same normalized units used by the
    calibration pipeline (annual/C_{year}.npy and annual/P_{year}.npy).
    Each year is divided by that year's mean of positive values.
    """
    countries = pd.read_csv(os.path.join(EXTRACTED_DIR, "countries_index.csv"))
    products = pd.read_csv(os.path.join(EXTRACTED_DIR, "products_index.csv"))
    country_list = countries["location_code"].tolist()
    product_list = products["hs_product_code"].astype(str).tolist()

    ann_dir = os.path.join(EXTRACTED_DIR, "annual")
    print(f"Loading normalized annual arrays from {ann_dir}...")

    C, P = {}, {}
    for yr in range(YEAR_START, YEAR_END + 1):
        c_path = os.path.join(ann_dir, f"C_{yr}.npy")
        p_path = os.path.join(ann_dir, f"P_{yr}.npy")
        if not (os.path.exists(c_path) and os.path.exists(p_path)):
            continue
        C[yr] = np.load(c_path).astype(float)
        P[yr] = np.load(p_path).astype(float)

    if not C:
        raise FileNotFoundError(
            f"No annual C/P arrays found in {ann_dir}. "
            "Run data_extraction_all_years.py first."
        )
    print(f"  loaded {len(C)} years ({min(C.keys())}–{max(C.keys())})")

    return C, P, country_list, product_list


def load_alpha_panel(years):
    """
    Load annual alpha_obs arrays as a dict {year: (SC, SP) row-normalized}.
    """
    ann_dir = os.path.join(EXTRACTED_DIR, "annual")
    alpha = {}
    for yr in years:
        path = os.path.join(ann_dir, f"alpha_{yr}.npy")
        if os.path.exists(path):
            alpha[yr] = np.load(path).astype(float)
    return alpha


def compute_mutualism_proxy(C, P, alpha_panel, beta_C_ref, eps=1e-12):
    """
    Data mutualism proxy for every country-year.

    For each year t:
      E_k(t)      = sum_j alpha_obs[j,k,t] * beta_C_ref[j,k] * C_obs[j,t]
      xi_hat_k(t) = P_obs[k,t] / max(E_k(t), eps)
      mut_j(t)    = sum_k alpha_obs[j,k,t] * beta_C_ref[j,k] * xi_hat_k(t)
    """
    mut = {}
    for yr, alpha in alpha_panel.items():
        if yr not in C or yr not in P:
            continue
        C_yr = C[yr]
        P_yr = P[yr]
        # E_k = sum over countries of alpha[j,k] * beta[j,k] * C[j]
        E = ((alpha * beta_C_ref).T @ C_yr) # shape (SP,)
        xi_hat = P_yr / np.maximum(E, eps) # shape (SP,)
        # mut_j = sum over products of alpha[j,k] * beta[j,k] * xi_hat[k]
        mut_j = (alpha * beta_C_ref) @ xi_hat # shape (SC,)
        mut[yr] = mut_j
    return mut


def build_country_data(C, P, i, mut_panel=None, dt=1):
    """
    Build regression arrays for country i.

    Returns y (growth rates), X_C (country levels at year t), X_P
    (product levels at year t), mut (country i's mutualism proxy at year
    t or None), years (time labels). Skips years where C_i is zero.
    """
    sorted_years = sorted(C.keys())
    y, xc, xp, mut_vals, ts = [], [], [], [], []
    for yr in sorted_years:
        if yr + dt not in C:
            continue
        ci = C[yr][i]
        ci_next = C[yr + dt][i]
        if ci <= 0 or ci_next <= 0:
            continue
        g = (ci_next - ci) / (dt * ci)
        if not np.isfinite(g):
            continue
        if mut_panel is not None:
            if yr not in mut_panel:
                continue
            mut_vals.append(mut_panel[yr][i])
        y.append(g)
        xc.append(C[yr])
        xp.append(P[yr])
        ts.append(yr)
    mut_arr = np.array(mut_vals) if mut_panel is not None else None
    return np.array(y), np.array(xc), np.array(xp), mut_arr, np.array(ts)


# Estimation

def _ridge_fit(X, y, alphas, n_folds):
    """
    Standardize, fit RidgeCV, return coefficients and intercept.
    """
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    folds = min(n_folds, len(y) - 1)
    model = RidgeCV(alphas=alphas, cv=folds).fit(Xs, y)
    coef = model.coef_ / scaler.scale_
    intercept = model.intercept_
    return coef, intercept, model, scaler


def _bootstrap_se(X, y, base_pi, base_r, n_folds, n_boot, pi_slice=None):
    """
    Bootstrap the final-stage ridge fit on (X, y) and return SEs.
    """
    T = len(y)
    n_features = X.shape[1]
    pi_boot = np.zeros((n_boot, n_features))
    r_boot = np.zeros(n_boot)
    folds = min(n_folds, T - 1)
    rng = np.random.default_rng(42)
    for b in range(n_boot):
        idx = rng.choice(T, T, replace=True)
        scaler_b = StandardScaler()
        try:
            Xb = scaler_b.fit_transform(X[idx])
            rb = RidgeCV(alphas=RIDGE_ALPHAS, cv=folds).fit(Xb, y[idx])
            pi_boot[b] = rb.coef_ / scaler_b.scale_
            r_boot[b] = rb.intercept_
        except Exception:
            pi_boot[b] = base_pi if pi_slice is None else np.concatenate([[base_r], base_pi])[pi_slice]
            r_boot[b] = base_r
    se_all = np.std(pi_boot, axis=0)
    return se_all, float(np.std(r_boot))


def estimate_country(y, X_C, mut_i=None, variant="raw", n_folds=5):
    """
    Ridge estimate for country i.

    Parameters:

    - y : (T,)
        Growth rates g_i(t).
    - X_C : (T, SC)
        Country level regressors C_j(t).
    - mut_i : (T,) or None
        Mutualism proxy for country i, required unless variant="raw".
    - variant : {"raw", "free", "fixed"}
        raw   — fit g_i ~ r_i + pi · C (no mutualism control)
        free  — fit g_i ~ r_i + eta_i · mut_i + pi · C
        fixed — fit g_i - mut_i ~ r_i + pi · C (eta_i = 1)
    """
    T, SC = X_C.shape
    ss_tot = np.sum((y - y.mean()) ** 2)

    if variant == "raw":
        X = X_C
        y_fit = y
    elif variant == "free":
        if mut_i is None:
            raise ValueError("variant='free' requires mut_i")
        X = np.column_stack([mut_i.reshape(-1, 1), X_C])
        y_fit = y
    elif variant == "fixed":
        if mut_i is None:
            raise ValueError("variant='fixed' requires mut_i")
        X = X_C
        y_fit = y - mut_i
    else:
        raise ValueError(f"Unknown variant: {variant}")

    coef, intercept, model, scaler = _ridge_fit(X, y_fit, RIDGE_ALPHAS, n_folds)
    y_hat = model.predict(scaler.transform(X))
    ss_res = np.sum((y_fit - y_hat) ** 2)
    ss_tot_fit = np.sum((y_fit - y_fit.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot_fit if ss_tot_fit > 1e-20 else 0.0

    if variant == "fixed":
        y_hat_g = y_hat + mut_i
    else:
        y_hat_g = y_hat
    ss_res_g = np.sum((y - y_hat_g) ** 2)
    r2_g = 1.0 - ss_res_g / ss_tot if ss_tot > 1e-20 else 0.0

    # Separate eta_i from pi if variant="free"
    if variant == "free":
        eta_i = float(coef[0])
        pi = coef[1:]
    else:
        eta_i = None
        pi = coef

    # Bootstrap SEs
    se_all, se_r_boot = _bootstrap_se(X, y_fit, coef, intercept, n_folds, N_BOOTSTRAP)
    if variant == "free":
        se_eta = float(se_all[0])
        se_pi = se_all[1:]
    else:
        se_eta = None
        se_pi = se_all

    return {
        "r_i": float(intercept),
        "pi": pi,
        "eta_i": eta_i,
        "se_r": se_r_boot,
        "se_pi": se_pi,
        "se_eta": se_eta,
        "r2": float(r2),
        "r2_g": float(r2_g),
        "alpha_cv": float(model.alpha_),
        "n_obs": T,
    }



# Outputs

def save_variant(codes, names, all_res, suffix):
    """
    Save coefficients and diagnostics for one regression variant.
    """
    os.makedirs(OUT_DIR, exist_ok=True)

    r_vals = np.array([all_res[c]["r_i"] for c in codes])
    se_r = np.array([all_res[c]["se_r"] for c in codes])
    r2 = np.array([all_res[c]["r2"] for c in codes])
    r2_g = np.array([all_res[c]["r2_g"] for c in codes])

    pi_mat = np.row_stack([all_res[c]["pi"] for c in codes])
    se_pi_mat = np.row_stack([all_res[c]["se_pi"] for c in codes])

    eta_vals = np.array([
        all_res[c]["eta_i"] if all_res[c]["eta_i"] is not None else np.nan
        for c in codes
    ])
    se_eta_vals = np.array([
        all_res[c]["se_eta"] if all_res[c]["se_eta"] is not None else np.nan
        for c in codes
    ])

    # r + eta values
    df_r = pd.DataFrame({
        "country": codes, "name": names,
        "r_i": r_vals, "se_r": se_r,
        "eta_i": eta_vals, "se_eta": se_eta_vals,
        "r2": r2, "r2_on_g": r2_g,
    })
    df_r.to_csv(os.path.join(OUT_DIR, f"r_values{suffix}.csv"), index=False)

    # pi matrix
    pd.DataFrame(pi_mat, index=codes, columns=codes).to_csv(
        os.path.join(OUT_DIR, f"pi_matrix{suffix}.csv"))
    np.save(os.path.join(OUT_DIR, f"pi_matrix{suffix}.npy"), pi_mat)
    pd.DataFrame(se_pi_mat, index=codes, columns=codes).to_csv(
        os.path.join(OUT_DIR, f"pi_se_matrix{suffix}.csv"))

    return r_vals, se_r, r2, r2_g, pi_mat, eta_vals


def plot_summary(codes, r_vals, se_r, r2, pi_mat, suffix=""):
    """
    Three-panel summary figure for one variant.
    """
    SC = len(codes)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Pi heatmap
    ax = axes[0]
    vmax = np.percentile(np.abs(pi_mat), 95)
    if vmax < 1e-10:
        vmax = 1.0
    im = ax.imshow(pi_mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="equal")
    ax.set_xticks(range(SC))
    ax.set_xticklabels(codes, rotation=90, fontsize=7)
    ax.set_yticks(range(SC))
    ax.set_yticklabels(codes, fontsize=7)
    ax.set_title(r"$\pi_{ij}$" + f" ({suffix.strip('_') or 'raw'})")
    ax.set_xlabel("country j (regressor)")
    ax.set_ylabel("country i (dependent)")
    plt.colorbar(im, ax=ax, shrink=0.8)

    # 2. Intrinsic growth rates
    ax = axes[1]
    colors = ["tab:red" if v < 0 else "tab:blue" for v in r_vals]
    ax.barh(range(SC), r_vals, color=colors, xerr=se_r, capsize=2)
    ax.set_yticks(range(SC))
    ax.set_yticklabels(codes, fontsize=7)
    ax.axvline(0, color="k", linewidth=0.5)
    ax.set_xlabel(r"$r_i$")
    ax.set_title(f"Intercepts ({suffix.strip('_') or 'raw'})")
    ax.invert_yaxis()

    # 3. R-squared
    ax = axes[2]
    ax.bar(np.arange(SC), r2, color="steelblue")
    ax.set_xticks(np.arange(SC))
    ax.set_xticklabels(codes, rotation=90, fontsize=7)
    ax.set_ylabel(r"$R^2$")
    ax.set_ylim(0, max(1.0, r2.max() * 1.05))
    ax.set_title(f"Model fit ({suffix.strip('_') or 'raw'})")

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"regression_summary{suffix}.png"), dpi=150)
    plt.close()



# Main

def _variant_diagnostics(codes, all_res, label):
    """
    Print a per-variant summary of r_i, pi_ii, and sign structure.
    """
    SC = len(codes)
    r_vals = np.array([all_res[c]["r_i"] for c in codes])
    pi_mat = np.row_stack([all_res[c]["pi"] for c in codes])
    r2 = np.array([all_res[c]["r2_g"] for c in codes])
    diag = np.diag(pi_mat)
    off = pi_mat[~np.eye(SC, dtype=bool)]
    print(f"\n--- variant: {label} ---")
    print(f"  r_i       mean={r_vals.mean():+.3f}  range=[{r_vals.min():+.3f}, {r_vals.max():+.3f}]")
    print(f"  pi_ii     mean={diag.mean():+.3f}  range=[{diag.min():+.3f}, {diag.max():+.3f}]  "
          f"all_neg={bool(np.all(diag < 0))}")
    print(f"  pi_off    mean={off.mean():+.3f}  range=[{off.min():+.3f}, {off.max():+.3f}]  "
          f"%neg={np.mean(off < 0) * 100:.0f}")
    print(f"  R² on g   mean={r2.mean():.3f}")
    if all_res[codes[0]]["eta_i"] is not None:
        eta = np.array([all_res[c]["eta_i"] for c in codes])
        print(f"  eta_i     mean={eta.mean():+.3f}  range=[{eta.min():+.3f}, {eta.max():+.3f}]")


def main():
    countries_df = pd.read_csv(os.path.join(EXTRACTED_DIR, "countries_index.csv"))

    C, P, codes, product_codes = load_panel()
    names = countries_df["country_name_short"].tolist()
    SC = len(codes)
    SP = len(product_codes)
    print(f"Panel: {len(C)} years, SC={SC}, SP={SP}")
    sample_yr = sorted(C.keys())[0]
    print(f"  C range ({sample_yr}): {C[sample_yr].min():.4f} - {C[sample_yr].max():.4f} (normalized)")

    # Load observed alpha panel and build pure-data mutualism proxy
    alpha_panel = load_alpha_panel(sorted(C.keys()))
    beta_C_ref = np.load(os.path.join(EXTRACTED_DIR, "beta_C_ref.npy")).astype(float)
    mut_panel = compute_mutualism_proxy(C, P, alpha_panel, beta_C_ref)

    if mut_panel:
        all_mut = np.concatenate([v for v in mut_panel.values()])
        print(
            f"  mut_proxy: {len(mut_panel)} years loaded, "
            f"mean={all_mut.mean():.3f}, std={all_mut.std():.3f}, "
            f"range=[{all_mut.min():.3f}, {all_mut.max():.3f}]"
        )
    np.save(os.path.join(OUT_DIR, "mut_proxy_panel.npy"),
            np.row_stack([mut_panel[yr] for yr in sorted(mut_panel)]))
    print()

    variants = ["raw", "free", "fixed"]
    results_by_variant = {v: {} for v in variants}

    for i in range(SC):
        y, xc, xp, mut_i, ts = build_country_data(C, P, i, mut_panel=mut_panel)
        if len(y) < 10:
            print(f"  {codes[i]}: only {len(y)} obs, skipping")
            continue

        for variant in variants:
            res = estimate_country(y, xc, mut_i=mut_i, variant=variant)
            results_by_variant[variant][codes[i]] = res

        rres = results_by_variant["raw"][codes[i]]
        fres = results_by_variant["free"][codes[i]]
        xres = results_by_variant["fixed"][codes[i]]
        print(
            f"  {codes[i]:3s} "
            f"raw: r={rres['r_i']:+.3f} pi_ii={rres['pi'][i]:+.3f} R²={rres['r2_g']:.2f} | "
            f"free: r={fres['r_i']:+.3f} eta={fres['eta_i']:+.3f} pi_ii={fres['pi'][i]:+.3f} R²={fres['r2_g']:.2f} | "
            f"fixed: r={xres['r_i']:+.3f} pi_ii={xres['pi'][i]:+.3f} R²={xres['r2_g']:.2f}"
        )

    if not results_by_variant["raw"]:
        print("No countries with sufficient data.")
        return

    suffixes = {"raw": "_raw", "free": "_free", "fixed": "_fixed"}
    for variant in variants:
        label = suffixes[variant]
        r_vals, se_r, r2, r2_g, pi_mat, eta_vals = save_variant(
            codes, names, results_by_variant[variant], label)
        plot_summary(codes, r_vals, se_r, r2_g, pi_mat, suffix=label)
        _variant_diagnostics(codes, results_by_variant[variant], label.strip("_"))

    print(f"\nResults saved to {OUT_DIR}/ (suffixes: _raw, _free, _fixed)")


if __name__ == "__main__":
    main()