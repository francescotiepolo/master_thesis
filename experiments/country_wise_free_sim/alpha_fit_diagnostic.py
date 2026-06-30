"""
Per-country, per-product alpha-fit diagnostic.

For each (country, product), compute Spearman trajectory correlation between
simulated and observed alpha shares, separately over the calibration window
(1988-2014) and the validation window (2015-2024). Aggregate per country
weighted by observed alpha mass.

Outputs (under --out-dir):
  - alpha_fit_per_product.csv      one row per (country, product)
  - alpha_fit_per_country.csv      one row per country (mass-weighted summary)
  - alpha_fit_histograms.png       19 panels: per-product corr distribution
  - alpha_fit_ranking.png          bar chart, countries ranked by mass-weighted corr
  - alpha_fit_heatmap.png          19 x top-K product heatmaps (calib | valid)
  - alpha_fit_mass_vs_corr.png     scatter: obs alpha mass vs trajectory corr

Trajectories are reused from the existing _cache/ directory (same keying as
run_free_sim/run_grid_figures); pass --fresh to force a recompute.
"""

import argparse
import csv
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)

from run_free_sim import (  # noqa: E402
    DEFAULT_COUNTRY_DIR,
    DEFAULT_R_VALUES_PATH,
    PER_COUNTRY_TIMEOUT_S,
    _conditioned_worker,
    load_country_index,
    load_growth_regression_r,
    load_products_index,
    set_simulation_sources,
)
from run_grid_figures import run_per_country  # noqa: E402
from calibration_utils import load_data  # noqa: E402

CALIB_END = 2014  # COUNTRY_CALIB_END from country_wise_calibration
EPS_STD = 1e-12   # minimum std of observed alpha to treat a product as "active"


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def _spearman(a, b):
    if len(a) < 3:
        return np.nan
    if np.std(a) < EPS_STD or np.std(b) < EPS_STD:
        return np.nan
    r, _ = spearmanr(a, b)
    return float(r) if np.isfinite(r) else np.nan


def _stack_alpha(traj, country_idx, years):
    """Return (sim_mat, n_years) where sim_mat has shape (n_years, n_products)."""
    return np.stack([traj[y]["alpha"][country_idx] for y in years])


def compute_country_metrics(traj, data, country_idx):
    """
    Build per-product trajectory metrics for one country. Returns a dict:

      {
        "calib_years": [...], "valid_years": [...],
        "corr_calib": (n_products,),   per-product Spearman over calib years
        "corr_valid": (n_products,),
        "mass_obs":   (n_products,),   mean observed alpha across all years
        "active":     (n_products,) bool, has non-degenerate observed trajectory
      }
    """
    sim_years = sorted(traj.keys())
    obs_years = sorted(data["alpha_obs"].keys())
    years = [y for y in sim_years if y in obs_years]

    calib_years = [y for y in years if y <= CALIB_END]
    valid_years = [y for y in years if y > CALIB_END]

    sim_mat = _stack_alpha(traj, country_idx, years)
    obs_mat = np.stack([data["alpha_obs"][y][country_idx] for y in years])
    n_products = sim_mat.shape[1]

    def _per_period(period_years):
        if len(period_years) < 3:
            return np.full(n_products, np.nan)
        idx = [years.index(y) for y in period_years]
        s = sim_mat[idx]
        o = obs_mat[idx]
        out = np.full(n_products, np.nan)
        for i in range(n_products):
            out[i] = _spearman(s[:, i], o[:, i])
        return out

    return {
        "calib_years": calib_years,
        "valid_years": valid_years,
        "corr_calib": _per_period(calib_years),
        "corr_valid": _per_period(valid_years),
        "mass_obs":   obs_mat.mean(axis=0),
        "active":     obs_mat.std(axis=0) > EPS_STD,
    }


def country_summary(metrics):
    """Mass-weighted + unweighted summaries over active products."""
    mass = metrics["mass_obs"]
    active = metrics["active"]

    def _summary(corr):
        valid = active & np.isfinite(corr)
        if not np.any(valid):
            return dict(
                mass_weighted=np.nan, mean=np.nan, median=np.nan,
                p25=np.nan, p75=np.nan, frac_pos=np.nan,
                frac_above_0_5=np.nan, n_active=int(active.sum()),
            )
        w = mass[valid]
        c = corr[valid]
        w_sum = w.sum()
        return dict(
            mass_weighted=float((c * w).sum() / w_sum) if w_sum > 0 else float(c.mean()),
            mean=float(c.mean()),
            median=float(np.median(c)),
            p25=float(np.percentile(c, 25)),
            p75=float(np.percentile(c, 75)),
            frac_pos=float((c > 0).mean()),
            frac_above_0_5=float((c > 0.5).mean()),
            n_active=int(active.sum()),
        )

    return {"calib": _summary(metrics["corr_calib"]),
            "valid": _summary(metrics["corr_valid"])}


# ---------------------------------------------------------------------------
# CSV writers
# ---------------------------------------------------------------------------

def write_per_product_csv(path, rows_meta, results, products):
    def _prod_name(i):
        r = products.get(i)
        if r is None:
            return ""
        return r.get("product_name_short") or r.get("hs_product_code") or ""

    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow([
            "country_code", "country_name", "product_idx", "product_name",
            "mean_obs_mass", "active",
            "corr_calib", "corr_valid",
        ])
        for row in rows_meta:
            code = row["location_code"]
            if code not in results:
                continue
            m = results[code]
            for i in range(m["mass_obs"].size):
                w.writerow([
                    code, row["country_name_short"], i, _prod_name(i),
                    f"{m['mass_obs'][i]:.6e}",
                    int(bool(m["active"][i])),
                    "" if not np.isfinite(m["corr_calib"][i]) else f"{m['corr_calib'][i]:.4f}",
                    "" if not np.isfinite(m["corr_valid"][i]) else f"{m['corr_valid'][i]:.4f}",
                ])
    print(f"Saved {path}")


def write_per_country_csv(path, rows_meta, summaries):
    fields = [
        "country_code", "country_name",
        "calib_mass_weighted", "calib_mean", "calib_median",
        "calib_p25", "calib_p75", "calib_frac_pos", "calib_frac_above_0_5",
        "valid_mass_weighted", "valid_mean", "valid_median",
        "valid_p25", "valid_p75", "valid_frac_pos", "valid_frac_above_0_5",
        "n_active",
    ]
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(fields)
        for row in rows_meta:
            code = row["location_code"]
            if code not in summaries:
                continue
            s = summaries[code]
            sc, sv = s["calib"], s["valid"]
            w.writerow([
                code, row["country_name_short"],
                *[f"{sc[k]:.4f}" if np.isfinite(sc[k]) else "" for k in
                  ("mass_weighted", "mean", "median", "p25", "p75",
                   "frac_pos", "frac_above_0_5")],
                *[f"{sv[k]:.4f}" if np.isfinite(sv[k]) else "" for k in
                  ("mass_weighted", "mean", "median", "p25", "p75",
                   "frac_pos", "frac_above_0_5")],
                sc["n_active"],
            ])
    print(f"Saved {path}")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_histograms(rows_meta, results, summaries, out_path):
    fig, axes = plt.subplots(4, 5, figsize=(22, 14), sharex=True, sharey=True)
    bins = np.linspace(-1, 1, 21)
    for idx, row in enumerate(rows_meta):
        ax = axes.flat[idx]
        code = row["location_code"]
        m = results.get(code)
        if m is None:
            ax.set_visible(False)
            continue
        active = m["active"]
        cc = m["corr_calib"][active & np.isfinite(m["corr_calib"])]
        cv = m["corr_valid"][active & np.isfinite(m["corr_valid"])]
        ax.hist(cc, bins=bins, alpha=0.55, color="#4C72B0", label="calib")
        ax.hist(cv, bins=bins, alpha=0.55, color="#DD8452", label="valid")
        sc = summaries[code]["calib"]["mass_weighted"]
        sv = summaries[code]["valid"]["mass_weighted"]
        if np.isfinite(sc):
            ax.axvline(sc, color="#4C72B0", ls="--", lw=1.2)
        if np.isfinite(sv):
            ax.axvline(sv, color="#DD8452", ls="--", lw=1.2)
        ax.axvline(0, color="grey", lw=0.6, ls=":")
        ax.set_title(
            f"{row['country_name_short']} ({code})  "
            f"calib={sc:.2f} valid={sv:.2f}",
            fontsize=9, fontweight="bold",
        )
        ax.tick_params(labelsize=7)
        ax.spines[["top", "right"]].set_visible(False)
        if idx == 0:
            ax.legend(fontsize=7, frameon=False)
    for idx in range(len(rows_meta), axes.size):
        axes.flat[idx].set_visible(False)
    fig.suptitle(
        "Per-product alpha trajectory correlation (Spearman) — distribution per country\n"
        "Dashed lines: mass-weighted mean over active products",
        fontsize=13, fontweight="bold",
    )
    fig.supxlabel("Spearman correlation (sim vs obs over time)", fontsize=11)
    fig.supylabel("Number of products", fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


def plot_ranking(rows_meta, summaries, out_path):
    pairs = []
    for row in rows_meta:
        code = row["location_code"]
        if code not in summaries:
            continue
        pairs.append((
            code, row["country_name_short"],
            summaries[code]["calib"]["mass_weighted"],
            summaries[code]["valid"]["mass_weighted"],
        ))
    pairs.sort(key=lambda t: (-t[2] if np.isfinite(t[2]) else -np.inf))
    codes = [p[0] for p in pairs]
    calib = np.array([p[2] for p in pairs])
    valid = np.array([p[3] for p in pairs])

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(codes))
    w = 0.4
    ax.bar(x - w / 2, calib, w, color="#4C72B0", label="calib (1988-2014)")
    ax.bar(x + w / 2, valid, w, color="#DD8452", label="valid (2015-2024)")
    ax.axhline(0, color="grey", lw=0.6, ls=":")
    ax.set_xticks(x)
    ax.set_xticklabels(codes, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Mass-weighted Spearman", fontsize=10)
    ax.set_title(
        "Country ranking by mass-weighted per-product alpha trajectory correlation",
        fontsize=11, fontweight="bold",
    )
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


def plot_heatmap(rows_meta, results, products, out_path, top_k=30):
    codes = [r["location_code"] for r in rows_meta if r["location_code"] in results]
    if not codes:
        return

    def _prod_label(i):
        r = products.get(i)
        if r is None:
            return str(i)
        return r.get("product_name_short") or r.get("hs_product_code") or str(i)

    # Pick top-K products by global mean observed mass across countries
    mass_stack = np.stack([results[c]["mass_obs"] for c in codes])
    global_mass = mass_stack.mean(axis=0)
    top_idx = np.argsort(global_mass)[::-1][:top_k]

    def _matrix(key):
        return np.stack([results[c][key][top_idx] for c in codes])

    mat_calib = _matrix("corr_calib")
    mat_valid = _matrix("corr_valid")

    fig, axes = plt.subplots(1, 2, figsize=(20, 8), sharey=True)
    for ax, mat, title in zip(axes, [mat_calib, mat_valid],
                              ["calibration (1988-2014)", "validation (2015-2024)"]):
        im = ax.imshow(mat, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(len(top_idx)))
        ax.set_xticklabels([_prod_label(i) for i in top_idx],
                           rotation=75, fontsize=7, ha="right")
        ax.set_yticks(range(len(codes)))
        ax.set_yticklabels(codes, fontsize=8)
        ax.set_title(title, fontsize=11, fontweight="bold")
        fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="Spearman")
    fig.suptitle(
        f"Per-(country, product) alpha trajectory correlation — top {top_k} products by mass",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


def plot_mass_vs_corr(rows_meta, results, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    cmap = plt.get_cmap("tab20")
    for col, (period, ax) in enumerate(zip(["calib", "valid"], axes)):
        for k, row in enumerate(rows_meta):
            code = row["location_code"]
            m = results.get(code)
            if m is None:
                continue
            active = m["active"]
            mass = m["mass_obs"][active]
            corr = m[f"corr_{period}"][active]
            ok = np.isfinite(corr) & (mass > 0)
            if not np.any(ok):
                continue
            ax.scatter(mass[ok], corr[ok], s=10, alpha=0.35,
                       color=cmap(k % 20), label=code)
        ax.set_xscale("log")
        ax.axhline(0, color="grey", lw=0.6, ls=":")
        ax.set_xlabel("Mean observed alpha mass (log)", fontsize=10)
        ax.set_title(f"{period}", fontsize=11, fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Spearman trajectory correlation", fontsize=10)
    axes[1].legend(fontsize=6, ncol=2, frameon=False,
                   loc="lower right", markerscale=2.0)
    fig.suptitle(
        "Does the model fit large or small products better?",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--params-dir", default=DEFAULT_COUNTRY_DIR)
    p.add_argument("--r-values", default=DEFAULT_R_VALUES_PATH)
    p.add_argument("--out-dir", required=True,
                   help="Where to save the CSVs and figures.")
    p.add_argument("--fresh", action="store_true",
                   help="Ignore trajectory cache; recompute every country.")
    p.add_argument("--timeout-s", type=float,
                   default=max(PER_COUNTRY_TIMEOUT_S, 300.0))
    p.add_argument("--top-k", type=int, default=30,
                   help="Number of products to show in the heatmap.")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    set_simulation_sources(args.params_dir, args.r_values)
    print(f"Params dir: {args.params_dir}", flush=True)
    print(f"R-values:   {args.r_values}", flush=True)

    data = load_data()
    rows = load_country_index()
    data["r_C_growth_regression"] = load_growth_regression_r(rows)
    code_to_idx = {r["location_code"]: r["position"] for r in rows}
    products = load_products_index()

    print("\n=== conditioned per-country trajectories ===", flush=True)
    trajs = run_per_country("conditioned", _conditioned_worker, rows, data,
                            code_to_idx, fresh=args.fresh,
                            timeout_s=args.timeout_s)

    print("\nComputing per-product metrics...", flush=True)
    results, summaries = {}, {}
    for row in rows:
        code = row["location_code"]
        if code not in trajs:
            continue
        m = compute_country_metrics(trajs[code], data, code_to_idx[code])
        results[code] = m
        summaries[code] = country_summary(m)
        sc = summaries[code]["calib"]
        sv = summaries[code]["valid"]
        print(
            f"  [{code}] active={sc['n_active']:3d} | "
            f"calib mw={sc['mass_weighted']:+.3f} median={sc['median']:+.3f} | "
            f"valid mw={sv['mass_weighted']:+.3f} median={sv['median']:+.3f}",
            flush=True,
        )

    write_per_product_csv(
        os.path.join(args.out_dir, "alpha_fit_per_product.csv"),
        rows, results, products,
    )
    write_per_country_csv(
        os.path.join(args.out_dir, "alpha_fit_per_country.csv"),
        rows, summaries,
    )
    plot_histograms(rows, results, summaries,
                    os.path.join(args.out_dir, "alpha_fit_histograms.png"))
    plot_ranking(rows, summaries,
                 os.path.join(args.out_dir, "alpha_fit_ranking.png"))
    plot_heatmap(rows, results, products,
                 os.path.join(args.out_dir, "alpha_fit_heatmap.png"),
                 top_k=args.top_k)
    plot_mass_vs_corr(rows, results,
                      os.path.join(args.out_dir, "alpha_fit_mass_vs_corr.png"))


if __name__ == "__main__":
    main()
