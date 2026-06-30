"""Plotting for the alpha-injection experiment.

The primary outcome (DEFAULT_METRIC = "C_pct") is the percentage change in the
studied country's OWN exports versus the strength=0 baseline. Only that country
changes in this design, so its own-export change is the natural metric; world
market share ("share") and absolute C ("C") remain available in METRIC_CONFIG.
Sweep and heatmap plots show deltas against the baseline, preferably on 5-year
bucket means when available, with endpoint fallbacks for older result files.
"""
import os

import matplotlib.pyplot as plt
import numpy as np


SHORT_BUCKET_IDX = 1   # second 5-year bucket — e.g. 2000-2004 (5y of intervention)
LONG_BUCKET_IDX = -1   # last bucket — e.g. 2020-2024 (~25y of intervention)
DEFAULT_METRIC = "C_pct"

METRIC_CONFIG = {
    "share": {
        "bucket_key": "share_buckets",
        "final_key": "share_final",
        "trajectory_key": "share_target",
        "scale": 100.0,
        "relative": False,
        "level_label": "Target country market share (%)",
        "delta_label": "market-share delta vs baseline (percentage points)",
    },
    "C": {
        "bucket_key": "C_buckets",
        "final_key": "final_C",
        "trajectory_key": "C_target",
        "scale": 1.0,
        "relative": False,
        "level_label": "Target country C",
        "delta_label": "C delta vs baseline",
    },
    # Change in the studied country's OWN exports, as a percentage of its
    # baseline value: (C_cf - C_base) / C_base * 100. This is the metric used
    # in the text and tables; it only involves the studied country, not world
    # totals, so it is the appropriate figure metric for this experiment.
    "C_pct": {
        "bucket_key": "C_buckets",
        "final_key": "final_C",
        "trajectory_key": "C_target",
        "scale": 100.0,
        "relative": True,
        "level_label": "Target country exports C",
        "delta_label": "change in country exports vs baseline (%)",
    },
}


def _country_tag(results_dict):
    code = (results_dict.get("settings") or {}).get("country")
    return f" — {code}" if code else ""


def _bucket_labels(results_dict):
    return (results_dict.get("settings") or {}).get("bucket_labels") or []


def _metric_config(metric):
    if metric not in METRIC_CONFIG:
        raise ValueError(f"unknown metric {metric!r}; choose one of {sorted(METRIC_CONFIG)}")
    return METRIC_CONFIG[metric]


def _finite_or_none(value):
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def _bucket_value(run, bucket_idx, metric=DEFAULT_METRIC):
    """Return a bucket-level metric value, or None if unavailable."""
    if run is None:
        return None
    cfg = _metric_config(metric)
    arr = run.get(cfg["bucket_key"])
    if arr is None:
        return None
    try:
        v = arr[bucket_idx]
    except (IndexError, TypeError):
        return None
    return _finite_or_none(v)


def _endpoint_value(run, metric=DEFAULT_METRIC):
    if run is None:
        return None
    cfg = _metric_config(metric)
    return _finite_or_none(run.get(cfg["final_key"], run.get(f"{cfg['final_key']}_2024")))


def _horizon_value(run, baseline, bucket_idx, metric=DEFAULT_METRIC):
    """Prefer bucket means; fall back to final endpoint for the long horizon."""
    v = _bucket_value(run, bucket_idx, metric)
    b = _bucket_value(baseline, bucket_idx, metric)
    if v is not None and b is not None:
        return v, b
    if bucket_idx == LONG_BUCKET_IDX:
        return _endpoint_value(run, metric), _endpoint_value(baseline, metric)
    return None, None


def _horizon_delta(run, baseline, bucket_idx, metric=DEFAULT_METRIC):
    """Return the run-vs-baseline delta for a horizon, or None.

    For absolute metrics this is ``run - baseline``; for relative metrics
    (e.g. C_pct) it is the fractional change ``(run - baseline) / baseline``,
    which is turned into a percentage by the metric's scale in _scale_values.
    """
    a, b = _horizon_value(run, baseline, bucket_idx, metric)
    if a is None or b is None:
        return None
    if _metric_config(metric).get("relative"):
        if b == 0:
            return None
        return (a - b) / b
    return a - b


def _short_long_labels(results_dict):
    labels = _bucket_labels(results_dict)
    short = labels[SHORT_BUCKET_IDX] if len(labels) > SHORT_BUCKET_IDX else None
    long_idx = LONG_BUCKET_IDX if LONG_BUCKET_IDX >= 0 else len(labels) + LONG_BUCKET_IDX
    long = labels[long_idx] if labels and 0 <= long_idx < len(labels) else "final year"
    return short, long


def _resolved_long_idx(results_dict):
    labels = _bucket_labels(results_dict)
    if LONG_BUCKET_IDX < 0 and labels:
        return len(labels) + LONG_BUCKET_IDX
    return LONG_BUCKET_IDX


def _scale_values(values, metric):
    scale = _metric_config(metric)["scale"]
    return [np.nan if v is None else scale * v for v in values]


def _grid_edges(values):
    values = np.asarray(sorted(values), dtype=float)
    if values.size == 1:
        width = 1.0 if values[0] == 0 else abs(values[0]) * 0.1
        return np.array([values[0] - width, values[0] + width])
    mids = (values[:-1] + values[1:]) / 2.0
    first = values[0] - (mids[0] - values[0])
    last = values[-1] + (values[-1] - mids[-1])
    return np.concatenate([[first], mids, [last]])


def plot_strategy_comparison_C(results_dict, output_dir):
    """One line per strategy + baseline: the country's own exports C vs year."""
    fig, ax = plt.subplots(figsize=(9, 5))
    baseline = results_dict.get("baseline")
    if baseline and baseline.get("trajectory"):
        ax.plot(
            baseline["trajectory"]["years"],
            np.asarray(baseline["trajectory"]["C_target"]),
            color="black", lw=2.0, ls="--", label="baseline (strength=0)",
        )
    for name, run in results_dict["strategies"].items():
        if run is None or not run.get("trajectory"):
            continue
        ax.plot(
            run["trajectory"]["years"],
            np.asarray(run["trajectory"]["C_target"]),
            lw=1.6, label=name,
        )
    ax.set_xlabel("Year")
    ax.set_ylabel(METRIC_CONFIG["C"]["level_label"])
    ax.set_title(f"Strategy comparison — country exports trajectory under alpha injection{_country_tag(results_dict)}")
    ax.legend(fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    out = os.path.join(output_dir, "strategy_comparison_C.png")
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def plot_strategy_comparison_alpha(results_dict, output_dir, top_k=10):
    """Small-multiples of basket alpha mass when baskets are saved.

    Older result files did not store baskets; for those, fall back to the
    previous top-alpha trajectory plot.
    """
    strategies = [
        (n, r) for n, r in results_dict["strategies"].items()
        if r and r.get("trajectory")
    ]
    n = len(strategies)
    if n == 0:
        return None
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 3.4), sharey=True)
    if n == 1:
        axes = [axes]
    baseline = results_dict.get("baseline")
    for ax, (name, run) in zip(axes, strategies):
        years = run["trajectory"]["years"]
        alpha_t = np.asarray(run["trajectory"]["alpha_target"])  # (T, SP)
        basket = run.get("basket")
        if basket:
            basket = np.asarray(basket, dtype=int)
            ax.plot(years, alpha_t[:, basket].sum(axis=1), lw=1.8,
                    label="strategy")
            if baseline and baseline.get("trajectory"):
                alpha_base = np.asarray(baseline["trajectory"]["alpha_target"])
                if alpha_base.ndim == 2 and alpha_base.shape[1] > basket.max():
                    ax.plot(
                        baseline["trajectory"]["years"],
                        alpha_base[:, basket].sum(axis=1),
                        color="black", lw=1.2, ls="--", label="baseline",
                    )
            ax.set_ylabel("basket alpha mass")
            ax.legend(fontsize=7, frameon=False)
        else:
            top = np.argsort(alpha_t.mean(axis=0))[-top_k:]
            for i in top:
                ax.plot(years, alpha_t[:, i], lw=0.8, alpha=0.8)
            ax.set_ylabel(f"top-{top_k} alpha")
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("Year")
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"Injected alpha concentration by strategy{_country_tag(results_dict)}", fontsize=11)
    out = os.path.join(output_dir, "strategy_comparison_alpha.png")
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def plot_strategy_effect_summary(results_dict, output_dir, metric=DEFAULT_METRIC):
    """Bar chart of default strategy effects against baseline."""
    baseline = results_dict.get("baseline")
    if baseline is None or baseline.get("status") != "ok":
        return None
    short_label, long_label = _short_long_labels(results_dict)
    long_idx = _resolved_long_idx(results_dict)
    names, y_short, y_long = [], [], []
    for name, run in results_dict.get("strategies", {}).items():
        if run is None or run.get("status") != "ok":
            continue
        names.append(name)
        y_short.append(_horizon_delta(run, baseline, SHORT_BUCKET_IDX, metric))
        y_long.append(_horizon_delta(run, baseline, long_idx, metric))
    if not names:
        return None

    cfg = _metric_config(metric)
    x = np.arange(len(names))
    width = 0.36
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    if any(v is not None for v in y_short) and short_label is not None:
        ax.bar(x - width / 2, _scale_values(y_short, metric), width,
               label=f"short ({short_label})", color="#7aa6c2")
        x_long = x + width / 2
    else:
        x_long = x
    ax.bar(x_long, _scale_values(y_long, metric), width,
           label=f"long ({long_label})", color="#315f72")
    ax.axhline(0.0, color="grey", lw=0.8, ls=":")
    ax.set_xticks(x, names, rotation=25, ha="right")
    ax.set_ylabel(cfg["delta_label"])
    ax.set_title(f"Default strategy effect vs baseline{_country_tag(results_dict)}")
    ax.legend(fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    out = os.path.join(output_dir, "strategy_effect_summary.png")
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def plot_strategy_transient(results_dict, output_dir, metric=DEFAULT_METRIC):
    """Per-bucket Δ vs baseline for each default strategy, across ALL buckets.

    The short/long summary collapses each strategy's effect to two horizons,
    which hides spike-then-revert behaviour: several strategies post a large
    early gain that decays or flips sign by the final bucket (e.g. CHN
    high_complexity_offgrid is strongly positive in 2000-04 and slightly
    negative by 2010-14). This plots the full bucket-by-bucket delta profile at
    the default (strength, K) so transients are explicit rather than inferred.
    """
    baseline = results_dict.get("baseline")
    if baseline is None or baseline.get("status") != "ok":
        return None
    labels = _bucket_labels(results_dict)
    if not labels:
        return None
    cfg = _metric_config(metric)
    n_buckets = len(labels)
    x = np.arange(n_buckets)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.axhline(0.0, color="grey", lw=0.8, ls=":", alpha=0.7)
    cmap = plt.get_cmap("tab10")
    plotted = False
    for strat_idx, (name, run) in enumerate(results_dict.get("strategies", {}).items()):
        if run is None or run.get("status") != "ok":
            continue
        deltas = [_horizon_delta(run, baseline, b, metric) for b in range(n_buckets)]
        ys = _scale_values(deltas, metric)
        if all((y is None or np.isnan(y)) for y in ys):
            continue
        ax.plot(x, ys, "o-", lw=1.6, ms=4, color=cmap(strat_idx % 10), label=name)
        plotted = True
    if not plotted:
        plt.close(fig)
        return None

    ax.set_xticks(x, labels, rotation=25, ha="right")
    ax.set_xlabel("5-year bucket")
    ax.set_ylabel(cfg["delta_label"])
    ax.set_title(
        f"Strategy effect over time — per-bucket Δ vs baseline{_country_tag(results_dict)}"
    )
    ax.legend(fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    out = os.path.join(output_dir, "strategy_transient.png")
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def _plot_sweep(results_dict, axis: str, output_dir, metric=DEFAULT_METRIC):
    """Two lines per strategy: short-horizon and long-horizon bucket delta vs baseline."""
    baseline = results_dict.get("baseline")
    if baseline is None or baseline.get("status") != "ok":
        baseline = None  # deltas fall back to None

    short_label, long_label = _short_long_labels(results_dict)
    long_idx_resolved = _resolved_long_idx(results_dict)
    cfg = _metric_config(metric)

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.axhline(0.0, color="grey", lw=0.8, ls=":", alpha=0.7)
    cmap = plt.get_cmap("tab10")
    for strat_idx, (name, runs) in enumerate(results_dict["sweeps"][axis].items()):
        runs_ok = [r for r in runs if r.get("status") == "ok"]
        if not runs_ok:
            continue
        runs_ok.sort(key=lambda r: r[axis])
        xs = [r[axis] for r in runs_ok]
        y_short = [_horizon_delta(r, baseline, SHORT_BUCKET_IDX, metric) for r in runs_ok]
        y_long = [_horizon_delta(r, baseline, long_idx_resolved, metric) for r in runs_ok]
        color = cmap(strat_idx % 10)
        if any(v is not None for v in y_short) and short_label is not None:
            ax.plot(xs, _scale_values(y_short, metric), "o--", lw=1.2, ms=4,
                    color=color, alpha=0.7, label=f"{name} — short ({short_label})")
        if any(v is not None for v in y_long):
            ax.plot(xs, _scale_values(y_long, metric), "o-", lw=1.8, ms=5,
                    color=color, label=f"{name} — long ({long_label})")
    ax.set_xlabel(axis)
    ax.set_ylabel(cfg["delta_label"])
    ax.set_title(f"{axis} sweep — short vs long horizon, Δ vs baseline{_country_tag(results_dict)}")
    ax.legend(fontsize=7, frameon=False, ncol=2, loc="best")
    ax.spines[["top", "right"]].set_visible(False)
    out = os.path.join(output_dir, f"sweep_{axis}.png")
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def plot_sweep_strength(results_dict, output_dir):
    return _plot_sweep(results_dict, "strength", output_dir, metric=DEFAULT_METRIC)


def plot_sweep_K(results_dict, output_dir):
    return _plot_sweep(results_dict, "K", output_dir, metric=DEFAULT_METRIC)


def plot_heatmaps(results_dict, output_dir, metric=DEFAULT_METRIC):
    """Two panels per strategy: strength x K -> Δ mean C vs baseline, for the
    short-horizon and long-horizon buckets."""
    baseline = results_dict.get("baseline")
    if baseline is not None and baseline.get("status") != "ok":
        baseline = None

    short_label, long_label = _short_long_labels(results_dict)
    long_idx_resolved = _resolved_long_idx(results_dict)
    cfg = _metric_config(metric)

    paths = []
    for name, runs in results_dict["heatmaps"].items():
        if not runs:
            continue
        strengths = sorted(set(r["strength"] for r in runs))
        Ks = sorted(set(r["K"] for r in runs))
        Z_short = np.full((len(Ks), len(strengths)), np.nan)
        Z_long = np.full((len(Ks), len(strengths)), np.nan)
        for r in runs:
            if r.get("status") != "ok":
                continue
            i = Ks.index(r["K"])
            j = strengths.index(r["strength"])
            ds = _horizon_delta(r, baseline, SHORT_BUCKET_IDX, metric)
            dl = _horizon_delta(r, baseline, long_idx_resolved, metric)
            if ds is not None:
                Z_short[i, j] = cfg["scale"] * ds
            if dl is not None:
                Z_long[i, j] = cfg["scale"] * dl

        # Symmetric diverging scale shared across the two panels.
        finite = np.concatenate([
            Z_short[np.isfinite(Z_short)].ravel(),
            Z_long[np.isfinite(Z_long)].ravel(),
        ]) if (np.any(np.isfinite(Z_short)) or np.any(np.isfinite(Z_long))) else np.array([0.0])
        vmax = float(np.max(np.abs(finite))) if finite.size else 0.0
        vmax = vmax if vmax > 0 else 1.0
        vmin = -vmax

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
        strength_edges = _grid_edges(strengths)
        K_edges = _grid_edges(Ks)
        for ax, Z, panel_label in (
            (axes[0], Z_short, f"short ({short_label})"),
            (axes[1], Z_long, f"long ({long_label})"),
        ):
            im = ax.pcolormesh(
                strength_edges, K_edges, Z,
                cmap="RdBu_r", vmin=vmin, vmax=vmax,
                shading="auto",
            )
            ax.set_xlabel("strength")
            ax.set_title(panel_label, fontsize=10)
        axes[0].set_ylabel("K (basket size)")
        fig.colorbar(im, ax=axes, label=cfg["delta_label"],
                     fraction=0.045, pad=0.04)
        fig.suptitle(f"Heatmap — {name}{_country_tag(results_dict)}",
                     fontsize=11, y=1.02)
        path = os.path.join(output_dir, f"heatmap_{name}.png")
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
    return paths
