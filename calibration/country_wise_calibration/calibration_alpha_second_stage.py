"""
Second-stage country-wise alpha calibration.

Thin wrapper around calibration_country_wise.py:
- start from country_wise_alpha_frozen/<COUNTRY>/best_theta.npy
- tune only s_pi, beta_trade_off, G, nu
- keep entry disabled during every simulation
- minimise alpha loss while penalising deterioration of the first-stage C fit
"""

import argparse
import csv
import json
import multiprocessing as mp
import os
import sys
import time
from argparse import Namespace

import numpy as np
from scipy.optimize import differential_evolution

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(THIS_DIR))
sys.path[:0] = [ROOT, os.path.join(ROOT, "calibration"), THIS_DIR]

import calibration_country_wise as ccw  # noqa: E402
from calibration_config import CALIB_DIR, MAX_PARALLEL_JOBS, SEED, TRAJECTORY_TIMEOUT_S  # noqa: E402
from calibration_utils import PENALTY, load_data  # noqa: E402


OUTPUT_DIR = os.path.join(CALIB_DIR, "country_wise_alpha_second_stage")
TUNED_PARAM_NAMES = ("s_pi", "beta_trade_off", "G", "nu")
TUNED_IDX = tuple(ccw.COUNTRY_PARAM_NAMES.index(n) for n in TUNED_PARAM_NAMES)
TUNED_BOUNDS = [ccw.COUNTRY_PARAM_BOUNDS[i] for i in TUNED_IDX]


def entry_off():
    ccw.FIXED["enable_entry"] = False


def load_base_theta(params_dir, code):
    path = os.path.join(params_dir, code, "best_theta.npy")
    if not os.path.exists(path):
        return None
    theta = np.load(path).astype(float)
    if theta.shape != (len(ccw.COUNTRY_PARAM_NAMES),):
        return None
    return theta


def load_baseline_C_loss(params_dir, code):
    path = os.path.join(params_dir, code, "de_result.json")
    with open(path) as handle:
        comps = json.load(handle)["loss_components"]
    # Same C-only weights used by alpha_frozen mode in calibration_country_wise.
    return float(0.45 * comps["nrmse_C"] + 0.55 * comps["traj_corr_C"])


def with_tuned(base_theta, tuned_theta):
    theta = np.asarray(base_theta, dtype=float).copy()
    theta[list(TUNED_IDX)] = np.asarray(tuned_theta, dtype=float)
    return theta


def C_loss(agg):
    return float(0.45 * agg["nrmse_C"] + 0.55 * agg["traj_corr_C"])


def alpha_loss(agg, rank_weight, traj_weight):
    """
    Alpha-trajectory loss. Three components, weights sum to 1:
      - per-year log-NRMSE on the alpha row (`nrmse_P`)
      - per-year rank correlation on the alpha row (`rank_products`)
      - per-product time-trajectory correlation (`traj_corr_alpha`), aggregated
        as a weighted mean over products by observed alpha mass
    """
    rank_weight = float(rank_weight)
    traj_weight = float(traj_weight)
    nrmse_weight = max(0.0, 1.0 - rank_weight - traj_weight)
    return float(
        nrmse_weight * agg["nrmse_P"]
        + rank_weight * agg["rank_products"]
        + traj_weight * agg["traj_corr_alpha"]
    )


def evaluate_candidate(tuned_theta, base_theta, data, country_idx, baseline_C, args):
    theta = with_tuned(base_theta, tuned_theta)
    traj = ccw.windowed_simulate_country_conditioned(
        theta,
        data,
        country_idx,
        ccw.COUNTRY_CALIB_YEARS,
        mode="conditioned",
    )
    if traj is None:
        return PENALTY
    agg = ccw.aggregate_country_loss(
        traj,
        data,
        country_idx,
        years=ccw.COUNTRY_CALIB_YEARS,
        mode="conditioned",
    )
    if agg is None:
        return PENALTY
    c_ratio = C_loss(agg) / max(float(baseline_C), 1e-12)
    c_excess = max(0.0, c_ratio - (1.0 + args.c_tolerance))
    return float(
        alpha_loss(agg, args.alpha_rank_weight, args.alpha_traj_weight)
        + args.c_penalty * c_excess ** 2
    )


def final_metrics(theta, data, country_idx, baseline_C, args):
    traj = ccw.windowed_simulate_country_conditioned(
        theta,
        data,
        country_idx,
        ccw.COUNTRY_CALIB_YEARS,
        mode="conditioned",
    )
    if traj is None:
        return {}
    agg = ccw.aggregate_country_loss(
        traj,
        data,
        country_idx,
        years=ccw.COUNTRY_CALIB_YEARS,
        mode="conditioned",
    )
    if agg is None:
        return {}
    c = C_loss(agg)
    a = alpha_loss(agg, args.alpha_rank_weight, args.alpha_traj_weight)
    return {
        "C_loss": c,
        "alpha_loss": a,
        "C_loss_baseline_alpha_frozen": float(baseline_C),
        "C_loss_ratio_vs_baseline": c / max(float(baseline_C), 1e-12),
        "nrmse_C": float(agg["nrmse_C"]),
        "traj_corr_C": float(agg["traj_corr_C"]),
        "nrmse_P": float(agg["nrmse_P"]),
        "rank_products": float(agg["rank_products"]),
        "traj_corr_alpha": float(agg["traj_corr_alpha"]),
        "entry_enabled": bool(ccw.FIXED.get("enable_entry", False)),
    }


def calibrate_country(row, data, args):
    code = row["location_code"]
    country_idx = int(row["position"])
    out_dir = os.path.join(args.output_dir, code)
    os.makedirs(out_dir, exist_ok=True)

    base_theta = load_base_theta(args.params_dir, code)
    if base_theta is None:
        print(f"[skip] {code}: missing/incompatible base theta", flush=True)
        return None, None
    baseline_C = load_baseline_C_loss(args.params_dir, code)

    print(f"\n{'=' * 72}")
    print(f"Second-stage alpha calibration: {code} ({row['country_name_short']})")
    print(f"{'=' * 72}")
    print(
        f"  tuned={list(TUNED_PARAM_NAMES)} | entry={ccw.FIXED.get('enable_entry')} | "
        f"baseline_C={baseline_C:.5f}",
        flush=True,
    )

    seed = SEED + 1901 * country_idx
    theta_all = ccw.lhs_maximin(TUNED_BOUNDS, args.lhs_samples, seed)
    theta_all[0] = base_theta[list(TUNED_IDX)]

    def loss_func(tuned):
        return evaluate_candidate(tuned, base_theta, data, country_idx, baseline_C, args)

    print(
        f"  LHS: evaluating {len(theta_all)} candidates "
        f"(workers={args.n_jobs}, timeout={args.timeout_s:.0f}s)...",
        flush=True,
    )
    t_lhs = time.time()
    lhs_map = ccw._TimedProcessMap(args.n_jobs, args.timeout_s, PENALTY)
    loss_all = np.asarray(lhs_map(loss_func, theta_all), dtype=float)
    finite = loss_all < PENALTY * 0.1
    if not np.any(finite):
        raise RuntimeError(f"{code}: LHS produced no finite evaluations")

    finite_theta = theta_all[finite]
    finite_loss = loss_all[finite]
    n_elite = max(1, int(np.ceil(ccw.FAST_LHS_ELITE_FRACTION * len(finite_theta))))
    elite = finite_theta[np.argsort(finite_loss)[:n_elite]]
    bounds = ccw.nroy_bounding_box(elite, TUNED_BOUNDS, ccw.FAST_LHS_PADDING)
    print(
        f"  LHS done in {time.time() - t_lhs:.0f}s | "
        f"finite={int(finite.sum())}/{len(theta_all)} | "
        f"best={float(np.min(finite_loss)):.5f} | "
        f"elite={n_elite}",
        flush=True,
    )

    np.save(os.path.join(out_dir, "lhs_theta_all.npy"), theta_all)
    np.save(os.path.join(out_dir, "lhs_loss_all.npy"), loss_all)
    with open(os.path.join(out_dir, "lhs_nroy_bounds.json"), "w") as handle:
        json.dump({"params": list(TUNED_PARAM_NAMES), "bounds": bounds}, handle, indent=2)

    pop_size = ccw.FAST_DE_POPSIZE * len(TUNED_PARAM_NAMES)
    workers = min(args.n_jobs, pop_size)
    de_map = ccw._TimedProcessMap(workers, args.timeout_s, PENALTY)
    rng = np.random.default_rng(SEED + 2701 * country_idx)
    if len(finite_theta) >= pop_size:
        n_best = max(1, pop_size // 2)
        best_idx = np.argsort(finite_loss)[:n_best]
        rand_idx = rng.choice(len(finite_theta), pop_size - n_best, replace=False)
        init = np.vstack([finite_theta[best_idx], finite_theta[rand_idx]])
        init[0] = base_theta[list(TUNED_IDX)]
    else:
        init = "latinhypercube"

    de_log = []
    best_seen = [float(np.min(finite_loss))]
    t0 = time.time()

    def callback(xk, convergence):
        cached = de_map.last_results.get(ccw._theta_key(xk))
        if cached is not None:
            best_seen[0] = min(best_seen[0], float(cached))
        print(
            f"  DE gen {len(de_log) + 1:3d} | best={best_seen[0]:.5f} | "
            f"conv={float(convergence):.4f} | {time.time() - t0:.0f}s",
            flush=True,
        )
        de_log.append(
            {
                "generation": len(de_log) + 1,
                "best_loss": best_seen[0],
                "convergence": float(convergence),
                "time_s": time.time() - t0,
            }
        )

    result = differential_evolution(
        loss_func,
        bounds=bounds,
        seed=SEED + 3109 * country_idx,
        init=init,
        popsize=ccw.FAST_DE_POPSIZE,
        maxiter=args.de_maxiter,
        tol=ccw.FAST_DE_TOL,
        atol=0.0,
        mutation=ccw.FAST_DE_MUTATION,
        recombination=0.7,
        polish=False,
        updating="deferred",
        workers=de_map,
        callback=callback,
    )

    best_tuned = np.asarray(result.x, dtype=float)
    best_theta = with_tuned(base_theta, best_tuned)
    metrics = final_metrics(best_theta, data, country_idx, baseline_C, args)

    np.save(os.path.join(out_dir, "best_theta.npy"), best_theta)
    np.save(os.path.join(out_dir, "best_theta_second_stage_tuned.npy"), best_tuned)
    payload = {
        "mode": "alpha_second_stage",
        "country_idx": country_idx,
        "tuned_params": dict(zip(TUNED_PARAM_NAMES, map(float, best_tuned))),
        "best_loss": float(result.fun),
        "message": str(result.message),
        "nfev": int(result.nfev),
        "nit": int(result.nit),
        "params": dict(zip(ccw.COUNTRY_PARAM_NAMES, map(float, best_theta))),
        "baseline_C_loss": float(baseline_C),
        "final_metrics": metrics,
        "de_log": de_log,
    }
    with open(os.path.join(out_dir, "de_result.json"), "w") as handle:
        json.dump(payload, handle, indent=2)

    summary = {
        "country_idx": country_idx,
        "country_code": code,
        "country_name": row["country_name_short"],
        "best_loss": float(result.fun),
        "nfev": int(result.nfev),
        "nit": int(result.nit),
        "message": str(result.message),
        "entry_enabled": False,
        "baseline_C_loss": float(baseline_C),
    }
    for name, value in zip(TUNED_PARAM_NAMES, best_tuned):
        summary[f"tuned_{name}"] = float(value)
    for name, value in zip(ccw.COUNTRY_PARAM_NAMES, best_theta):
        summary[name] = float(value)
    for key, value in metrics.items():
        if isinstance(value, (int, float, np.floating)) and np.isfinite(value):
            summary[f"final_{key}"] = float(value)

    print(
        f"  best={result.fun:.5f} | alpha={metrics.get('alpha_loss', np.nan):.5f} | "
        f"C ratio={metrics.get('C_loss_ratio_vs_baseline', np.nan):.3f}",
        flush=True,
    )
    return summary, best_theta


def load_saved(row, output_dir):
    code = row["location_code"]
    result_path = os.path.join(output_dir, code, "de_result.json")
    theta_path = os.path.join(output_dir, code, "best_theta.npy")
    if not (os.path.exists(result_path) and os.path.exists(theta_path)):
        return None
    with open(result_path) as handle:
        result = json.load(handle)
    theta = np.load(theta_path)
    summary = {
        "country_idx": int(row["position"]),
        "country_code": code,
        "country_name": row["country_name_short"],
        "best_loss": float(result["best_loss"]),
        "nfev": int(result["nfev"]),
        "nit": int(result["nit"]),
        "message": str(result["message"]),
        "entry_enabled": bool(result.get("final_metrics", {}).get("entry_enabled", False)),
        "baseline_C_loss": float(result.get("baseline_C_loss", np.nan)),
    }
    for name, value in result.get("tuned_params", {}).items():
        summary[f"tuned_{name}"] = float(value)
    for name, value in zip(ccw.COUNTRY_PARAM_NAMES, theta):
        summary[name] = float(value)
    for key, value in result.get("final_metrics", {}).items():
        if isinstance(value, (int, float)) and np.isfinite(value):
            summary[f"final_{key}"] = float(value)
    return summary, theta


def write_summary(rows, output_dir):
    if not rows:
        return None
    path = os.path.join(output_dir, "summary.csv")
    fields = []
    for row in rows:
        fields.extend(k for k in row if k not in fields)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


class _PrefixedStream:
    """Prefix each printed line with the country code."""

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


def country_worker(row, args_dict, data, inner_n_jobs, conn):
    tag = f"[{row['location_code']}] "
    sys.stdout = _PrefixedStream(sys.stdout, tag)
    sys.stderr = _PrefixedStream(sys.stderr, tag)
    entry_off()
    args = Namespace(**args_dict)
    args.n_jobs = int(inner_n_jobs)
    try:
        conn.send(("ok", *calibrate_country(row, data, args)))
    except Exception as exc:
        conn.send(("err", f"[{row['location_code']}] {type(exc).__name__}: {exc}", None))
    finally:
        conn.close()


def run_parallel(rows, data, args, inner_n_jobs):
    ctx = mp.get_context("fork" if "fork" in mp.get_all_start_methods() else mp.get_all_start_methods()[0])
    pending = iter(rows)
    active = []
    results = {}

    def start(row):
        parent, child = ctx.Pipe(duplex=False)
        proc = ctx.Process(target=country_worker, args=(row, vars(args), data, inner_n_jobs, child))
        proc.start()
        child.close()
        return {"row": row, "proc": proc, "conn": parent}

    while True:
        while len(active) < args.countries_in_parallel:
            try:
                active.append(start(next(pending)))
            except StopIteration:
                break
        if not active:
            break
        for slot in list(active):
            code = slot["row"]["location_code"]
            if slot["conn"].poll():
                status, summary, theta = slot["conn"].recv()
                slot["conn"].close()
                slot["proc"].join(timeout=5)
                active.remove(slot)
                if status != "ok":
                    raise RuntimeError(summary)
                results[code] = (summary, theta)
            elif not slot["proc"].is_alive():
                slot["conn"].close()
                slot["proc"].join(timeout=1)
                active.remove(slot)
                raise RuntimeError(f"[{code}] worker exited without result")
        time.sleep(0.1)

    summaries, thetas = [], {}
    for row in rows:
        code = row["location_code"]
        summary, theta = results[code]
        if summary is not None:
            summaries.append(summary)
            thetas[code] = theta
    return summaries, thetas


def parse_args():
    p = argparse.ArgumentParser(description="Second-stage alpha calibration")
    p.add_argument("--countries", nargs="*")
    p.add_argument("--params-dir", default=ccw.ALPHA_FROZEN_DIR)
    p.add_argument("--output-dir", default=OUTPUT_DIR)
    p.add_argument("--n-jobs", type=int, default=min(8, MAX_PARALLEL_JOBS))
    p.add_argument("--countries-in-parallel", type=int, default=1)
    p.add_argument("--lhs-samples", type=int, default=120)
    p.add_argument("--de-maxiter", type=int, default=ccw.FAST_DE_MAXITER)
    p.add_argument("--timeout-s", type=float, default=TRAJECTORY_TIMEOUT_S)
    p.add_argument("--c-tolerance", type=float, default=0.05)
    p.add_argument("--c-penalty", type=float, default=100.0)
    p.add_argument("--alpha-rank-weight", type=float, default=0.3)
    p.add_argument(
        "--alpha-traj-weight",
        type=float,
        default=0.4,
        help="Weight on per-product time-trajectory correlation of alpha "
        "(remainder after rank-weight goes to per-year log-NRMSE).",
    )
    p.add_argument("--resume", action="store_true")
    p.add_argument("--no-plots", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    args.n_jobs = max(1, min(int(args.n_jobs), MAX_PARALLEL_JOBS))
    args.countries_in_parallel = max(1, int(args.countries_in_parallel))
    entry_off()
    os.makedirs(args.output_dir, exist_ok=True)

    print("Second-stage alpha calibration")
    print(f"  tuned params: {list(TUNED_PARAM_NAMES)}")
    print(f"  entry enabled: {ccw.FIXED.get('enable_entry')}")
    print(f"  params dir: {args.params_dir}")
    print(f"  output dir: {args.output_dir}")

    data = load_data()
    rows = ccw.load_country_index()
    data["r_C_growth_regression"] = ccw.load_growth_regression_r(rows)
    if args.countries:
        wanted = {c.upper() for c in args.countries}
        rows = [r for r in rows if r["location_code"] in wanted]
        missing = sorted(wanted - {r["location_code"] for r in rows})
        if missing:
            raise ValueError(f"Unknown country codes: {', '.join(missing)}")

    summaries, best_thetas, todo = [], {}, []
    for row in rows:
        if args.resume:
            loaded = load_saved(row, args.output_dir)
            if loaded:
                print(f"[resume] {row['location_code']}", flush=True)
                summary, theta = loaded
                summaries.append(summary)
                best_thetas[row["location_code"]] = theta
                continue
        todo.append(row)

    inner_n_jobs = max(1, args.n_jobs // args.countries_in_parallel)
    print(
        f"  parallelism: {args.countries_in_parallel} countries x "
        f"{inner_n_jobs} inner workers",
        flush=True,
    )
    if todo and args.countries_in_parallel > 1:
        new_summaries, new_thetas = run_parallel(todo, data, args, inner_n_jobs)
        summaries.extend(new_summaries)
        best_thetas.update(new_thetas)
    else:
        old_n_jobs = args.n_jobs
        args.n_jobs = inner_n_jobs
        try:
            for row in todo:
                summary, theta = calibrate_country(row, data, args)
                if summary is not None:
                    summaries.append(summary)
                    best_thetas[row["location_code"]] = theta
        finally:
            args.n_jobs = old_n_jobs

    summary_path = write_summary(summaries, args.output_dir)
    print(f"Saved summary to: {summary_path}")

    if not args.no_plots and best_thetas:
        old_dir, old_mode = ccw.COUNTRY_WISE_DIR, ccw._MODE
        ccw.COUNTRY_WISE_DIR = args.output_dir
        ccw._MODE = "conditioned"
        try:
            collected = ccw._collect_country_trajectories(best_thetas, rows, data)
            ccw.plot_country_wise_trajectory_fit(collected, rows)
            ccw.plot_country_wise_trajectory_summary(collected)
        finally:
            ccw.COUNTRY_WISE_DIR = old_dir
            ccw._MODE = old_mode


if __name__ == "__main__":
    main()
