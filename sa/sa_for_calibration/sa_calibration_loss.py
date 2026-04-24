"""
Sobol global sensitivity analysis of the calibration loss function.

Evaluates how all 6 PS-only parameters (s, c, c_prime, gamma, kappa, sigma)
affect the calibration loss and its two components (rank_C, rank_products),
using empirical UN Comtrade data (1988-2010) as fixed input.

This complements the hysteresis SA in ../sensitivity_analysis.py providing a
basis for the free/fixed parameter selection used in calibration.

Outputs (per parameter):
  loss_total — weighted aggregate calibration loss
  rank_C — 1 - Spearman(C_sim, C_obs), country activity rank
  rank_products — 1 - mean per-country Spearman(alpha_sim[j,:], alpha_obs[j,:])
"""

import os
import sys
import csv
import time
import warnings
import multiprocessing as _mp
import numpy as np
import matplotlib.pyplot as plt

from SALib.sample import sobol as saltelli
from SALib.analyze import sobol

# Directories
_this_dir = os.path.dirname(os.path.abspath(__file__))
_root_dir = os.path.dirname(os.path.dirname(_this_dir))
_calib_dir = os.path.join(_root_dir, "calibration")
sys.path.insert(0, _root_dir)
sys.path.insert(0, _calib_dir)

from calibration_config import (
    FIXED, CALIB_YEARS, LOSS_WEIGHTS, TIME_WEIGHT_SLOPE,
    YEAR_START, PARAM_NAMES, PARAM_BOUNDS, COUNTRY_GROUPS,
)
from calibration_utils import load_data, aggregate_loss_components, _patch_model, _build_r_C_from_groups
from product_space_model import ProductSpaceModel

# Configuration
N_SOBOL = int(os.environ.get("N_SOBOL_OVERRIDE", "1024"))
N_JOBS = int(os.environ.get("N_JOBS_OVERRIDE", "32"))
CALC_SECOND_ORDER = False  # Skip second-order to halve run count
EVAL_TIMEOUT_S = float(os.environ.get("SA_EVAL_TIMEOUT_S", "180"))
WORKER_MEM_MB = int(os.environ.get("SA_WORKER_MEM_MB", "3000"))

RESULTS_DIR = os.path.join(_this_dir, "results")

# Fix r_C groups to 0
_SA_FIXED = {"r_C_declining": 0.0, "r_C_rising": 0.0, "r_C_stable": 0.0}
SA_PARAM_NAMES = [p for p in PARAM_NAMES if p not in _SA_FIXED]
SA_PARAM_BOUNDS = [b for p, b in zip(PARAM_NAMES, PARAM_BOUNDS) if p not in _SA_FIXED]

PROBLEM = {
    "num_vars": len(SA_PARAM_NAMES),
    "names": SA_PARAM_NAMES,
    "bounds": [list(b) for b in SA_PARAM_BOUNDS],
}

OUTPUT_NAMES = ["loss_total", "nrmse_C", "traj_corr_C", "rank_products", "nrmse_P"]

COLORS = {
    "loss_total": ("#333333", "#999999"),
    "nrmse_C": ("#E64B35", "#F4A582"),
    "traj_corr_C": ("#4DBBD5", "#92C5DE"),
    "rank_products": ("#00A087", "#91D1C2"),
    "nrmse_P": ("#8491B4", "#B09C85"),
}


# Model construction

def _build_model_sa(params_dict, data):
    """
    Build ProductSpaceModel matching the calibration free parameters exactly.
    Fixed parameters are read from FIXED; r_C is built from group parameters.
    """
    fp = FIXED
    model = ProductSpaceModel(
        N_products = data["SP"],
        n_countries = data["SC"],
        patch_network = True,
        seed = 133,
        phi_space = data["phi_space"],
        s = float(fp["s"]),
        c = float(fp["c"]),
        c_prime = float(fp["c_prime"]),
        gamma = float(fp["gamma"]),
        kappa = float(params_dict["kappa"]),
        sigma = float(params_dict["sigma"]),
        nu = float(params_dict["nu"]),
        G = float(params_dict["G"]),
        q = float(fp["q"]),
        mu = float(fp["mu"]),
        beta_trade_off = float(params_dict["beta_trade_off"]),
        enable_entry = bool(fp.get("enable_entry", True)),
        entry_threshold = float(params_dict["entry_threshold"]),
    )
    _patch_model(model, data, params=params_dict,
                 h_mean=float(params_dict["h_mean"]),
                 C_diag_mean=float(params_dict["C_diag_mean"]),
                 C_offdiag_mean=float(params_dict["C_offdiag_mean"]))
    return model


# Simulation

def _simulate_sa(params_dict, data, years):
    """
    Year-by-year simulation from YEAR_START with the calibration parameter set.
    Returns dict {year: (alpha, C, P)} or None if failure.
    """
    model = _build_model_sa(params_dict, data)
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
                    n_steps = 10 * n_yr,
                    y0 = y0,
                    rtol = 1e-3,
                    atol = 1e-6,
                    max_solver_steps = 5000,
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

        # Allow product entry between yearly steps.
        model.activate_new_links(P_t, C_t, alpha_t)

        results[year] = (alpha_t, C_t, P_t)
        y0 = np.concatenate([P_t, C_t, alpha_t.flatten()])
        prev_year = year

    return results


# Evaluation

def _evaluate(params_row, data):
    """
    Evaluate calibration loss components for one parameter vector.
    Returns array [loss_total, nrmse_C, traj_corr_C, rank_products, nrmse_P].
    Returns NaN array on failure.
    """
    nan_out = np.full(len(OUTPUT_NAMES), np.nan)
    params_dict = dict(zip(SA_PARAM_NAMES, params_row))
    params_dict.update(_SA_FIXED)  # fix r_C groups to 0, entry_threshold to neutral value

    try:
        traj = _simulate_sa(params_dict, data, CALIB_YEARS)
        if traj is None:
            return nan_out

        agg = aggregate_loss_components(
            traj, data, years=CALIB_YEARS, loss_weights=LOSS_WEIGHTS,
            time_weight_slope=TIME_WEIGHT_SLOPE, year_start=YEAR_START
        )
        if agg is None:
            return nan_out

        return np.array([
            agg["loss_total"],
            agg["nrmse_C"],
            agg["traj_corr_C"],
            agg["rank_products"],
            agg["nrmse_P"],
        ])

    except Exception:
        return nan_out


def _slurm_mem_limit_mb():
    """
    Returns the job memory limit in MB, read from SLURM environment variables. Returns None if unavailable.
    """
    per_node = os.environ.get("SLURM_MEM_PER_NODE")
    if per_node and per_node.isdigit():
        return int(per_node)
    per_cpu = os.environ.get("SLURM_MEM_PER_CPU")
    cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    if per_cpu and per_cpu.isdigit() and cpus and cpus.isdigit():
        return int(per_cpu) * int(cpus)
    return None


def _resolve_n_workers():
    """
    Cap concurrency by CPUs and, when available, node memory.
    """
    n_cpus = _mp.cpu_count()
    requested = n_cpus if N_JOBS <= 0 else min(N_JOBS, n_cpus)
    mem_limit_mb = _slurm_mem_limit_mb()
    if mem_limit_mb is None:
        return max(1, requested), requested, None
    mem_cap = max(1, mem_limit_mb // max(1, WORKER_MEM_MB))
    return max(1, min(requested, mem_cap)), requested, mem_limit_mb


def _eval_worker_process(params_row, data, conn):
    """
    Evaluate one Sobol sample in a killable subprocess.
    """
    os.environ["_CALIB_SUBPROCESS"] = "1"
    try:
        values = np.asarray(_evaluate(params_row, data), dtype=np.float64)
        conn.send(("ok", values))
    except Exception as exc:
        conn.send(("err", repr(exc)))
    finally:
        conn.close()


class _TimedVectorEvaluator:
    """
    Evaluate Sobol samples in isolated subprocesses with timeouts.
    """

    def __init__(self, max_workers, timeout_s):
        self.ctx = _mp.get_context("spawn")
        self.max_workers = max(1, int(max_workers))
        self.timeout_s = float(timeout_s)
        self.nan_vec = np.full(len(OUTPUT_NAMES), np.nan, dtype=np.float64)

    def evaluate(self, candidates, data, progress_label=None):
        rows = [np.asarray(row, dtype=np.float64).copy() for row in candidates]
        if not rows:
            return np.empty((0, len(self.nan_vec)), dtype=np.float64)

        results = np.tile(self.nan_vec, (len(rows), 1))
        active = {}
        next_idx = 0
        completed = 0
        last_report = time.time()

        while completed < len(rows):
            while next_idx < len(rows) and len(active) < self.max_workers:
                parent_conn, child_conn = self.ctx.Pipe(duplex=False)
                proc = self.ctx.Process(
                    target=_eval_worker_process,
                    args=(rows[next_idx], data, child_conn),
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

                    if status == "ok":
                        values = np.asarray(payload, dtype=np.float64)
                        if values.shape == self.nan_vec.shape:
                            results[idx] = values
                        else:
                            print(
                                f"    [eval-error] sample={idx} unexpected output shape={values.shape}",
                                flush=True,
                            )
                    else:
                        print(f"    [eval-error] sample={idx} err={payload}", flush=True)
                    continue

                if now - state["start"] > self.timeout_s:
                    print(
                        f"    [eval-timeout] sample={idx} limit={self.timeout_s:.0f}s",
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
                        f"    [eval-error] sample={idx} worker exited without result",
                        flush=True,
                    )

            if progress_label is not None and (completed == len(rows) or now - last_report >= 30.0):
                pct = 100 * completed / len(rows)
                print(
                    f"    [{progress_label}] progress: {completed}/{len(rows)} done ({pct:.0f}%) "
                    f"| {len(active)} still running",
                    flush=True,
                )
                last_report = now

            if not progress:
                time.sleep(0.05)

        return results


# Sobol run

def run_sobol(data, n=N_SOBOL):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    param_values = saltelli.sample(PROBLEM, n, calc_second_order=CALC_SECOND_ORDER)
    n_runs = len(param_values)

    print("=" * 60)
    print("Sobol SA — Calibration Loss")
    print(f"Params : {PROBLEM['num_vars']}   N: {n}   Runs: {n_runs}")
    print(f"Calib years: {CALIB_YEARS}")
    print("=" * 60)

    # Pre-compile numba JIT functions in the main process.
    print("Pre-compiling numba cache...", flush=True)
    _evaluate(param_values[0], data)
    print("Numba cache ready.", flush=True)

    n_workers, requested_workers, mem_limit_mb = _resolve_n_workers()
    if mem_limit_mb is None:
        print(f"Workers: {n_workers} (requested={requested_workers})", flush=True)
    else:
        print(
            f"Workers: {n_workers} (requested={requested_workers}, "
            f"node_mem={mem_limit_mb}MB, worker_budget={WORKER_MEM_MB}MB)",
            flush=True,
        )

    evaluator = _TimedVectorEvaluator(max_workers=n_workers, timeout_s=EVAL_TIMEOUT_S)
    batches = []
    chunk = max(n_workers * 8, 1)
    for chunk_start in range(0, n_runs, chunk):
        chunk_end = min(chunk_start + chunk, n_runs)
        print(f"  chunk {chunk_start}–{chunk_end} / {n_runs}", flush=True)
        batch = evaluator.evaluate(
            param_values[chunk_start:chunk_end],
            data,
            progress_label=f"chunk {chunk_start}-{chunk_end}",
        )
        batches.append(batch)
    sys.stdout.flush()

    Y = np.vstack(batches)

    n_failed = np.isnan(Y).any(axis=1).sum()
    if n_failed:
        frac = n_failed / len(Y)
        print(f"  WARNING: {n_failed}/{len(Y)} ({frac:.1%}) runs failed or violated "
              f"c_prime <= c — imputing with column medians.")
        for col in range(Y.shape[1]):
            mask = np.isnan(Y[:, col])
            Y[mask, col] = np.nanmedian(Y[:, col])

    np.save(os.path.join(RESULTS_DIR, "Y.npy"), Y)
    np.save(os.path.join(RESULTS_DIR, "param_values.npy"), param_values)

    si_dict = {}
    for k, out_name in enumerate(OUTPUT_NAMES):
        Si = sobol.analyze(PROBLEM, Y[:, k].astype(float), calc_second_order=CALC_SECOND_ORDER,
                           print_to_console=False)
        for key in ("S1", "S1_conf", "ST", "ST_conf"):
            Si[key] = np.array([float(np.squeeze(v)) for v in Si[key]])
        if CALC_SECOND_ORDER and "S2" in Si and "S2_conf" in Si:
            Si["S2"] = np.array(Si["S2"], dtype=float)
            Si["S2_conf"] = np.array(Si["S2_conf"], dtype=float)
        si_dict[out_name] = Si
        print(f"\n  [{out_name}]")
        print(f"    {'Param':<12}  S1 +/- conf      ST +/- conf")
        for j, pname in enumerate(PROBLEM["names"]):
            print(f"    {pname:<12}"
                  f"  {float(Si['S1'][j]):+.3f} +/- {float(Si['S1_conf'][j]):.3f}"
                  f"  {float(Si['ST'][j]):+.3f} +/- {float(Si['ST_conf'][j]):.3f}")
        if CALC_SECOND_ORDER and "S2" in Si:
            pairs = []
            for i in range(len(PROBLEM["names"])):
                for j in range(i + 1, len(PROBLEM["names"])):
                    pairs.append((Si["S2"][i, j], Si["S2_conf"][i, j], PROBLEM["names"][i], PROBLEM["names"][j]))
            pairs = sorted(pairs, key=lambda x: abs(x[0]), reverse=True)
            print("    Top |S2| pairs:")
            for s2, s2c, pi, pj in pairs[:5]:
                print(f"      {pi} x {pj}: {s2:+.3f} +/- {s2c:.3f}")

    return {"Y": Y, "param_values": param_values, "Si": si_dict}


# Plotting

def _bar_pair(ax, names, S1, S1c, ST, STc, title, c1, c2):
    x = np.arange(len(names))
    w = 0.35
    ax.bar(x - w/2, S1, w, yerr=S1c, label="S1 (first-order)", color=c1, capsize=4, alpha=0.9)
    ax.bar(x + w/2, ST, w, yerr=STc, label="ST (total-order)", color=c2, capsize=4, alpha=0.85)
    ax.axhline(0, color="black", linewidth=0.7, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_ylabel("Sobol index")
    ax.legend(fontsize=8)
    ax.set_ylim(-0.1, 1.1)
    ax.spines[["top", "right"]].set_visible(False)


def plot_results(result):
    names = PROBLEM["names"]
    Si    = result["Si"]

    fig, axes = plt.subplots(
        len(OUTPUT_NAMES), 1,
        figsize=(max(8, len(names) * 1.4), len(OUTPUT_NAMES) * 3.5),
    )
    fig.suptitle("Sobol SA — Calibration Loss Components",
                 fontsize=13, fontweight="bold", y=1.01)

    for out_name, ax in zip(OUTPUT_NAMES, axes):
        s = Si[out_name]
        c1, c2 = COLORS[out_name]
        _bar_pair(ax, names, s["S1"], s["S1_conf"], s["ST"], s["ST_conf"],
                  f"Output: {out_name}", c1, c2)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(RESULTS_DIR, "sobol_calibration_loss.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close()


def plot_output_distributions(result):
    Y = result["Y"]
    fig, axes = plt.subplots(1, len(OUTPUT_NAMES),
                             figsize=(len(OUTPUT_NAMES) * 3.5, 3.5))
    fig.suptitle("Output Distributions Across Sobol Samples",
                 fontsize=11, fontweight="bold")

    for k, (out_name, ax) in enumerate(zip(OUTPUT_NAMES, axes)):
        vals = Y[:, k][~np.isnan(Y[:, k])]
        c1, _ = COLORS[out_name]
        ax.hist(vals, bins=30, color=c1, alpha=0.85, edgecolor="white")
        ax.axvline(np.median(vals), color="black", linestyle="--", linewidth=1.2,
                   label=f"median={np.median(vals):.3f}")
        ax.set_title(out_name, fontsize=9, fontweight="bold")
        ax.set_xlabel("Loss")
        ax.set_ylabel("Count")
        ax.legend(fontsize=7)
        ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "output_distributions.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close()


def save_indices_csv(result):
    path  = os.path.join(RESULTS_DIR, "sobol_indices.csv")
    names = PROBLEM["names"]
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["output", "parameter", "S1", "S1_conf", "ST", "ST_conf"])
        for out_name in OUTPUT_NAMES:
            Si = result["Si"][out_name]
            for j, pname in enumerate(names):
                writer.writerow([
                    out_name, pname,
                    round(Si["S1"][j], 5), round(Si["S1_conf"][j], 5),
                    round(Si["ST"][j], 5), round(Si["ST_conf"][j], 5),
                ])
    print(f"Saved: {path}")


def save_second_order_csv(result):
    """
    Save second-order Sobol interaction indices (S2) for each output.
    """
    if not CALC_SECOND_ORDER:
        return
    path = os.path.join(RESULTS_DIR, "sobol_indices_s2.csv")
    names = PROBLEM["names"]
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["output", "param_i", "param_j", "S2", "S2_conf"])
        for out_name in OUTPUT_NAMES:
            Si = result["Si"][out_name]
            if "S2" not in Si or "S2_conf" not in Si:
                continue
            for i in range(len(names)):
                for j in range(i + 1, len(names)):
                    writer.writerow([
                        out_name, names[i], names[j],
                        round(float(Si["S2"][i, j]), 5),
                        round(float(Si["S2_conf"][i, j]), 5),
                    ])
    print(f"Saved: {path}")


if __name__ == "__main__":
    extracted_dir = os.path.join(_calib_dir, "extracted_data")
    data = load_data(extracted_dir=extracted_dir, years=list(range(1988, 2025)))

    result = run_sobol(data, n=N_SOBOL)
    plot_results(result)
    plot_output_distributions(result)
    save_indices_csv(result)
    save_second_order_csv(result)

    print(f"\nAll results saved to: {RESULTS_DIR}/")
    print("Done.")
