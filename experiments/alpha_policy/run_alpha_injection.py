"""
Alpha-injection experiment: counterfactual alpha trajectories on one country.

This is the canonical alpha-policy experiment. Alpha is not simulated
endogenously. Instead, the target country's alpha row is deterministically fixed
at each year boundary:

    alpha_target[year] = f(alpha_obs[year], basket, strength)

By construction, strength=0 reproduces the alpha-frozen baseline (sim alpha
= observed alpha), whose C trajectories are validated by the first-stage
calibration. So the counterfactual is interpretable as a deviation from a
baseline that actually matches the data.

Task design:
  - 1 baseline (strength=0)
  - 5 strategies at calibrated defaults
  - strength sweep                       — 5 strategies x 7 points
  - basket-size sweep                    — 5 strategies x 7 points
  - heatmap (strength x K)                — 5 strategies x 7x7

Run:
  python experiments/alpha_policy/run_alpha_injection.py --country BRA --n-jobs 8
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np
import pandas as pd

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(THIS_DIR))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "calibration"))
sys.path.insert(0, os.path.join(ROOT, "calibration", "country_wise_calibration"))
sys.path.insert(0, THIS_DIR)

from calibration_config import CALIB_DIR, SEED, YEAR_START
from calibration_utils import load_data
from calibration_country_wise import (
    COUNTRY_PARAM_NAMES,
    build_country_model,
    country_theta_to_dict,
    load_country_index,
    load_growth_regression_r,
)

from strategies import STRATEGY_NAMES, build_strategies
from alpha_injection import inject
from sim_injection import simulate_injection_conditioned

EXP_YEAR_START = 1995
YEAR_END = 2024
SIM_YEARS = [y for y in range(EXP_YEAR_START, YEAR_END + 1) if y not in (1993, 1994)]
BUCKETS = [(1995, 1999), (2000, 2004), (2005, 2009),
           (2010, 2014), (2015, 2019), (2020, 2024)]
BUCKET_LABELS = [f"{a}-{b}" for a, b in BUCKETS]


def _bucket_means(traj: dict, country_idx: int):
    """Return per-bucket mean C_target and mean share_target.

    For each (start, end) inclusive bucket, average values over the years
    present in the trajectory that fall in [start, end].
    """
    C_buckets, share_buckets = [], []
    for start, end in BUCKETS:
        years_in = [y for y in traj.keys() if start <= y <= end]
        if not years_in:
            C_buckets.append(None)
            share_buckets.append(None)
            continue
        cs = [float(traj[y]["C"][country_idx]) for y in years_in]
        shs = []
        for y in years_in:
            tot = float(traj[y]["C"].sum())
            shs.append(float(traj[y]["C"][country_idx] / tot) if tot > 0 else 0.0)
        C_buckets.append(float(np.mean(cs)))
        share_buckets.append(float(np.mean(shs)))
    return C_buckets, share_buckets

DEFAULT_PARAMS_DIR = os.path.join(CALIB_DIR, "country_wise_alpha_frozen")


@dataclass
class Task:
    run_id: str
    kind: str
    strategy: Optional[str]
    basket: list
    strength: float
    K: int
    save_full_traj: bool


# ---------------------------------------------------------------------------
# PCI lookup from raw trade data (full world network, time-averaged).
# ---------------------------------------------------------------------------

def load_pci(products_index_path: str, raw_data_path: str) -> np.ndarray:
    """Return PCI vector aligned to products_index order.

    Uses the pre-computed PCI column from the raw Comtrade file. PCI is a
    product-year quantity repeated across country-product rows, so first
    collapse to one value per product-year and then average over all available
    years. The result is matched to the products in products_index.
    """
    products = pd.read_csv(products_index_path)
    raw = pd.read_csv(raw_data_path, usecols=["product_hs92_code", "year", "pci"],
                      low_memory=False)
    raw["product_hs92_code"] = raw["product_hs92_code"].astype(str).str.zfill(4)
    product_year_pci = (
        raw.dropna(subset=["pci"])
           .groupby(["product_hs92_code", "year"], as_index=False)["pci"]
           .mean()
    )
    pci_mean = product_year_pci.groupby("product_hs92_code")["pci"].mean()
    codes = products["hs_product_code"].astype(str).str.zfill(4)
    pci = codes.map(pci_mean).to_numpy(dtype=float)
    if np.any(np.isnan(pci)):
        missing = codes[np.isnan(pci)].tolist()
        raise ValueError(f"PCI missing for products: {missing}")
    return pci


# ---------------------------------------------------------------------------
# Revealed comparative advantage (within-system Balassa RCA).
# ---------------------------------------------------------------------------

def within_system_rca(C_vec: np.ndarray, alpha_mat: np.ndarray) -> np.ndarray:
    """Balassa RCA over the closed SC x SP system, identical in definition to
    calibration/data_extraction_all_years.compute_rca_from_exports.

    Exports are reconstructed as X_cp = C_c * alpha_cp (country c's exports of
    product p), and RCA_cp = (X_cp / X_c) / (X_p / X_world). RCA > 1 means the
    country is specialised in that product relative to the 19-country system.
    Returns an (SC, SP) array.
    """
    C_vec = np.asarray(C_vec, dtype=float)
    alpha_mat = np.asarray(alpha_mat, dtype=float)
    exports = C_vec[:, None] * alpha_mat
    country_totals = exports.sum(axis=1, keepdims=True)
    product_totals = exports.sum(axis=0, keepdims=True)
    world_total = float(exports.sum())
    if world_total <= 0:
        return np.zeros_like(exports)
    country_share = np.divide(
        exports, country_totals,
        out=np.zeros_like(exports), where=country_totals > 0,
    )
    world_share = product_totals / world_total
    return np.divide(
        country_share, world_share,
        out=np.zeros_like(exports), where=world_share > 0,
    )


# ---------------------------------------------------------------------------
# Task list construction.
# ---------------------------------------------------------------------------

def _strength_grid(low: float, high: float, n: int) -> np.ndarray:
    return np.linspace(low, high, n)


def _K_grid(low: int, high: int, n: int) -> np.ndarray:
    # integer, monotone, deduplicated
    raw = np.linspace(low, high, n)
    return np.unique(np.clip(np.round(raw).astype(int), 1, None))


def _basket_for(
    strategy_name: str,
    alpha_target: np.ndarray,
    phi_space: np.ndarray,
    K: int,
    seed: int,
    pci: np.ndarray,
    feasible_mask: np.ndarray = None,
    rca_target: np.ndarray = None,
) -> list[int]:
    strategies = build_strategies(alpha_target, phi_space, K=int(K),
                                  seed=seed, pci=pci,
                                  feasible_mask=feasible_mask,
                                  rca_target=rca_target)
    return [int(i) for i in strategies[strategy_name]]


def build_task_list(
    alpha_target: np.ndarray,
    phi_space: np.ndarray,
    pci: np.ndarray,
    seed: int,
    strength_default: float,
    K_default: int,
    strength_lo: float = 0.0,
    strength_hi: float = 0.5,
    K_lo: int = 5,
    K_hi: int = 40,
    sweep_n: int = 7,
    heatmap_n: int = 7,
    feasible_mask: np.ndarray = None,
    rca_target: np.ndarray = None,
) -> list[Task]:
    s_grid = _strength_grid(strength_lo, strength_hi, sweep_n)
    K_grid = _K_grid(K_lo, K_hi, sweep_n)
    s_grid_hm = _strength_grid(strength_lo, strength_hi, heatmap_n)
    K_grid_hm = _K_grid(K_lo, K_hi, heatmap_n)

    tasks: list[Task] = []
    # 1. baseline: strength=0 -> sim alpha == observed alpha (basket inert)
    baseline_basket = _basket_for("top_rca", alpha_target, phi_space,
                                  K_default, seed, pci, feasible_mask, rca_target)
    tasks.append(Task("baseline", "baseline", None, baseline_basket,
                      0.0, K_default, save_full_traj=True))

    # 2. each strategy at calibrated defaults
    for name in STRATEGY_NAMES:
        basket = _basket_for(name, alpha_target, phi_space, K_default, seed,
                             pci, feasible_mask, rca_target)
        tasks.append(Task(
            run_id=f"strategy__{name}",
            kind="strategy",
            strategy=name,
            basket=basket,
            strength=strength_default,
            K=K_default,
            save_full_traj=True,
        ))

    # 3. strength sweep (replaces G sweep). K fixed at default.
    for name in STRATEGY_NAMES:
        basket = _basket_for(name, alpha_target, phi_space, K_default, seed,
                             pci, feasible_mask, rca_target)
        for k, s in enumerate(s_grid):
            tasks.append(Task(
                run_id=f"strength_sweep__{name}__{k:02d}",
                kind="strength_sweep",
                strategy=name,
                basket=basket,
                strength=float(s),
                K=K_default,
                save_full_traj=False,
            ))

    # 4. K sweep (replaces nu sweep). strength fixed at default.
    #    Each K value uses its own basket (basket grows/shrinks accordingly).
    for name in STRATEGY_NAMES:
        for k, K in enumerate(K_grid):
            basket = _basket_for(name, alpha_target, phi_space, int(K), seed,
                                 pci, feasible_mask, rca_target)
            tasks.append(Task(
                run_id=f"K_sweep__{name}__{k:02d}",
                kind="K_sweep",
                strategy=name,
                basket=basket,
                strength=strength_default,
                K=int(K),
                save_full_traj=False,
            ))

    # 5. heatmap (strength x K) -- 7x7 per strategy.
    for name in STRATEGY_NAMES:
        for ki, K in enumerate(K_grid_hm):
            basket = _basket_for(name, alpha_target, phi_space, int(K), seed,
                                 pci, feasible_mask, rca_target)
            for si, s in enumerate(s_grid_hm):
                tasks.append(Task(
                    run_id=f"heatmap__{name}__{si:02d}_{ki:02d}",
                    kind="heatmap",
                    strategy=name,
                    basket=basket,
                    strength=float(s),
                    K=int(K),
                    save_full_traj=False,
                ))
    return tasks


# ---------------------------------------------------------------------------
# Per-task execution.
# ---------------------------------------------------------------------------

def run_one_task(
    task: Task,
    theta_base: np.ndarray,
    data: dict,
    country_idx: int,
) -> dict:
    model = build_country_model(theta_base, data, country_idx)
    traj = simulate_injection_conditioned(
        model, data, country_idx, SIM_YEARS,
        basket=task.basket,
        strength=float(task.strength),
        start_year=EXP_YEAR_START,
    )
    if traj is None:
        return {
            "run_id": task.run_id, "kind": task.kind, "strategy": task.strategy,
            "strength": task.strength, "K": task.K, "status": "failed",
            "basket": [int(i) for i in task.basket],
            "basket_size_actual": len(task.basket),
            "final_year": None, "final_C": None, "share_final": None,
            "final_C_2024": None, "share_2024": None,
            "C_buckets": None, "share_buckets": None,
            "bucket_labels": BUCKET_LABELS,
        }
    final_year = max(traj.keys())
    final = traj[final_year]
    C_total = float(final["C"].sum())
    share = float(final["C"][country_idx] / C_total) if C_total > 0 else 0.0
    C_buckets, share_buckets = _bucket_means(traj, country_idx)
    out = {
        "run_id": task.run_id, "kind": task.kind, "strategy": task.strategy,
        "strength": float(task.strength), "K": int(task.K),
        "basket": [int(i) for i in task.basket],
        "basket_size_actual": len(task.basket),
        "status": "ok",
        "final_year": int(final_year),
        "final_C": float(final["C"][country_idx]),
        "share_final": share,
        "final_C_2024": float(final["C"][country_idx]),
        "share_2024": share,
        "C_buckets": C_buckets,
        "share_buckets": share_buckets,
        "bucket_labels": BUCKET_LABELS,
    }
    if task.save_full_traj:
        years = sorted(traj.keys())
        out["trajectory"] = {
            "years": years,
            "C_target": [float(traj[y]["C"][country_idx]) for y in years],
            "share_target": [
                float(traj[y]["C"][country_idx] / traj[y]["C"].sum())
                for y in years
            ],
            "alpha_target": [traj[y]["alpha"][country_idx].tolist() for y in years],
        }
    return out


def _task_worker(task_dict, theta_base, data, country_idx, conn):
    os.environ["_CALIB_SUBPROCESS"] = "1"
    try:
        task = Task(**task_dict)
        result = run_one_task(task, theta_base, data, country_idx)
        conn.send(("ok", result))
    except Exception as exc:
        conn.send(("err", f"{type(exc).__name__}: {exc}"))
    finally:
        conn.close()


def dispatch_tasks(
    tasks: list[Task],
    theta_base: np.ndarray,
    data: dict,
    country_idx: int,
    n_jobs: int,
    timeout_s: float,
) -> list[dict]:
    n_jobs = int(n_jobs)
    if n_jobs < 1:
        raise ValueError(f"n_jobs must be >= 1, got {n_jobs}")
    timeout_s = float(timeout_s)
    if timeout_s <= 0:
        raise ValueError(f"timeout_s must be > 0, got {timeout_s}")

    methods = mp.get_all_start_methods()
    method = "fork" if "fork" in methods else methods[0]
    ctx = mp.get_context(method)

    results: list[Optional[dict]] = [None] * len(tasks)
    active: dict = {}
    next_idx = 0
    completed = 0
    last_report = time.time()

    while completed < len(tasks):
        while next_idx < len(tasks) and len(active) < n_jobs:
            parent_conn, child_conn = ctx.Pipe(duplex=False)
            proc = ctx.Process(
                target=_task_worker,
                args=(asdict(tasks[next_idx]), theta_base, data,
                      country_idx, child_conn),
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
                    status, payload = ("err", "worker closed pipe early")
                conn.close()
                proc.join(timeout=0.1)
                active.pop(idx)
                completed += 1
                progress = True
                if status == "ok":
                    results[idx] = payload
                else:
                    results[idx] = {
                        "run_id": tasks[idx].run_id, "kind": tasks[idx].kind,
                        "strategy": tasks[idx].strategy,
                        "strength": tasks[idx].strength, "K": tasks[idx].K,
                        "basket": [int(i) for i in tasks[idx].basket],
                        "basket_size_actual": len(tasks[idx].basket),
                        "status": "error", "error": payload,
                        "final_year": None, "final_C": None, "share_final": None,
                        "final_C_2024": None, "share_2024": None,
                    }
                continue

            if now - state["start"] > timeout_s:
                if hasattr(proc, "kill"):
                    proc.kill()
                else:
                    proc.terminate()
                proc.join(timeout=1.0)
                conn.close()
                active.pop(idx)
                completed += 1
                progress = True
                results[idx] = {
                    "run_id": tasks[idx].run_id, "kind": tasks[idx].kind,
                    "strategy": tasks[idx].strategy,
                    "strength": tasks[idx].strength, "K": tasks[idx].K,
                    "basket": [int(i) for i in tasks[idx].basket],
                    "basket_size_actual": len(tasks[idx].basket),
                    "status": "timeout",
                    "final_year": None, "final_C": None, "share_final": None,
                    "final_C_2024": None, "share_2024": None,
                }
                continue

            if not proc.is_alive():
                proc.join(timeout=0.1)
                conn.close()
                active.pop(idx)
                completed += 1
                progress = True
                results[idx] = {
                    "run_id": tasks[idx].run_id, "kind": tasks[idx].kind,
                    "strategy": tasks[idx].strategy,
                    "strength": tasks[idx].strength, "K": tasks[idx].K,
                    "basket": [int(i) for i in tasks[idx].basket],
                    "basket_size_actual": len(tasks[idx].basket),
                    "status": "exited_no_result",
                    "final_year": None, "final_C": None, "share_final": None,
                    "final_C_2024": None, "share_2024": None,
                }

        if completed == len(tasks) or now - last_report >= 30.0:
            pct = 100 * completed / max(1, len(tasks))
            print(f"  [dispatch] {completed}/{len(tasks)} ({pct:.0f}%) | "
                  f"{len(active)} running", flush=True)
            last_report = now

        if not progress:
            time.sleep(0.05)
    return results


# ---------------------------------------------------------------------------
# Aggregation / I/O.
# ---------------------------------------------------------------------------

def aggregate_results(results: list[dict], settings: dict) -> dict:
    by_id = {r["run_id"]: r for r in results if r is not None}

    out = {
        "settings": settings,
        "baseline": by_id.get("baseline"),
        "strategies": {},
        "sweeps": {"strength": {}, "K": {}},
        "heatmaps": {},
    }
    for name in STRATEGY_NAMES:
        out["strategies"][name] = by_id.get(f"strategy__{name}")
        out["sweeps"]["strength"][name] = []
        out["sweeps"]["K"][name] = []

    for r in results:
        if r is None:
            continue
        if r["kind"] == "strength_sweep":
            out["sweeps"]["strength"][r["strategy"]].append({
                "strength": r["strength"],
                "basket": r.get("basket"),
                "basket_size_actual": r.get("basket_size_actual"),
                "final_year": r.get("final_year"),
                "final_C": r.get("final_C"),
                "share_final": r.get("share_final"),
                "final_C_2024": r["final_C_2024"],
                "share_2024": r["share_2024"],
                "C_buckets": r.get("C_buckets"),
                "share_buckets": r.get("share_buckets"),
                "status": r["status"],
            })
        elif r["kind"] == "K_sweep":
            out["sweeps"]["K"][r["strategy"]].append({
                "K": r["K"],
                "basket": r.get("basket"),
                "basket_size_actual": r.get("basket_size_actual"),
                "final_year": r.get("final_year"),
                "final_C": r.get("final_C"),
                "share_final": r.get("share_final"),
                "final_C_2024": r["final_C_2024"],
                "share_2024": r["share_2024"],
                "C_buckets": r.get("C_buckets"),
                "share_buckets": r.get("share_buckets"),
                "status": r["status"],
            })
        elif r["kind"] == "heatmap":
            out["heatmaps"].setdefault(r["strategy"], []).append({
                "strength": r["strength"], "K": r["K"],
                "basket": r.get("basket"),
                "basket_size_actual": r.get("basket_size_actual"),
                "final_year": r.get("final_year"),
                "final_C": r.get("final_C"),
                "share_final": r.get("share_final"),
                "final_C_2024": r["final_C_2024"],
                "share_2024": r["share_2024"],
                "C_buckets": r.get("C_buckets"),
                "share_buckets": r.get("share_buckets"),
                "status": r["status"],
            })

    for name in STRATEGY_NAMES:
        out["sweeps"]["strength"][name].sort(key=lambda d: d["strength"])
        out["sweeps"]["K"][name].sort(key=lambda d: d["K"])

    return out


def write_results_json(results_dict: dict, output_dir: str) -> str:
    path = os.path.join(output_dir, "results.json")
    with open(path, "w") as f:
        json.dump(results_dict, f, indent=2)
    print(f"  wrote {path}")
    return path


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Alpha-injection experiment.")
    p.add_argument("--country", required=True, help="Country code, e.g. BRA")
    p.add_argument("--params-dir", default=DEFAULT_PARAMS_DIR,
                   help=f"Country-wise calibration results dir. Default: {DEFAULT_PARAMS_DIR}")
    p.add_argument("--strength", type=float, default=0.5,
                   help="Default injection strength (used for the named-strategy "
                        "runs and as the fixed value in the K sweep).")
    p.add_argument("--basket-size", type=int, default=10,
                   help="Default basket size K (used for baseline / named strategies "
                        "and as the fixed K in the strength sweep).")
    p.add_argument("--strength-lo", type=float, default=0.0)
    p.add_argument("--strength-hi", type=float, default=1.0)
    p.add_argument("--K-lo", type=int, default=5)
    p.add_argument("--K-hi", type=int, default=40)
    p.add_argument("--sweep-n", type=int, default=7)
    p.add_argument("--heatmap-n", type=int, default=7)
    p.add_argument("--n-jobs", type=int,
                   default=int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))
    p.add_argument("--timeout-s", type=float, default=600.0)
    p.add_argument("--output-dir", default=None,
                   help="Output directory. Defaults to experiments/alpha_policy/<COUNTRY>/.")
    return p.parse_args()


def load_country_theta(params_dir: str, country_code: str) -> np.ndarray:
    path = os.path.join(params_dir, country_code.upper(), "best_theta.npy")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing best_theta: {path}")
    theta = np.load(path)
    if theta.shape != (len(COUNTRY_PARAM_NAMES),):
        raise ValueError(
            f"theta shape {theta.shape} != ({len(COUNTRY_PARAM_NAMES)},)"
        )
    return theta


def main():
    args = parse_args()
    if args.output_dir is None:
        args.output_dir = os.path.join(THIS_DIR, args.country.upper())
    print(f"Alpha-injection experiment | country={args.country} "
          f"params_dir={args.params_dir}")
    print(f"  strength_default={args.strength} "
          f"K_default={args.basket_size} n_jobs={args.n_jobs}")

    print("Loading data...")
    data = load_data()
    rows = load_country_index()
    data["r_C_growth_regression"] = load_growth_regression_r(rows)

    target = next((r for r in rows if r["location_code"] == args.country.upper()),
                  None)
    if target is None:
        raise ValueError(f"Unknown country: {args.country}")
    country_idx = int(target["position"])

    theta = load_country_theta(args.params_dir, args.country)
    print(f"  loaded theta (len={theta.size}) for {args.country}")
    params = country_theta_to_dict(theta)

    alpha_target = data["alpha_obs"][EXP_YEAR_START][country_idx].astype(float)
    rca_target = within_system_rca(
        data["C_obs"][EXP_YEAR_START], data["alpha_obs"][EXP_YEAR_START]
    )[country_idx]
    print(f"  RCA (start year {EXP_YEAR_START}): {int((rca_target >= 1).sum())} "
          f"products with RCA>=1 (specialised), "
          f"{int((rca_target < 1).sum())} off-grid (RCA<1)")
    phi_space = data["phi_space"]
    pci = load_pci(
        os.path.join(ROOT, "calibration", "extracted_data", "products_index.csv"),
        os.path.join(ROOT, "calibration", "raw_data", "hs92_country_product_year_4.csv"),
    )
    # beta_C_ref is fully dense for this G20/top-100 panel (every country has
    # exported every product at some point), so every product is already
    # reachable and basket selection is unrestricted across all products.
    feasible_mask = None
    n_feasible = int(data["SP"])
    basket_restriction = (
        "none; beta_C_ref is fully dense for this panel so every product is "
        "feasible for every country; baskets are drawn from all products"
    )
    print(f"  feasible beta_C products: {n_feasible}/{data['SP']} "
          f"(beta_C_ref fully dense; all products eligible)")

    tasks = build_task_list(
        alpha_target=alpha_target,
        phi_space=phi_space,
        pci=pci,
        seed=int(SEED),
        strength_default=float(args.strength),
        K_default=int(args.basket_size),
        strength_lo=float(args.strength_lo),
        strength_hi=float(args.strength_hi),
        K_lo=int(args.K_lo),
        K_hi=int(args.K_hi),
        sweep_n=int(args.sweep_n),
        heatmap_n=int(args.heatmap_n),
        feasible_mask=feasible_mask,
        rca_target=rca_target,
    )
    print(f"  total tasks: {len(tasks)}")

    print(f"Dispatching {len(tasks)} tasks across {args.n_jobs} workers...")
    t0 = time.time()
    raw = dispatch_tasks(tasks, theta, data, country_idx,
                         n_jobs=args.n_jobs,
                         timeout_s=args.timeout_s)
    print(f"  elapsed: {time.time() - t0:.0f}s")

    settings = {
        "country": args.country.upper(), "country_idx": country_idx,
        "params_dir": args.params_dir,
        "strength_default": float(args.strength),
        "K_default": int(args.basket_size),
        "baseline": (
            "alpha-frozen country-wise parameters; full annual ODE with "
            "year-boundary alpha pinning/injection; strength=0 matches the "
            "calibration alpha-frozen baseline"
        ),
        "basket_restriction": basket_restriction,
        "basket_ranking": (
            "top_rca/bottom_rca and the high_complexity_offgrid threshold rank "
            f"by within-system Balassa RCA at year {EXP_YEAR_START} "
            "(offgrid = RCA<1); proximity is alpha-weighted product-space density"
        ),
        "n_feasible_products": n_feasible,
        "calibrated_theta": {n: float(params[n]) for n in COUNTRY_PARAM_NAMES},
        "n_tasks": len(tasks),
        "sweep_n": args.sweep_n, "heatmap_n": args.heatmap_n,
        "horizon_years": SIM_YEARS,
        "exp_year_start": EXP_YEAR_START,
        "bucket_labels": BUCKET_LABELS,
        "buckets": BUCKETS,
    }
    agg = aggregate_results(raw, settings)
    os.makedirs(args.output_dir, exist_ok=True)
    write_results_json(agg, args.output_dir)

    from plotting_injection import (
        plot_strategy_comparison_C,
        plot_strategy_comparison_alpha,
        plot_strategy_effect_summary,
        plot_strategy_transient,
        plot_sweep_strength,
        plot_sweep_K,
        plot_heatmaps,
    )
    print("Plotting...")
    plot_strategy_comparison_C(agg, args.output_dir)
    plot_strategy_comparison_alpha(agg, args.output_dir)
    plot_strategy_effect_summary(agg, args.output_dir)
    plot_strategy_transient(agg, args.output_dir)
    plot_sweep_strength(agg, args.output_dir)
    plot_sweep_K(agg, args.output_dir)
    plot_heatmaps(agg, args.output_dir)
    print("Done.")


if __name__ == "__main__":
    main()
