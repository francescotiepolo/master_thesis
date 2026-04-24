"""
Phase 2b: multi-objective NSGA-II optimizer.

Runs NSGA-II (pymoo) to find the Pareto optimum across 4 objectives:
nrmse_C, traj_corr_C, rank_products, nrmse_P. All are minimized.

Saves to NSGA2_DIR:
  pareto_F.npy    -- Pareto objective values  (n_pareto, 4)
  pareto_X.npy    -- Pareto parameter vectors (n_pareto, N_PARAMS)
  best_theta.npy  -- compromise point
  nsga2_log.json  -- summary: n_evals, n_pareto, best per objective, compromise
"""

import os
import json
import time
import multiprocessing as mp
import numpy as np

try:
    from calibration_config import (
        PARAM_NAMES, N_PARAMS, PARAM_BOUNDS,
        NSGA2_POP_SIZE, NSGA2_N_GEN, NSGA2_SEED,
        NSGA2_TIMEOUT_S,
        LOSS_OBJECTIVES,
        LHS_DIR, NSGA2_DIR,
    )
    from calibration_utils import (
        multi_objective_loss, load_nroy_bounds, PENALTY,
    )
except ModuleNotFoundError as exc:
    if exc.name not in {"calibration_config", "calibration_utils"}:
        raise
    from .calibration_config import (
        PARAM_NAMES, N_PARAMS, PARAM_BOUNDS,
        NSGA2_POP_SIZE, NSGA2_N_GEN, NSGA2_SEED,
        NSGA2_TIMEOUT_S,
        LOSS_OBJECTIVES,
        LHS_DIR, NSGA2_DIR,
    )
    from .calibration_utils import (
        multi_objective_loss, load_nroy_bounds, PENALTY,
    )

from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.termination.default import DefaultMultiObjectiveTermination
from pymoo.optimize import minimize as pymoo_minimize
from pymoo.core.population import Population
from pymoo.decomposition.asf import ASF


def _theta_key(theta):
    """
    Convert a parameter vector to a rounded tuple for use as a dict key,
    so previously evaluated candidates can be looked up without re-running the simulation.
    """
    return tuple(np.round(np.asarray(theta, dtype=np.float64), 12))


def _mo_eval_worker(theta, data, conn):
    """
    Evaluate one candidate in an isolated subprocess.
    """
    os.environ["_CALIB_SUBPROCESS"] = "1"
    try:
        values = np.asarray(multi_objective_loss(theta, data), dtype=np.float64)
        conn.send(("ok", values))
    except Exception as exc:
        conn.send(("err", repr(exc)))
    finally:
        conn.close()


class _TimedMultiObjectiveEvaluator:
    """
    Evaluate candidates in killable subprocesses with per-candidate timeouts.
    """

    def __init__(self, max_workers, timeout_s, n_obj):
        methods = mp.get_all_start_methods()
        method = "fork" if "fork" in methods else methods[0]
        self.ctx = mp.get_context(method)
        self.max_workers = max(1, int(max_workers))
        self.timeout_s = float(timeout_s)
        self.penalty_vec = np.full(n_obj, PENALTY, dtype=np.float64)
        self.cache = {}

    def evaluate(self, X, data, generation):
        """
        Evaluate a batch of candidates, reusing cached results when available.
        """
        candidates = [np.asarray(theta, dtype=np.float64).copy() for theta in X]
        results = np.empty((len(candidates), len(self.penalty_vec)), dtype=np.float64)
        active = {}
        pending = []

        for idx, theta in enumerate(candidates):
            key = _theta_key(theta)
            cached = self.cache.get(key)
            if cached is not None:
                results[idx] = cached
            else:
                pending.append((idx, theta, key))

        total_pending = len(pending)
        if total_pending == 0:
            print(f"  [gen {generation}] all {len(candidates)} candidates already evaluated (served from cache)", flush=True)
            return results

        n_cached = len(candidates) - total_pending
        print(
            f"  [gen {generation}] evaluating {total_pending} new candidates "
            f"({n_cached} reused from cache) | {self.max_workers} parallel workers | timeout={self.timeout_s:.0f}s/candidate",
            flush=True,
        )

        next_idx = 0
        completed = 0
        last_report = time.time()

        while completed < total_pending:
            while next_idx < total_pending and len(active) < self.max_workers:
                result_idx, theta, key = pending[next_idx]
                parent_conn, child_conn = self.ctx.Pipe(duplex=False)
                proc = self.ctx.Process(
                    target=_mo_eval_worker,
                    args=(theta, data, child_conn),
                )
                proc.start()
                child_conn.close()
                active[result_idx] = {
                    "proc": proc,
                    "conn": parent_conn,
                    "start": time.time(),
                    "key": key,
                }
                next_idx += 1

            progress = False
            now = time.time()

            for result_idx, state in list(active.items()):
                proc = state["proc"]
                conn = state["conn"]

                if conn.poll():
                    status, payload = conn.recv()
                    conn.close()
                    proc.join(timeout=0.1)
                    active.pop(result_idx)
                    completed += 1
                    progress = True

                    if status == "ok":
                        values = np.asarray(payload, dtype=np.float64)
                        if values.shape == self.penalty_vec.shape and np.all(np.isfinite(values)):
                            results[result_idx] = values
                            self.cache[state["key"]] = values.copy()
                        else:
                            print(
                                f"    [WARNING] gen={generation} candidate={result_idx}: "
                                f"ODE returned unexpected output shape {values.shape} — assigning penalty",
                                flush=True,
                            )
                            results[result_idx] = self.penalty_vec
                            self.cache[state["key"]] = self.penalty_vec.copy()
                    else:
                        print(
                            f"    [ERROR] gen={generation} candidate={result_idx}: "
                            f"simulation raised an exception — {payload}",
                            flush=True,
                        )
                        results[result_idx] = self.penalty_vec
                        self.cache[state["key"]] = self.penalty_vec.copy()
                    continue

                if now - state["start"] > self.timeout_s:
                    print(
                        f"    [TIMEOUT] gen={generation} candidate={result_idx}: "
                        f"ODE solve exceeded {self.timeout_s:.0f}s wall-clock limit — candidate skipped",
                        flush=True,
                    )
                    if hasattr(proc, "kill"):
                        proc.kill()
                    else:
                        proc.terminate()
                    proc.join(timeout=5.0)
                    conn.close()
                    active.pop(result_idx)
                    results[result_idx] = self.penalty_vec
                    self.cache[state["key"]] = self.penalty_vec.copy()
                    completed += 1
                    progress = True
                    continue

                if not proc.is_alive():
                    proc.join(timeout=0.1)
                    conn.close()
                    active.pop(result_idx)
                    results[result_idx] = self.penalty_vec
                    self.cache[state["key"]] = self.penalty_vec.copy()
                    completed += 1
                    progress = True
                    print(
                        f"    [ERROR] gen={generation} candidate={result_idx}: "
                        f"worker process crashed unexpectedly — assigning penalty",
                        flush=True,
                    )

            if completed == total_pending or now - last_report >= 30.0:
                pct = 100 * completed / total_pending
                print(
                    f"  [gen {generation}] progress: {completed}/{total_pending} done ({pct:.0f}%) "
                    f"| {len(active)} still running",
                    flush=True,
                )
                last_report = now

            if not progress:
                time.sleep(0.05)

        n_penalty = int(np.sum(np.any(results >= PENALTY, axis=1)))
        n_ok = len(candidates) - n_penalty
        print(
            f"  [gen {generation}] done — {n_ok}/{len(candidates)} candidates evaluated successfully"
            + (f", {n_penalty} assigned penalty (ODE timeout/crash)" if n_penalty > 0 else ""),
            flush=True,
        )
        return results


class _CalibrationProblem(Problem):
    """
    Calibration problem definition for NSGA-II.
    """
    def __init__(self, data, bounds, n_jobs=-1):
        xl = np.array([b[0] for b in bounds])
        xu = np.array([b[1] for b in bounds])
        n_obj = len(LOSS_OBJECTIVES)
        super().__init__(n_var=N_PARAMS, n_obj=n_obj, xl=xl, xu=xu)
        self.data = data
        self.n_jobs = mp.cpu_count() if n_jobs is None or n_jobs <= 0 else n_jobs
        self._eval_count = 0
        self._generation = 0
        self._evaluator = _TimedMultiObjectiveEvaluator(
            max_workers=self.n_jobs,
            timeout_s=NSGA2_TIMEOUT_S,
            n_obj=n_obj,
        )

    def _evaluate(self, X, out, *args, **kwargs):
        """
        Evaluate a batch of candidates.
        """
        t0 = time.time()
        self._generation += 1
        out["F"] = self._evaluator.evaluate(X, self.data, self._generation)
        self._eval_count += len(X)
        elapsed = time.time() - t0
        print(
            f"  [gen {self._generation}] generation complete | "
            f"batch={len(X)} | total evaluations so far={self._eval_count} | time={elapsed:.1f}s",
            flush=True,
        )


def run_nsga2(data, nroy_bounds=None, n_jobs=-1):
    """
    1. Start population from LHS NROY archive
    2. Run NSGA-II for up to NSGA2_N_GEN generations
    3. Select compromise point
    4. Save Pareto front and summary
    """
    os.makedirs(NSGA2_DIR, exist_ok=True)

    if nroy_bounds is None:
        nroy_bounds = load_nroy_bounds()

    bounds = [(lo, hi) for lo, hi in nroy_bounds]
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])

    rng = np.random.default_rng(NSGA2_SEED)

    # Build warm-start population
    seed_rows = []

    lhs_path = os.path.join(LHS_DIR, "nroy_theta.npy")
    if os.path.exists(lhs_path):
        lhs_theta = np.load(lhs_path)
        if lhs_theta.ndim != 2 or lhs_theta.shape[1] != N_PARAMS:
            raise ValueError(
                f"LHS archive dimension mismatch: expected (*, {N_PARAMS}), got {lhs_theta.shape}. "
                "Re-run `python calibration/run_pipeline.py lhs`."
            )
        n_from_lhs = min(len(lhs_theta), NSGA2_POP_SIZE)
        if n_from_lhs > 0:
            top_idx = rng.choice(len(lhs_theta), n_from_lhs, replace=False)
            seed_rows.append(lhs_theta[top_idx])
            print(f"Loaded {n_from_lhs} LHS NROY points as warm-start seeds")

    if seed_rows:
        seed_array = np.vstack(seed_rows)
    else:
        seed_array = np.empty((0, N_PARAMS))

    n_remaining = NSGA2_POP_SIZE - len(seed_array)
    if n_remaining > 0:
        random_fill = lo + rng.random((n_remaining, N_PARAMS)) * (hi - lo)
        seed_array = np.vstack([seed_array, random_fill]) if len(seed_array) > 0 else random_fill

    seed_array = seed_array[:NSGA2_POP_SIZE]
    init_pop = Population.new("X", seed_array)
    print(f"Initial population: {len(seed_array)} individuals")

    # Problem and algorithm
    problem = _CalibrationProblem(data, bounds, n_jobs=n_jobs)

    algorithm = NSGA2(
        pop_size=NSGA2_POP_SIZE,
        sampling=init_pop,
        crossover=SBX(prob=0.9, eta=10),
        mutation=PM(prob=0.2, eta=15),
    )

    termination = DefaultMultiObjectiveTermination(
        n_max_gen=NSGA2_N_GEN,
        period=50,
    )

    print(f"\n{'='*60}")
    print(f"NSGA-II multi-objective calibration")
    print(f"  Parameters   : {N_PARAMS} free parameters")
    print(f"  Objectives   : {LOSS_OBJECTIVES}")
    print(f"  Population   : {NSGA2_POP_SIZE} candidates per generation")
    print(f"  Max gens     : {NSGA2_N_GEN} (early-stop if Pareto front converges)")
    print(f"  Random seed  : {NSGA2_SEED}")
    print(f"{'='*60}\n")

    t0 = time.time()
    res = pymoo_minimize(
        problem,
        algorithm,
        termination,
        seed=NSGA2_SEED,
        verbose=True,
    )
    elapsed = time.time() - t0
    print(f"\nNSGA-II completed in {elapsed:.1f}s ({elapsed/3600:.2f}h)")

    # Extract Pareto front
    pareto_F = res.F   # (n_pareto, 4)
    pareto_X = res.X   # (n_pareto, N_PARAMS)

    print(f"\nPareto front size: {len(pareto_F)}")
    for k, name in enumerate(LOSS_OBJECTIVES):
        print(f"  {name:20s}: best={pareto_F[:, k].min():.5f}  "
              f"worst={pareto_F[:, k].max():.5f}")

    # Compromise point via ASF
    weights = np.array([0.25, 0.30, 0.25, 0.20])
    decomp = ASF()
    idx_best = decomp(pareto_F, weights).argmin()
    best_theta = pareto_X[idx_best]
    best_objectives = pareto_F[idx_best]

    print(f"\nCompromise point (ASF index={idx_best}):")
    for name, val in zip(PARAM_NAMES, best_theta):
        print(f"  {name:20s} = {val:.6f}")
    print(f"  Objectives:")
    for name, val in zip(LOSS_OBJECTIVES, best_objectives):
        print(f"    {name:20s} = {val:.5f}")

    # Save
    np.save(os.path.join(NSGA2_DIR, "pareto_F.npy"), pareto_F)
    np.save(os.path.join(NSGA2_DIR, "pareto_X.npy"), pareto_X)
    np.save(os.path.join(NSGA2_DIR, "best_theta.npy"), best_theta)

    log = {
        "n_evals": int(problem._eval_count),
        "n_generations": int(res.algorithm.n_gen) if res.algorithm is not None else NSGA2_N_GEN,
        "elapsed_s": float(elapsed),
        "n_pareto": int(len(pareto_F)),
        "best_per_objective": {
            name: float(pareto_F[:, k].min())
            for k, name in enumerate(LOSS_OBJECTIVES)
        },
        "compromise_idx": int(idx_best),
        "compromise_objectives": {
            name: float(best_objectives[k])
            for k, name in enumerate(LOSS_OBJECTIVES)
        },
        "compromise_theta": {
            name: float(best_theta[k])
            for k, name in enumerate(PARAM_NAMES)
        },
    }
    with open(os.path.join(NSGA2_DIR, "nsga2_log.json"), "w") as f:
        json.dump(log, f, indent=2)

    print(f"\nNSGA-II phase complete")
    print(f"  Pareto front points : {len(pareto_F)}")
    print(f"  Total evaluations   : {problem._eval_count}")
    print(f"  Results saved to    : {NSGA2_DIR}")

    return {
        "pareto_F": pareto_F,
        "pareto_X": pareto_X,
        "best_theta": best_theta,
        "best_objectives": best_objectives,
    }


if __name__ == "__main__":
    import sys
    _n_jobs = -1
    if len(sys.argv) > 1:
        _n_jobs = int(sys.argv[1])
    try:
        from calibration_utils import load_data
    except ModuleNotFoundError as exc:
        if exc.name != "calibration_utils":
            raise
        from .calibration_utils import load_data
    print("Loading empirical data...")
    data = load_data()
    run_nsga2(data, n_jobs=_n_jobs)