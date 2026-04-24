"""
Phase 2a: Multi-restart differential evolution.
"""

import os
import json
import time
import multiprocessing as mp
import numpy as np

from scipy.optimize import differential_evolution, minimize

try:
    from calibration_config import (
        PARAM_NAMES, N_PARAMS,
        DE_POPSIZE, DE_MAXITER, DE_TOL, DE_STALL_LIMIT,
        DE_LOCAL_SAMPLES, DE_LOCAL_SCALE, DE_MUTATION, DE_N_RESTARTS,
        DE_START_RESTART, TRAJECTORY_TIMEOUT_S,
        LHS_DIR, DE_DIR, SEED,
    )
    from calibration_utils import (
        trajectory_loss, evaluate_batch, load_nroy_bounds, PENALTY,
    )
except ModuleNotFoundError as exc:
    if exc.name not in {"calibration_config", "calibration_utils"}:
        raise
    from .calibration_config import (
        PARAM_NAMES, N_PARAMS,
        DE_POPSIZE, DE_MAXITER, DE_TOL, DE_STALL_LIMIT,
        DE_LOCAL_SAMPLES, DE_LOCAL_SCALE, DE_MUTATION, DE_N_RESTARTS,
        DE_START_RESTART, TRAJECTORY_TIMEOUT_S,
        LHS_DIR, DE_DIR, SEED,
    )
    from .calibration_utils import (
        trajectory_loss, evaluate_batch, load_nroy_bounds, PENALTY,
    )

N_JOBS = -1


def _theta_key(theta):
    """
    Convert theta to a key for timed_map results tracking.
    """
    return tuple(np.round(np.asarray(theta, dtype=np.float64), 12))


def _timed_eval_worker(func, theta, conn):
    """
    Run one objective evaluation in its own process.
    """
    os.environ["_CALIB_SUBPROCESS"] = "1"
    try:
        conn.send(("ok", float(func(theta))))
    except Exception as exc:
        conn.send(("err", repr(exc)))
    finally:
        conn.close()


class _TimedProcessMap:
    """
    Callable for SciPy DE that enforces per-candidate
    timeouts by isolating each evaluation in its own subprocess.
    """

    def __init__(self, max_workers, timeout_s, penalty):
        self.max_workers = max_workers
        self.timeout_s = timeout_s
        self.penalty = penalty
        methods = mp.get_all_start_methods()
        method = "fork" if "fork" in methods else methods[0]
        self.ctx = mp.get_context(method)
        self.last_results = {}

    def __call__(self, func, iterable):
        """
        Map func over iterable with per-call timeouts.
        """
        thetas = [np.asarray(theta, dtype=np.float64).copy() for theta in iterable]
        if not thetas:
            return []

        results = [self.penalty] * len(thetas)
        active = {}
        next_idx = 0
        completed = 0

        while completed < len(thetas):
            while next_idx < len(thetas) and len(active) < self.max_workers:
                parent_conn, child_conn = self.ctx.Pipe(duplex=False)
                proc = self.ctx.Process(
                    target=_timed_eval_worker,
                    args=(func, thetas[next_idx], child_conn),
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
                    status, payload = conn.recv()
                    conn.close()
                    proc.join(timeout=0.1)
                    active.pop(idx)
                    completed += 1
                    progress = True
                    if status == "ok":
                        results[idx] = payload
                    else:
                        print(f"    [eval-error] candidate={idx} err={payload}")
                    continue

                if now - state["start"] > self.timeout_s:
                    print(f"    [eval-timeout] candidate={idx} limit={self.timeout_s}s")
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
                    print(f"    [eval-error] candidate={idx} worker exited without result")

            if not progress:
                time.sleep(0.05)

        self.last_results = {
            _theta_key(theta): value for theta, value in zip(thetas, results)
        }
        return results


class _LoggedLoss:
    """
    Wrapper around trajectory_loss that logs every evaluation to a
    per-process file. This allows us to recover ALL evaluations after DE completes,
    even if the process is killed mid-evaluation.
    """
    def __init__(self, log_dir, n_params):
        self.log_dir = log_dir
        self.n_params = n_params
        os.makedirs(log_dir, exist_ok=True)

    def __call__(self, theta, data):
        """
        Evaluate trajectory_loss and log the result with the parameter vector.
        """
        val = trajectory_loss(theta, data)
        record = np.empty(self.n_params + 1, dtype=np.float64)
        record[:self.n_params] = theta
        record[-1] = val
        log_file = os.path.join(self.log_dir, f"eval_{os.getpid()}.bin")
        with open(log_file, "ab") as f:
            record.tofile(f)
        return val


def _read_eval_logs(log_dir, n_params):
    """
    Read all logged evaluations from per-process files.
    """
    all_records = []
    if not os.path.isdir(log_dir):
        return np.empty((0, n_params)), np.empty(0)
    for fname in os.listdir(log_dir):
        if fname.startswith("eval_") and fname.endswith(".bin"):
            raw = np.fromfile(os.path.join(log_dir, fname), dtype=np.float64)
            if len(raw) >= n_params + 1:
                records = raw.reshape(-1, n_params + 1)
                all_records.append(records)
    if not all_records:
        return np.empty((0, n_params)), np.empty(0)
    combined = np.vstack(all_records)
    return combined[:, :n_params], combined[:, -1]


def run_de(data, nroy_bounds=None):
    """
    1. Multi-restart differential evolution on the simulator directly
    2. Collect ALL evaluations via per-process logging
    3. Local sampling around overall best
    """
    os.makedirs(DE_DIR, exist_ok=True)

    if nroy_bounds is None:
        nroy_bounds = load_nroy_bounds()

    bounds = [(lo, hi) for lo, hi in nroy_bounds]
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])

    # Load initial-design evaluations 
    lhs_theta_path = os.path.join(LHS_DIR, "nroy_theta.npy")
    lhs_loss_path = os.path.join(LHS_DIR, "nroy_loss.npy")
    missing = [p for p in (lhs_theta_path, lhs_loss_path) if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            "Initial design results not found. Run `python calibration/run_pipeline.py lhs` "
            "before DE. Missing: " + ", ".join(missing)
        )
    lhs_theta = np.load(lhs_theta_path)
    lhs_loss = np.load(lhs_loss_path)
    print(f"{len(lhs_theta)} finite initial-design points available.")

    n_pop = DE_POPSIZE * N_PARAMS
    n_workers = N_JOBS if N_JOBS > 0 else 8

    best_overall_theta = None
    best_overall_loss = np.inf
    all_de_logs = {}

    # Recover best from previously completed restarts
    for restart in range(DE_START_RESTART):
        log_dir = os.path.join(DE_DIR, f"eval_logs_restart_{restart}")
        prev_theta, prev_loss = _read_eval_logs(log_dir, N_PARAMS)
        if len(prev_theta) > 0:
            best_idx = np.argmin(prev_loss)
            if prev_loss[best_idx] < best_overall_loss:
                best_overall_loss = prev_loss[best_idx]
                best_overall_theta = prev_theta[best_idx].copy()
            print(f"  Restart {restart+1}: recovered {len(prev_theta)} evals "
                  f"(best={prev_loss[best_idx]:.5f})")
        else:
            print(f"  Restart {restart+1}: no previous logs found")

    # Multi-restart DE
    for restart in range(DE_START_RESTART, DE_N_RESTARTS):
        seed = SEED + restart * 1000
        rng = np.random.default_rng(seed)
        print(f"\n{'='*60}")
        print(f"DE restart {restart+1}/{DE_N_RESTARTS} (seed={seed})")
        print(f"{'='*60}")

        # Mixed population init: 50% best LHS, 50% random NROY
        if len(lhs_theta) >= n_pop:
            n_best = n_pop // 2
            best_idx = np.argsort(lhs_loss)[:n_best]
            rand_idx = rng.choice(len(lhs_theta), n_pop - n_best, replace=False)
            init_pop = np.vstack([lhs_theta[best_idx], lhs_theta[rand_idx]])
            print(f"  Population: {n_best} best LHS + {n_pop - n_best} random NROY = {n_pop}")
        else:
            init_pop = 'sobol'
            print("  Using Sobol initialisation (not enough initial-design points)")

        # Per-process evaluation logging
        log_dir = os.path.join(DE_DIR, f"eval_logs_restart_{restart}")
        obj_fn = _LoggedLoss(log_dir, N_PARAMS)
        timed_map = _TimedProcessMap(n_workers, TRAJECTORY_TIMEOUT_S, PENALTY)

        # Progress tracking
        de_log = []
        _gen = [0]
        _t0 = [time.time()]
        _best = [np.inf]
        _best_gen = [0]
        _max_time_s = 20 * 3600  # 20h safety valve

        def callback(xk, convergence):
            """
            DE callback to log progress and implement stopping criterion.
            """
            _gen[0] += 1
            elapsed = time.time() - _t0[0]
            loss = timed_map.last_results.get(_theta_key(xk))
            if loss is not None and loss < _best[0]:
                _best[0] = loss
                _best_gen[0] = _gen[0]
            entry = {
                "generation": _gen[0],
                "best_loss": float(_best[0]),
                "convergence": float(convergence),
                "time_s": float(elapsed),
            }
            print(f"  Gen {_gen[0]:4d} | best={_best[0]:.5f} | {elapsed:.0f}s")
            de_log.append(entry)
            if _gen[0] - _best_gen[0] >= DE_STALL_LIMIT:
                print(f"  Stopping: no improvement for {DE_STALL_LIMIT} generations")
                return True
            if elapsed > _max_time_s:
                print(f"  Stopping: time limit ({_max_time_s/3600:.0f}h) reached")
                return True

        print(f"\n  Population : {n_pop}")
        print(f"  Max iter   : {DE_MAXITER}")
        print(f"  Workers    : {n_workers}")
        print(f"  Tolerance  : {DE_TOL}")
        print(f"  Mutation   : {DE_MUTATION}")
        print(f"  Stall limit: {DE_STALL_LIMIT} gens")

        result = differential_evolution(
            obj_fn,
            bounds=bounds,
            args=(data,),
            init=init_pop,
            maxiter=DE_MAXITER,
            popsize=DE_POPSIZE,
            tol=DE_TOL,
            atol=0,
            mutation=DE_MUTATION,
            seed=seed,
            workers=timed_map,
            updating='deferred',
            polish=False,
            disp=False,
            callback=callback,
        )

        all_de_logs[str(restart)] = de_log

        # Manual polish
        print(f"  Polishing with L-BFGS-B...")
        polish_res = minimize(
            lambda x: obj_fn(x, data),
            result.x,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 50, 'maxfun': 200},
        )
        if polish_res.fun < result.fun:
            print(f"    Polish improved: {result.fun:.5f} -> {polish_res.fun:.5f}")
            result.x = polish_res.x
            result.fun = polish_res.fun
        else:
            print(f"    Polish did not improve ({polish_res.fun:.5f} >= {result.fun:.5f})")

        print(f"\n  Restart {restart+1} complete: {result.message}")
        print(f"    Function evaluations: {result.nfev}")
        print(f"    Best loss           : {result.fun:.5f}")
        for name, val in zip(PARAM_NAMES, result.x):
            print(f"    {name:20s} = {val:.5f}")

        if result.fun < best_overall_loss:
            best_overall_loss = result.fun
            best_overall_theta = result.x.copy()

    print(f"\n{'='*60}")
    print(f"Best across {DE_N_RESTARTS} restarts: {best_overall_loss:.5f}")
    for name, val in zip(PARAM_NAMES, best_overall_theta):
        print(f"  {name:20s} = {val:.5f}")

    # Collect ALL evaluations: LHS + DE logged + best
    theta_parts = [lhs_theta]
    loss_parts = [lhs_loss]

    for restart in range(DE_N_RESTARTS):
        log_dir = os.path.join(DE_DIR, f"eval_logs_restart_{restart}")
        de_theta, de_loss = _read_eval_logs(log_dir, N_PARAMS)
        if len(de_theta) > 0:
            theta_parts.append(de_theta)
            loss_parts.append(de_loss)
            print(f"  Restart {restart+1}: {len(de_theta)} logged evaluations recovered")

    theta_parts.append(best_overall_theta[np.newaxis])
    loss_parts.append(np.array([best_overall_loss]))

    theta_all = np.vstack(theta_parts)
    loss_all = np.concatenate(loss_parts)
    print(f"\nTotal evaluations collected: {len(theta_all)}")

    # Local sampling around optimum
    print(f"\nLocal sampling: {DE_LOCAL_SAMPLES} points around optimum "
          f"(scale={DE_LOCAL_SCALE:.0%} of range)...")
    rng = np.random.default_rng(SEED + 42)
    scale = DE_LOCAL_SCALE * (hi - lo)
    local_theta = np.clip(
        best_overall_theta + rng.normal(0, 1, (DE_LOCAL_SAMPLES, N_PARAMS)) * scale,
        lo, hi,
    )
    local_loss = evaluate_batch(local_theta, data, n_jobs=n_workers)
    n_finite_local = (local_loss < PENALTY * 0.1).sum()
    print(f"  {n_finite_local}/{DE_LOCAL_SAMPLES} finite evaluations")

    theta_all = np.vstack([theta_all, local_theta])
    loss_all = np.concatenate([loss_all, local_loss])

    # Save
    np.save(os.path.join(DE_DIR, "theta_all.npy"), theta_all)
    np.save(os.path.join(DE_DIR, "loss_all.npy"), loss_all)
    np.save(os.path.join(DE_DIR, "best_theta.npy"), best_overall_theta)
    np.save(os.path.join(DE_DIR, "nroy_bounds.npy"),
            np.array([[b[0], b[1]] for b in nroy_bounds]))
    with open(os.path.join(DE_DIR, "de_log.json"), "w") as f:
        json.dump(all_de_logs, f, indent=2)

    print(f"\nDE phase complete")
    print(f"  Total evaluated points: {len(theta_all)}")
    print(f"  Best loss             : {best_overall_loss:.5f}")

    return {
        "theta_all": theta_all,
        "loss_all": loss_all,
        "best_theta": best_overall_theta,
        "best_loss": best_overall_loss,
    }


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        N_JOBS = int(sys.argv[1])
    try:
        from calibration_utils import load_data
    except ModuleNotFoundError as exc:
        if exc.name != "calibration_utils":
            raise
        from .calibration_utils import load_data
    print("Loading empirical data...")
    data = load_data()
    run_de(data)