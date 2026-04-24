"""
Phase 1: Maximin Latin hypercube initial design.

Saves:
  initial_design/theta_all.npy    -- all sampled parameter vectors
  initial_design/loss_all.npy     -- corresponding losses
  initial_design/nroy_theta.npy   -- finite points inside NROY bounds
  initial_design/nroy_loss.npy    -- matching losses
  initial_design/nroy_bounds.json -- NROY bounding box for DE
"""

import os
import json
import numpy as np
from scipy.stats.qmc import LatinHypercube

try:
    from calibration_config import (
        PARAM_NAMES,
        PARAM_BOUNDS,
        N_PARAMS,
        LHS_N_SAMPLES,
        LHS_NROY_ELITE_FRACTION,
        LHS_NROY_PADDING,
        LHS_CHUNK_SIZE,
        LHS_DIR,
        SEED,
    )
    from calibration_utils import PENALTY, evaluate_batch
except ModuleNotFoundError as exc:
    if exc.name not in {"calibration_config", "calibration_utils"}:
        raise
    from .calibration_config import (
        PARAM_NAMES,
        PARAM_BOUNDS,
        N_PARAMS,
        LHS_N_SAMPLES,
        LHS_NROY_ELITE_FRACTION,
        LHS_NROY_PADDING,
        LHS_CHUNK_SIZE,
        LHS_DIR,
        SEED,
    )
    from .calibration_utils import PENALTY, evaluate_batch

N_JOBS = -1


def nroy_bounding_box(points, padding=LHS_NROY_PADDING):
    """
    Bounding box of points with padding, clipped to prior bounds.
    """
    lo = points.min(axis=0)
    hi = points.max(axis=0)
    width = hi - lo
    lo = lo - padding * width
    hi = hi + padding * width
    for i, (p0, p1) in enumerate(PARAM_BOUNDS):
        lo[i] = max(lo[i], p0)
        hi[i] = min(hi[i], p1)
    return [(float(lo[i]), float(hi[i])) for i in range(N_PARAMS)]


def _lhs_maximin(bounds, n, seed):
    """
    Center-discrepancy-optimized LHS.
    """
    sampler = LatinHypercube(d=len(bounds), seed=seed, optimization="random-cd")
    unit = sampler.random(n=n)
    lo = np.array([b[0] for b in bounds], dtype=float)
    hi = np.array([b[1] for b in bounds], dtype=float)
    return lo + unit * (hi - lo)


def run_initial_design(data):
    """
    1. Generate LHS samples
    2. Evaluate simulator runs in parallel
    3. Identify NROY region and save results
    """
    os.makedirs(LHS_DIR, exist_ok=True)

    print(f"\nGenerating center-discrepancy-optimized LHS design with {LHS_N_SAMPLES} points...")
    theta_all = _lhs_maximin(PARAM_BOUNDS, LHS_N_SAMPLES, SEED)

    print(f"Evaluating {LHS_N_SAMPLES} simulator runs...")
    print(f"  Workers    : {N_JOBS}")
    print(f"  Chunk size : {LHS_CHUNK_SIZE}")
    loss_all = evaluate_batch(
        theta_all,
        data,
        n_jobs=N_JOBS,
        chunk_size=LHS_CHUNK_SIZE,
    )

    finite = loss_all < PENALTY * 0.1
    n_finite = int(finite.sum())
    n_failed = int((~finite).sum())
    fail_rate = n_failed / len(loss_all) if len(loss_all) else 0.0

    if n_finite == 0:
        raise RuntimeError(
            "Initial design produced zero finite evaluations (all runs hit penalty)."
        )

    finite_loss = loss_all[finite]
    print(
        "Loss diagnostics (finite only): "
        f"min={finite_loss.min():.4f}  "
        f"p25={np.percentile(finite_loss, 25):.4f}  "
        f"median={np.median(finite_loss):.4f}  "
        f"p75={np.percentile(finite_loss, 75):.4f}"
    )
    print(f"Failure rate: {n_failed}/{len(loss_all)} ({fail_rate:.1%})")

    n_best = max(1, int(np.ceil(LHS_NROY_ELITE_FRACTION * n_finite)))
    best_idx_local = np.argsort(finite_loss)[:n_best]
    elite = theta_all[finite][best_idx_local]
    bounds = nroy_bounding_box(elite, padding=LHS_NROY_PADDING)

    lo = np.array([b[0] for b in bounds], dtype=float)
    hi = np.array([b[1] for b in bounds], dtype=float)
    inside = np.all((theta_all >= lo) & (theta_all <= hi), axis=1)
    nroy_mask = finite & inside
    nroy_theta = theta_all[nroy_mask]
    nroy_loss = loss_all[nroy_mask]

    np.save(os.path.join(LHS_DIR, "theta_all.npy"), theta_all)
    np.save(os.path.join(LHS_DIR, "loss_all.npy"), loss_all)
    np.save(os.path.join(LHS_DIR, "nroy_theta.npy"), nroy_theta)
    np.save(os.path.join(LHS_DIR, "nroy_loss.npy"), nroy_loss)
    with open(os.path.join(LHS_DIR, "nroy_bounds.json"), "w") as f:
        json.dump({"params": PARAM_NAMES, "bounds": bounds}, f, indent=2)

    best_idx = np.argmin(finite_loss)
    best_theta = theta_all[finite][best_idx]
    best_loss = finite_loss[best_idx]

    print("\nInitial design complete")
    print(f"  Total runs        : {len(theta_all)}")
    print(f"  Finite runs       : {n_finite}")
    print(f"  Elite fraction    : {LHS_NROY_ELITE_FRACTION:.0%}")
    print(f"  Box padding       : {LHS_NROY_PADDING:.0%}")
    print(f"  NROY finite points: {len(nroy_theta)}")
    print(f"  Best loss         : {best_loss:.5f}")
    for name, val in zip(PARAM_NAMES, best_theta):
        print(f"  {name:20s} = {val:.5f}")

    return {
        "theta_all": theta_all,
        "loss_all": loss_all,
        "nroy_theta": nroy_theta,
        "nroy_loss": nroy_loss,
        "nroy_bounds": bounds,
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
    run_initial_design(data)