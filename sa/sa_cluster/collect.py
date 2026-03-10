"""
Assembles individual .npy result files, runs Sobol analysis,
saves CSVs and generates all plots — same outputs as the original sensitivity_analysis.py.
"""

import os
import sys
import numpy as np
from SALib.analyze import sobol

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from sensitivity_analysis import (
    RESULTS_DIR, OUTPUT_NAMES, PROBLEM_BASE, PROBLEM_PS,
    plot_single_model, plot_comparison, plot_ps_only,
    plot_output_distributions, save_indices_csv,
)

def collect_Y(model_name, n_runs):
    """
    Load individual .npy files and stack into Y (n_runs, n_outputs).
    Reports any missing tasks.
    """
    results_dir = os.path.join(RESULTS_DIR, model_name)
    Y = np.full((n_runs, len(OUTPUT_NAMES)), np.nan)

    missing = []
    for i in range(n_runs):
        path = os.path.join(results_dir, f"{i}.npy")
        if os.path.exists(path):
            Y[i] = np.load(path)
        else:
            missing.append(i)

    if missing:
        print(f"  WARNING: {len(missing)} missing tasks for {model_name}: {missing[:10]}"
              f"{'...' if len(missing) > 10 else ''}")
        print(f"  Imputing missing rows with column medians.")
        for col in range(Y.shape[1]):
            mask = np.isnan(Y[:, col])
            Y[mask, col] = np.nanmedian(Y[:, col])

    n_failed = np.isnan(Y).any(axis=1).sum()
    if n_failed:
        print(f"  WARNING: {n_failed} additional NaN rows imputed with medians.")
        for col in range(Y.shape[1]):
            mask = np.isnan(Y[:, col])
            Y[mask, col] = np.nanmedian(Y[:, col])

    return Y


def run_analysis(model_name, problem):
    param_path = os.path.join(RESULTS_DIR, f"{model_name}_param_values.npy")
    if not os.path.exists(param_path):
        print(f"ERROR: {param_path} not found. Did you run generate_samples.py?")
        sys.exit(1)

    param_values = np.load(param_path)
    n_runs = param_values.shape[0]
    print(f"{'='*60}")
    print(f"Collecting {model_name}  ({n_runs} runs)")
    print(f"{'='*60}")

    Y = collect_Y(model_name, n_runs)

    # Save assembled Y for reference
    np.save(os.path.join(RESULTS_DIR, f"{model_name}_Y.npy"), Y)

    # Sobol analysis
    si_dict = {}
    for k, out_name in enumerate(OUTPUT_NAMES):
        Si = sobol.analyze(problem, Y[:, k], calc_second_order=False, print_to_console=False)
        si_dict[out_name] = Si
        print(f"   [{out_name}]")
        print(f"    {'Param':<18}  S1 +/- conf      ST +/- conf")
        for j, pname in enumerate(problem["names"]):
            print(f"  {pname:<18}"
                  f"  {Si['S1'][j]:+.3f} +/- {Si['S1_conf'][j]:.3f}"
                  f"  {Si['ST'][j]:+.3f} +/- {Si['ST_conf'][j]:.3f}")

    return {"problem": problem, "Y": Y, "param_values": param_values, "Si": si_dict}


if __name__ == "__main__":
    result_base = run_analysis("BaseModel",         PROBLEM_BASE)
    result_ps   = run_analysis("ProductSpaceModel", PROBLEM_PS)

    plot_single_model(result_base, "BaseModel")
    save_indices_csv(result_base,  "BaseModel")

    plot_single_model(result_ps,   "ProductSpaceModel")
    save_indices_csv(result_ps,    "ProductSpaceModel")

    plot_comparison(result_base, result_ps)
    plot_ps_only(result_ps)
    plot_output_distributions(result_base, result_ps)

    print(f"All results saved to: {RESULTS_DIR}/")
    print("Done.")