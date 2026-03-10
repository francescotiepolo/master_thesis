"""
Generates the Sobol sampling matrices for both models and saves them to disk.
"""

import os
import sys
import numpy as np
from SALib.sample import saltelli

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from sensitivity_analysis import (PROBLEM_BASE, PROBLEM_PS, N_SOBOL, RESULTS_DIR,)
print(f"SHARED_PARAMS names: {PROBLEM_BASE['names']}")
print(f"num_vars: {PROBLEM_BASE['num_vars']}")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(os.path.join(RESULTS_DIR, "BaseModel"), exist_ok=True)
os.makedirs(os.path.join(RESULTS_DIR, "ProductSpaceModel"), exist_ok=True)

for model_name, problem in [("BaseModel", PROBLEM_BASE), ("ProductSpaceModel", PROBLEM_PS)]:
    param_values = saltelli.sample(problem, N_SOBOL, calc_second_order=False)
    n_runs = param_values.shape[0]
    path = os.path.join(RESULTS_DIR, f"{model_name}_param_values.npy")
    np.save(path, param_values)
    print(f"{model_name}: {n_runs} runs  →  {path}")
    print(f"  SLURM --array=0-{n_runs - 1}")