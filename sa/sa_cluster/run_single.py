"""
Called by each SLURM array task.
Loads the appropriate parameter set, runs the model evaluation and saves the result.
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from sensitivity_analysis import (RESULTS_DIR, BASE_SEED, _evaluate_base, _evaluate_ps,)

EVALUATORS = {"BaseModel": _evaluate_base, "ProductSpaceModel": _evaluate_ps,}

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python run_single.py <model_name> <task_id> <chunk_size>")
        sys.exit(1)

    model_name = sys.argv[1]
    task_id = int(sys.argv[2])
    CHUNK_SIZE = int(sys.argv[3])

    if model_name not in EVALUATORS:
        print(f"Unknown model '{model_name}'. Choose: {list(EVALUATORS)}")
        sys.exit(1)

    param_path = os.path.join(RESULTS_DIR, f"{model_name}_param_values.npy")
    param_values = np.load(param_path)
    results_dir = os.path.join(RESULTS_DIR, model_name)

    start = task_id * CHUNK_SIZE
    end = min(start + CHUNK_SIZE, len(param_values))
    for i in range(start, end):
        result = EVALUATORS[model_name](param_values[i], BASE_SEED + i)
        np.save(os.path.join(results_dir, f"{i}.npy"), result)