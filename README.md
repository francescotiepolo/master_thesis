# master_thesis

Master's thesis codebase: a product-country mutualistic network model for international export dynamics, calibrated against UN Comtrade data (1988–2024). The model adapts the ecological framework of Terpstra et al. (2024) (adaptive plant-pollinator networks) to international trade, with the analogy: **pollinators → countries**, **plants → products**, **adaptive foraging → adaptive specialisation**.

## Overview

The core idea is to model the co-evolution of countries and products as a bipartite mutualistic network. Countries allocate productive effort across products via a replicator-style ODE; products grow in proportion to the effort countries invest. The `ProductSpaceModel` extends this with product-space proximity effects: knowledge spillovers, capability accumulation and proximity-weighted competition.

The pipeline calibrates the model's free parameters against historical trade trajectories (G20 countries, top 200 HS92 products) and validates out-of-sample on 2010–2024 data.

## Repository Structure

```
.
├── base_model.py               # BaseModel: core ODE system
├── product_space_model.py      # ProductSpaceModel: extends BaseModel with product-space effects
├── ode_solver.py               # Custom RK45/LSODA solver with equilibrium detection
├── plot_analysis.py            # Visualisation of network structure and dynamics
├── sensitivity_analysis.py     # Run sensitivity analysis locally from the root directory
├── requirements.txt
│
├── calibration/
│   ├── calibration_config.py       # Single source for calibration configuration: all parameters, bounds, loss weights
│   ├── calibration_utils.py        # Shared utilities: data loading, model construction, loss function
│   ├── run_pipeline.py             # Runs phases 0–3 via RUN_PHASE_* flags
│   ├── data_extraction_all_years.py # Phase 0: Comtrade CSV → annual numpy arrays
│   ├── initial_design.py           # Phase 1: space-filling initial design + NROY screening
│   ├── de_optimizer.py             # Phase 2a: multi-restart differential evolution
│   ├── nsga2_optimizer.py          # Phase 2b: NSGA-II multi-objective Pareto optimisation
│   ├── validation.py               # Phase 3: best-point evaluation and trajectory comparison
│   ├── growth_regression.py        # OLS estimation of piecewise growth rates r_P, r_C and competition matrices
│   ├── calibration_country_wise.py # Country-wise calibration variant
│   ├── test_pipeline.py            # Tests for the pipeline
│   ├── calibration.slurm           # Snellius HPC job script (full pipeline)
│   ├── calibration_country_wise.slurm
│   ├── extracted_data/             # Extracted numpy arrays (initial conditions, phi_space, annual snapshots)
│   ├── raw_data/                   # Gitignored: raw Comtrade CSV (~498 MB)
│   └── calibration_results/        # Optimizer outputs
│
├── sa/
│   ├── sensitivity_analysis_cp.py  # Sobol SA focused on critical-point outputs
│   ├── submit.sh / submit_cp_only.slurm / sa_cp.slurm  # HPC submission scripts
│   ├── sa_cluster/                 # Parallelised cluster SA: generate samples → run_single → collect
│   ├── sa_for_calibration/         # SA on the calibration loss function (parameter selection)
│   │   ├── sa_calibration_loss.py  
│   │   ├── parameter_selection.py  # Ranks parameters by Sobol importance for calibration
│   │   └── results/                # Sobol indices, parameter selection scores, recommendations
│   ├── sa_results_cluster/         # SA results (full parameter set, cluster run)
│   ├── sa_results_q_fix/           # SA results with q fixed
│   └── sa_cp_results_cluster/      # SA results on critical-point outputs
│
└── Figures/                        # Model dynamics and network structure plots
```

## Data

Raw Comtrade data (`calibration/raw_data/`) is gitignored (too large). Preprocessed arrays in `calibration/extracted_data/` are committed and sufficient to run phases 1–3 directly:

- `phi_space.npy` — product-space proximity matrix
- `beta_C_ref.npy` — binary RCA matrix at reference year 2000
- `alpha_init.npy`, `P_init.npy`, `C_init.npy` — initial conditions (1988)
- `r_P.npy`, `r_C.npy` — OLS trend growth rates
- `annual/alpha_{year}.npy`, `C_{year}.npy`, `P_{year}.npy` — annual snapshots 1988–2024