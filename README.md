# Tiepolo Francesco Master's Thesis: The Dynamics of Comparative Advantage

This repository contains the model, calibration pipeline and experiments for a master's thesis that treats the world economy as a mutualistic network of countries and products. The model is adapted from the adaptive plant-pollinator framework of Terpstra et al. (2024): countries play the role of pollinators, products play the role of plants and a country's choice of what to specialise in plays the role of a pollinator's adaptive foraging. It is calibrated against UN Comtrade export data (from the Atlas of Economic Complexity) covering 19 G20 countries and the 100 largest four-digit HS92 product categories over 1988 to 2024.

## What the model does

Countries and products evolve on a bipartite network. Each country spreads its productive effort across products, and that allocation shifts over time according to a replicator-style equation that rewards products giving good returns. Products, in turn, grow in proportion to the effort countries direct at them. The benefit each side gets from the other saturates (a Holling type-II response), so diversification has diminishing returns and effort on a crowded product is worth less than effort on an underexploited one.

`ProductSpaceModel` builds on this core with the structure of the product space itself. Goods that are close in the product space share capabilities, so a country benefits from knowledge spillovers between related products, accumulates capability by specialising in dense regions and faces stronger competition from neighbouring products than from distant ones. The empirical product-space proximities drive all of these effects.

The aim is to reproduce historical export trajectories and to use the calibrated model as a platform for counterfactual experiments: if a country reallocated its effort toward a different basket of products, how would its exports respond?

## Repository layout

```
.
├── base_model.py             Core ODE system (BaseModel) and critical-point / hysteresis analysis
├── product_space_model.py    ProductSpaceModel: adds product-space proximity effects
├── ode_solver.py             Custom RK45/LSODA solver with equilibrium and extinction detection
├── plot_analysis.py          Plots of network structure and forward/backward dynamics
├── sensitivity_analysis.py   Sobol sensitivity analysis on collapse, recovery and hysteresis
├── requirements.txt
│
├── calibration/
│   ├── calibration_config.py        All fixed parameters, bounds, loss weights, country groups
│   ├── calibration_utils.py         Shared data loading, model construction and loss functions
│   ├── data_extraction_all_years.py Turns the raw Comtrade CSVs into annual numpy arrays
│   ├── growth_regression.py         Estimates country growth rates from the panel
│   ├── product_growth_regression.py Estimates product growth rates from the panel
│   │
│   ├── run_pipeline.py              Driver for the original joint calibration (LHS, NSGA-II, validation)
│   ├── initial_design.py            Latin hypercube and NROY screening
│   ├── nsga2_optimizer.py           Multi-objective NSGA-II search over the shared parameter vector
│   ├── validation.py                Evaluates the chosen vector and compares trajectories
│   ├── calibration.slurm            Snellius job script for the joint pipeline
│   │
│   ├── country_wise_calibration/    The main calibration pipeline (see below)
│   ├── extracted_data/              Committed numpy arrays, enough to run without the raw data
│   ├── raw_data/                    Raw Comtrade CSVs (not committed; too large)
│   └── calibration_results/         Calibrated parameters and diagnostic plots
│
├── experiments/
│   ├── alpha_policy/                Counterfactual specialisation experiments (alpha injection)
│   └── country_wise_free_sim/       Trajectory plots and the specialisation-fit diagnostic
│
└── sa/
    ├── sa_cluster/                  Parallel Sobol run for the cluster (sample, run, collect)
    ├── sa_for_calibration/          Sobol analysis of the calibration loss
    ├── submit.sh, sa_cp.slurm       Snellius submission scripts
    └── sa_results_*/                Stored sensitivity-analysis outputs

```

## How the calibration works

The model is fitted in two stages, reflecting how the project developed.

The first stage is a **joint calibration**, in which all 19 countries evolve under a single shared parameter vector. It starts with a Latin hypercube survey of the parameter space, narrows the search to a not-ruled-out-yet region, and then runs a multi-objective NSGA-II search that balances the components of the loss (country output level, trajectory shape, product ranking, and allocation fit). This stage lives in `calibration/` and is driven by `run_pipeline.py`.

A single shared vector turns out to be too rigid to fit every country well, which motivates the **country-wise calibration** in `calibration/country_wise_calibration/`. Here the model is fitted one country at a time: only the target country evolves freely, while the other 18 are pinned to their observed trajectories at each year boundary. This splits the problem into 19 independent sub-problems, each solved with a Latin hypercube survey followed by differential evolution. There are two modes. In `conditioned` mode the target country's specialisation also evolves; in `alpha_frozen` mode every country's specialisation is held to the observed values, so only the target's export level is free. The `alpha_frozen` results are the ones the experiments build on.

The choice of which parameters to leave free and which to fix is informed by the sensitivity analysis in `sa/`, which decomposes the variance of the calibration loss across parameters and shows that a few parameters cannot be identified from this data.

## The experiments

Once a country is calibrated, `experiments/alpha_policy/` asks: starting from the observed allocation, what happens if a country shifts a fraction of its effort toward a chosen basket of products? The target country's specialisation is set externally each year rather than left to the model, so a baseline run with zero shift reproduces the calibrated trajectory exactly and every counterfactual is a clean departure from it. The experiment sweeps different strategies (reinforce existing strengths, move into product-space neighbours, target high-complexity products outside the current basket) across a grid of shift strengths and basket sizes.

`experiments/country_wise_free_sim/` produces the trajectory comparisons and a per-product diagnostic that measures how closely the model's specialisation tracks the data.

## Data

The raw Comtrade CSVs in `calibration/raw_data/` are not committed because of their size. The arrays in `calibration/extracted_data/` are committed and are enough to run the calibration and experiments without the raw files:

- `phi_space.npy` is the 100 by 100 product-space proximity matrix.
- `beta_C_ref.npy` is a continuous 19 by 100 matrix of capability strength in `[0, 1]`, built from how often and how strongly each country has exported each product across the panel. A value of zero means the country never exported that product in the selected years.
- `alpha_init.npy`, `P_init.npy`, and `C_init.npy` are the 1988 initial conditions. `alpha_init` is the observed export-share allocation.
- `r_P.npy` and `r_C.npy` are the regression-based growth rates.
- `annual/alpha_{year}.npy`, `C_{year}.npy` and `P_{year}.npy` are the yearly snapshots from 1988 to 2024.

## Running the code

Install the dependencies with `pip install -r requirements.txt`.

Some of the important steps:

```bash
# Sensitivity analysis on the model's dynamics (local)
python sensitivity_analysis.py

# Country-wise calibration on the cluster (alpha_frozen mode by default)
sbatch calibration/country_wise_calibration/calibration_country_wise.slurm

# Trajectory plots for all 19 countries
python experiments/country_wise_free_sim/run_free_sim.py

# Counterfactual specialisation experiment for one country
python experiments/alpha_policy/run_alpha_injection.py --country BRA --n-jobs 8
```

The heavier calibration and sensitivity runs are designed for the Snellius supercomputer and are submitted through the SLURM scripts in each directory.
