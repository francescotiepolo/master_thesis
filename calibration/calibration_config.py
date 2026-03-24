"""
Calibration configuration for ProductSpaceModel.

Free parameters (from Sobol SA on calibration loss, SP=100, SC=19, N=1024;
                   S1/ST reported as rank_C/alpha):
  nu             — S1=≈0/0.25,  ST=0.16/0.48; dominant on alpha, moderate on rank_C via interactions
  beta_trade_off — S1=0.04/0.33, ST=0.35/0.60; dominant on both outputs
  sigma          — S1=≈0/0.02,  ST=0.25/0.11; moderate through interactions
  h_mean         — S1=≈0/0.03,  ST=0.18/0.14; moderate through interactions
  c_prime        — S1=0.05/≈0,  ST=0.25/0.05; meaningful on rank_C, weak on alpha; constrained c_prime <= c
  gamma          — S1=0.02/≈0,  ST=0.10/0.01; weak, mostly rank_C interactions
  s              — S1=0.02/0.01, ST=0.10/0.04; weak through interactions
  c              — S1=≈0/0.01,  ST=0.07/0.04; weak, mostly interaction with c_prime
  C_offdiag_mean — S1=0.19/≈0,  ST=0.68/0.13; dominant on rank_C, moderate on alpha via interactions
  C_diag_mean    — S1=≈0/≈0,   ST=0.23/0.17; purely through interactions on both outputs

Fixed (identifiable/numerically fixed):
  kappa — irrelevant when q=0
  G, q, mu, d_C
  data parameters: phi_space, r_P, r_C, h_P, h_C, C_PP, C_CC

To see the results of the SA on calibration loss, look in sa/sa_for_calibration/results/
"""

import os
import numpy as np

# Directories
RAW_DATA_DIR = "raw_data"
EXTRACTED_DIR = "extracted_data"
CALIB_DIR = "calibration_results"
HM_DIR = os.path.join(CALIB_DIR, "history_matching")
BO_DIR = os.path.join(CALIB_DIR, "bo_emulator")
MCMC_DIR = os.path.join(CALIB_DIR, "mcmc")
VAL_DIR = os.path.join(CALIB_DIR, "validation")

# Data years
YEAR_START = 1988
YEAR_REF = 2000
CALIB_END = 2010
YEAR_END = 2024

CALIB_YEARS = [yr for yr in range(YEAR_START, CALIB_END + 1, 2) if yr not in (1993, 1994)] # 1993/1994 have 0% data coverage
VALID_YEARS = list(range(CALIB_END, YEAR_END + 1, 2))

# Free parameters (10 selected by Sobol SA or mechanistic reasoning)
PARAM_NAMES = ["sigma", "nu", "beta_trade_off", "h_mean", "s", "c", "c_prime", "gamma", "C_diag_mean", "C_offdiag_mean", "entry_threshold"]
PARAM_BOUNDS = [
    (0.1, 2.0),    # sigma:          exponent in xi_i; 0.1=almost flat returns, 2.0=strong diminishing
    (0.0, 1.0),    # nu:             replicator (0) vs stabilisation (1); full interval
    (0.0, 1.0),    # beta_trade_off: specialist premium in beta; full interval
    (0.05, 1.0),   # h_mean:         Holling saturation; 0.05=near-linear mutualism, 1.0=strong saturation
    (0.0, 0.5),    # s:              spillover weight; 0.5 keeps spillovers sub-dominant
    (0.001, 2.0),  # c:              related-product competition multiplier on C_offdiag_mean
    (0.0, 2.0),    # c_prime:        unrelated-product competition; constrained by c_prime <= c
    (0.0, 5.0),    # gamma:          capability amplification; Cap_j median~0.73 so gamma=1 gives ~73% boost to rho_C
    (0.3, 2.0),    # C_diag_mean:    self-competition; ST~0.23/0.17 through interactions
    (0.001, 0.5),  # C_offdiag_mean: cross-competition baseline; dominant on rank_C (ST=0.68)
    (0.01, 10.0),  # entry_threshold: activation threshold for new-product entry signal
]
N_PARAMS = len(PARAM_NAMES)

# Fixed parameters
FIXED = {
    "G": 1.0, # Adaptation speed; normalisation
    "q": 0.0, # Congestion exponent; simplest assumption
    "mu": 1e-4, # Immigration floor; purely numerical
    "d_C": 0.0, # Country decay; absorbed into r_C
    "kappa": 0.0, # Proximity competition in xi; not relevant when q=0
    "enable_entry": True, # Product entry mechanism enabled during calibration
}

# Loss function
LOSS_WEIGHTS = {
    "rank_C": 1.0,
    "alpha": 1.0,
}
TIME_WEIGHT_SLOPE = 0.02

# Observation and model uncertainty

# Trade data has measurement error. sigma_obs represents irreducible noise in the calibration targets.
# Set to 0.05; should ideally be estimated from the data.
SIGMA_OBS = 0.05

# The model cannot perfectly reproduce trade dynamics. Following Kennedy & O'Hagan
# (2001), this term prevents over-fitting to simulator noise.
SIGMA_MODEL = 0.02

# History matching
HM_N_WAVES = 3
HM_SIMS_PER_WAVE = 200 # Number of simulations to run per wave
HM_SCREEN_N = 100_000 # Number of Sobol samples to evaluate GP on for NROY screening
HM_THRESHOLD = 3.0 # Pukelsheim 3-sigma rule

# Bayesian optimisation
BO_N_INIT = 10 * N_PARAMS
BO_N_ROUNDS = 15
BO_BATCH_SIZE = 16

# MCMC (emcee)
MCMC_N_WALKERS = max(2 * N_PARAMS + 2, 32)
MCMC_N_STEPS = 15_000
MCMC_BURNIN = 5000 # Number of initial steps to discard as burn-in
MCMC_THIN = 5 # Thinning factor to reduce autocorrelation in samples; keep every 5th sample after burn-in

# ODE solver
SIM_STEPS_PER_YEAR = 2000

# Validation
STABILITY_WINDOWS = [
    (1988, 2000),
    (2000, 2012),
    (2012, 2024)
]

N_PPC_DRAWS = 200 # Posterior predictive check draws
N_OOS_DRAWS = 200 # Out-of-sample draws
GP_SOBOL_SAMPLES = 8192 # Sobol samples on GP

SEED = 133