"""
Calibration configuration for ProductSpaceModel.
"""

import os

# Directories
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_DIR = os.path.join(_SCRIPT_DIR, "raw_data")
EXTRACTED_DIR = os.path.join(_SCRIPT_DIR, "extracted_data")
CALIB_DIR = os.path.join(_SCRIPT_DIR, "calibration_results")
LHS_DIR = os.path.join(CALIB_DIR, "initial_design")
DE_DIR = os.path.join(CALIB_DIR, "de_optimizer")
NSGA2_DIR = os.path.join(CALIB_DIR, "nsga2_optimizer")
VAL_DIR = os.path.join(CALIB_DIR, "validation")

# Data years
YEAR_START = 1988
YEAR_REF = 2000
CALIB_END = 2010
YEAR_END = 2024

CALIB_YEARS = [yr for yr in range(YEAR_START, CALIB_END + 1) if yr not in (1993, 1994)] # 1993/1994 have 0% data coverage
VALID_YEARS = list(range(CALIB_END, YEAR_END + 1))

# Free parameters
PARAM_NAMES = [
    "C_offdiag_mean", "beta_trade_off",
    "sigma", "h_mean", "C_diag_mean",
    "kappa",
    "nu", "G",
    "r_C_declining", "r_C_rising", "r_C_stable",
    "entry_threshold",
]
PARAM_BOUNDS = [
    (0.001, 1.0),  # C_offdiag_mean
    (0.0, 1.0),    # beta_trade_off: specialist premium in beta
    (0.01, 4.0),   # sigma
    (0.05, 5.0),   # h_mean: Holling saturation
    (0.3, 2.0),    # C_diag_mean: self-competition
    (0.0, 0.05),   # kappa: proximity competition in xi
    (0.0, 1.0),    # nu: replicator (0) vs stabilisation (1)
    (0.01, 2.0),   # G: adaptation speed
    (-1.0, 0.10),  # r_C_declining: net intrinsic growth for DEU, FRA, GBR, ITA, JPN, USA, CAN
    (-1.0, 0.20),  # r_C_rising: net intrinsic growth for CHN, IND, KOR, MEX, RUS
    (-1.0, 0.15),  # r_C_stable: net intrinsic growth for ARG, AUS, BRA, IDN, SAU, TUR, ZAF
    (0.01, 50.0),  # entry_threshold: activation threshold for product entry
]

# Country groups for group-level intrinsic growth rates r_C
# Indices refer to position in countries_index.csv
COUNTRY_GROUPS = {
    "declining": [3, 5, 6, 7, 10, 11, 17],  # CAN, DEU, FRA, GBR, ITA, JPN, USA
    "rising":    [4, 9, 12, 13, 14],        # CHN, IND, KOR, MEX, RUS
    "stable":    [0, 1, 2, 8, 15, 16, 18],  # ARG, AUS, BRA, IDN, SAU, TUR, ZAF
}
N_PARAMS = len(PARAM_NAMES)

# Fixed parameters
FIXED = {
    "q": 1.0,          # Congestion exponent; simplest assumption
    "mu": 1e-4,        # Immigration; purely numerical
    "d_C": 0.0,        # Not needed with free r_C groups: r_C - d_C absorbed into r_C
    "enable_entry": True,
    "c": 1.0,          # Related-product competition multiplier
    "c_prime": 1.0,    # Fixed: optimizer converged to c_prime ≈ c; competition is uniform across products
    "s": 0.0,          # Fixed: knowledge spillovers negligible (optimizer converged to ~0)
    "gamma": 1.0,      # Capability amplification
}

# Loss function
LOSS_WEIGHTS = {
    "nrmse_C": 0.25,        # normalised RMSE on log country market shares
    "traj_corr_C": 0.30,    # per-country trajectory tracking (1 - mean Spearman over time)
    "rank_products": 0.25,  # product allocation ranking within each country
    "nrmse_P": 0.20,        # normalised RMSE on log product market shares
}
TIME_WEIGHT_SLOPE = 0.02

# Windowed simulation: re-initialise from observed data at these years to prevent error accumulation
WINDOW_REINIT_YEARS = [1991, 1995, 1998, 2001, 2004, 2007]  # every ~3 years

# Growth rate periods 
GROWTH_RATE_PERIODS = [
    (1988, 2000),
    (2000, 2010),
    (2010, 2024),
]

# Initial design
LHS_N_SAMPLES = 15000
LHS_NROY_ELITE_FRACTION = 0.40 
LHS_NROY_PADDING = 0.15
LHS_CHUNK_SIZE = int(os.environ.get("LHS_CHUNK_SIZE", "1000"))

# Differential evolution
DE_POPSIZE = 16 # population = DE_POPSIZE * N_PARAMS = 176
DE_MAXITER = 100_000
DE_TOL = 0.0 # disable scipy's convergence stop
DE_STALL_LIMIT = 100 # stop after this many generations without improvement
DE_LOCAL_SAMPLES = 600 # local evaluations around optimum
DE_LOCAL_SCALE = 0.12 # neighbourhood radius for local refinement
DE_MUTATION = (0.5, 1.8)
DE_N_RESTARTS = 8
DE_START_RESTART = 0 # run all restarts
# ODE solver
SIM_STEPS_PER_YEAR = 2    # minimum output grid; calibration only reads final state per year
SOLVE_TIMEOUT_S = 120 # time limit per ODE solve
TRAJECTORY_TIMEOUT_S = 180 # time limit per full trajectory evaluation
MAX_SOLVER_STEPS = 10_000 # limit on LSODA internal steps per yearly solve

SEED = 133
MAX_PARALLEL_JOBS = int(
    os.environ.get(
        "CALIB_MAX_JOBS",
        "128" if os.environ.get("ON_CLUSTER") else "128",
    )
)

# NSGA-II multi-objective optimizer
NSGA2_POP_SIZE = 200  # individuals per generation
NSGA2_N_GEN    = 800  # max generations
NSGA2_SEED     = SEED
NSGA2_TIMEOUT_S = 60   # time limit per NSGA-II candidate

# Objectives for multi-objective evaluation (all minimised)
LOSS_OBJECTIVES = ["nrmse_C", "traj_corr_C", "rank_products", "nrmse_P"]