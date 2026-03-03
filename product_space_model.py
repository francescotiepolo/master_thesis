"""
ProductSpaceModel: extends BaseModel with product space integration.

Implements the full final model:
  - Product dynamics with proximity-dependent competition and knowledge spillovers
  - Country dynamics with capability accumulation via density in product space
  - Adaptive specialization (unchanged from base)
  - Proximity-dependent resource competition

All unchanged logic (network generation, feasibility, solver, hysteresis, etc.)
is ported directly from BaseModel.
"""

import numpy as np
import numba
from base_model import BaseModel


class ProductSpaceModel(BaseModel):

    def __init__(
        self,
        # Base model parameters
        N_croducts: int = 15,
        n_countries: int = 35,
        nestedness: float = 0.6,
        connectance: float = 0.15,
        forbidden_links: float = 0.3,
        seed=None,
        nu: float = 1.0,
        G: float = 1.0,
        q: float = 0.0,
        mu: float = 0.0001,
        beta_trade_off: float = 0.5,
        feasible: bool = True,
        feasible_iters: int = 100,
        # Product space parameters
        s: float = 0.01,               # Spillover strength
        c: float = 0.02,               # Competition strength for related products
        c_prime: float = 0.005,        # Competition strength for unrelated products
        gamma: float = 0.05,           # Capability accumulation parameter
        kappa: float = 0.05,           # Proximity competition parameter
        sigma: float = 1.0,            # Diminishing returns to effort
        phi_space: np.ndarray = None,  # (SP × SP) proximity matrix; generated if None
        prod_space_steps: int = 100,   # Steps to bring product space into feasibility check
    ):
        self.s = s
        self.c = c
        self.c_prime = c_prime
        self.gamma = gamma
        self.kappa = kappa
        self.sigma = sigma
        self._phi_space_init = phi_space  # Will be set properly after network exists
        self.prod_space_steps = prod_space_steps

        super().__init__(
            N_croducts=N_croducts,
            n_countries=n_countries,
            nestedness=nestedness,
            connectance=connectance,
            forbidden_links=forbidden_links,
            seed=seed,
            nu=nu,
            G=G,
            q=q,
            mu=mu,
            beta_trade_off=beta_trade_off,
            feasible=feasible,
            feasible_iters=feasible_iters,
        )



    # Parameter initialisation

    def initialize_parameters(self):
        """
        Calls the base initialiser then sets up product space structures.
        """
        super().initialize_parameters()  # Sets base model parameters and generates network

        # Proximity matrix ϕ_{ii'} (SP × SP, symmetric, zero diagonal)
        if self._phi_space_init is not None:
            assert self._phi_space_init.shape == (self.SP, self.SP), ("phi_space must have shape (SP, SP)")
            self.phi_space = self._phi_space_init.astype(float)
        else:
            # Generate a random symmetric proximity matrix from Uniform(0,1), in case data is not used
            raw = self.rng.uniform(0.0, 1.0, (self.SP, self.SP))
            sym = (raw + raw.T) / 2.0
            sym[sym < 0.6] = 0.0   # Only keep strong proximities???
            np.fill_diagonal(sym, 0.0)
            self.phi_space = sym

        self._build_product_space_matrices()

    def _build_product_space_matrices(self):
        """
        Build matrices derived from the proximity matrix that are needed for the ODEs.
        """
        # In case phi_space is not yet set (during early feasibility checks), skip building these matrices
        if not hasattr(self, 'phi_space'):
            return
        
        # Proximity-dependent competition matrix θ_{ii'} (SP × SP)
        self.theta = self.phi_space * self.c + (1.0 - self.phi_space) * self.c_prime
        np.fill_diagonal(self.theta, 0.0)  # no self-competition via theta

        # Row-normalised proximity for density (sum_i' ϕ_{ii'})
        self.phi_row_sum = self.phi_space.sum(axis=1)  # Shape (SP,)



    # Solver override

    def solve(self, *args, **kwargs):
        kwargs.setdefault("rtol", 1e-6)
        kwargs.setdefault("atol", 1e-9)
        return super().solve(*args, **kwargs)
    
    def _set_sol(self, sol):
        """
        Adapts _set_sol for product space model
        """
        # Fix 1: ensure sol.y is a 2-D numpy array of shape (N, n_timepoints)
        if not isinstance(sol.y, np.ndarray) or sol.y.ndim < 2:
            y_2 = (
                np.clip(self.y[:, -1], 0, None)
                if self.y is not None
                else np.ones(self.N)
            )
            sol.y = y_2[:, np.newaxis]
            sol.t = np.array([0.0])

        super()._set_sol(sol)

        # Fix 2: ensure y_partial is never None
        if self.y_partial is None or self.y_partial.shape[1] == 0:
            self.y_partial = self.alpha.flatten()[:, np.newaxis]


    # ODE building blocks

    def _xi(self, P, alpha, C):
        """
        Proximity-dependent resource availability ξ_i.
        """
        # E_i: shape (SP,)
        E = (alpha * self.beta_C).T @ C  # Same as alpha_beta_C_prod in base

        effort_per_product = alpha.T @ C  # shape (SP,)

        # Avoid division by zero when a product has no effort devoted to it
        with np.errstate(divide="ignore", invalid="ignore"):
            phi_weighted_effort = self.phi_space @ effort_per_product # Shape (SP,)
            competition_ratio = np.where(
                effort_per_product > 0,
                phi_weighted_effort / effort_per_product,
                0.0,
            )

        X = 1.0 + self.kappa * competition_ratio # Shape (SP,)

        eps = 1e-10
        with np.errstate(divide="ignore", invalid="ignore"):
            xi = np.where(
                (E * X) > eps,
                (P / ((E * X) ** self.q)) ** self.sigma, # q kept here to study dynamics similar to base model
                0.0,
            )
        return np.nan_to_num(xi, nan=0.0, posinf=0.0, neginf=0.0)

    def _density(self, alpha):
        """
        Density of country j in the product space around product i.
        """
        numerator = alpha @ self.phi_space.T # Shape (SC, SP)

        denominator = self.phi_row_sum # Shape (SP,)

        with np.errstate(divide="ignore", invalid="ignore"):
            density = np.where(
                denominator > 0,
                numerator / denominator[np.newaxis, :],
                0.0,
            )
        return density # Shape (SC, SP)



    # ODEs

    def ODEs(self, t: float, z: np.ndarray, d_C: float) -> np.ndarray:
        """
        Full ODE system with product space integration.

        State vector  z = [P (SP) | C (SC) | alpha (SC×SP, flattened)]
        """
        P = z[:self.SP]
        C = z[self.SP:self.N]
        alpha = z[self.N:].reshape((self.SC, self.SP))

        # Ensure non-negativity of P, C, and alpha to avoid numerical issues in the ODEs
        P = np.clip(P, 0.0, None)
        C = np.clip(C, 0.0, None)
        alpha = np.clip(alpha, 0.0, 1.0)

        # Resource availability ξ_i
        xi = self._xi(P, alpha, C) # Shape (SP,)

        # Mutualistic benefits
        rho_C = (alpha * self.beta_C) @ xi # Shape (SC,)
        rho_P = (alpha * self.beta_P).T @ C # Shape (SP,)

        # Knowledge spillovers
        spillovers = self.s * (self.phi_space @ P) # Shape (SP,)

        # Product dynamics
        C_PP_eff = self.C_PP * self.theta # Off-diagonal weighted
        np.fill_diagonal(C_PP_eff, np.diag(self.C_PP)) # Restore self-competition
        competition_P = C_PP_eff @ P # Shape (SP,)

        dP = P * (
            self.r_P
            - competition_P
            + self._mutualism(rho_P + spillovers, self.h_P)
        ) + self.mu

        # Capability accumulation Cap_j
        density = self._density(alpha) # Shape (SC, SP)
        Cap = (alpha * density).sum(axis=1) # Shape (SC,)

        # Country dynamics
        dC = C * (
            self.r_C - d_C
            - self.C_CC @ C
            + self._mutualism(rho_C + self.gamma * Cap, self.h_C)
        ) + self.mu

        # Adaptive specialization
        if self.nu == 1.0 or self.G == 0.0:
            dalpha = np.zeros((self.SC, self.SP))
        else:
            dalpha = self.G * (
                (1.0 - self.nu) * alpha * (
                    self.beta_C * xi[np.newaxis, :] # Fitness per product
                    - rho_C[:, np.newaxis] # Mean fitness per country
                )
                + _nu_term(alpha, self.beta_C, self.SC, self.SP, self.nu)  # Stabilising term
            )

        return np.concatenate((dP, dC, dalpha.flatten()))



# Numba helper function

@numba.njit()
def _nu_term(alpha, beta_C, SC, SP, nu):
    """
    Numba-accelerated stabilising (nu) term in the alpha ODE.
    """
    nu_term = np.zeros((SC, SP))
    for i in range(SC):
        for j in range(SP):
            if beta_C[i, j] != 0:
                nu_term[i, j] = 1.0 / np.count_nonzero(beta_C[i, :]) - alpha[i, j]
    return nu_term * nu