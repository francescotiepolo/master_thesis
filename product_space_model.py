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
    ):
        self.s = s
        self.c = c
        self.c_prime = c_prime
        self.gamma = gamma
        self.kappa = kappa
        self.sigma = sigma
        self._phi_space_init = phi_space  # Will be set properly after network exists

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
        np.fill_diagonal(self.theta, 0.0)  # No self-competition via theta

        # Row sums of phi for density calculation
        self.phi_row_sum = self.phi_space.sum(axis=1) 



    # Solver override

    def solve(self, *args, **kwargs):
        kwargs.setdefault("rtol", 1e-6)
        kwargs.setdefault("atol", 1e-9)
        return super().solve(*args, **kwargs)
    
    def _set_sol(self, sol):
        """
        Adapts _set_sol for product space model
        """
        # Ensure sol.y is a 2-D numpy array of shape (N, n_timepoints)
        if not isinstance(sol.y, np.ndarray) or sol.y.ndim < 2:
            y_2 = (
                np.clip(self.y[:, -1], 0, None)
                if self.y is not None
                else np.ones(self.N)
            )
            sol.y = y_2[:, np.newaxis]
            sol.t = np.array([0.0])

        super()._set_sol(sol)

        # Ensure y_partial is never None
        if self.y_partial is None or self.y_partial.shape[1] == 0:
            self.y_partial = self.alpha.flatten()[:, np.newaxis]



    # ODEs

    def ODEs(self, t: float, z: np.ndarray, d_C: float) -> np.ndarray:
        """
        Full ODE system with product space integration.

        State vector  z = [P (SP) | C (SC) | alpha (SC×SP, flattened)]
        """
        P     = z[:self.SP]
        C     = z[self.SP:self.N]
        alpha = z[self.N:].reshape((self.SC, self.SP))

        dP, dC, dalpha = _odes_inner(
            P, C, alpha,
            self.r_P, self.r_C,
            self.C_PP, self.C_CC, self.theta,
            self.beta_C, self.beta_P,
            self.phi_space, self.phi_row_sum,
            self.h_P, self.h_C,
            self.mu, self.nu, self.G,
            self.gamma, self.kappa, self.q, self.sigma, self.s,
            d_C,
        )
        return np.concatenate((dP, dC, dalpha.flatten()))



# Numba functions

@numba.njit(cache=True)
def _xi_numba(P, alpha, beta_C, C, phi_space, kappa, q, sigma):
    """
    Proximity-dependent resource availability ξ_i.
    """
    SP = P.shape[0]

    E = (alpha * beta_C).T @ C
    effort = alpha.T @ C
    phi_w  = phi_space @ effort 

    xi = np.empty(SP)
    eps = 1e-10
    for i in range(SP):
        cr = phi_w[i] / effort[i] if effort[i] > 0.0 else 0.0
        X  = 1.0 + kappa * cr
        EX = E[i] * X
        if EX > eps:
            xi[i] = (P[i] / (EX ** q)) ** sigma
        else:
            xi[i] = 0.0
    return xi


@numba.njit(cache=True)
def _density_numba(alpha, phi_space, phi_row_sum):
    """
    Density of country j in the product space around product i.
    """
    numerator = alpha @ phi_space.T
    SC, SP = alpha.shape
    density = np.empty((SC, SP))
    for i in numba.prange(SP):
        denom = phi_row_sum[i]
        for j in range(SC):
            density[j, i] = numerator[j, i] / denom if denom > 0.0 else 0.0
    return density


@numba.njit(cache=True)
def _odes_inner(
    P, C, alpha,
    r_P, r_C,
    C_PP, C_CC, theta,
    beta_C, beta_P,
    phi_space, phi_row_sum,
    h_P, h_C,
    mu, nu, G,
    gamma, kappa, q, sigma, s,
    d_C,
):
    """
    Full product-space ODEs
    """
    SP = P.shape[0]
    SC = C.shape[0]

    # Ensure non-negativity
    P = np.maximum(P, 0.0)
    C = np.maximum(C, 0.0)
    alpha = np.minimum(np.maximum(alpha, 0.0), 1.0)

    # ξ_i
    xi = _xi_numba(P, alpha, beta_C, C, phi_space, kappa, q, sigma)

    # Mutualistic benefits
    rho_C = (alpha * beta_C) @ xi 
    rho_P = (alpha * beta_P).T @ C

    # Knowledge spillovers
    spillovers = s * (phi_space @ P)

    # Product dynamics
    # Proximity-weighted competition matrix (off-diagonal scaled by theta)
    C_PP_eff = C_PP * theta
    for i in range(SP):
        C_PP_eff[i, i] = C_PP[i, i] # Restore self-competition on diagonal
    competition_P = C_PP_eff @ P

    # Mutualism term for products
    rho_P_total = rho_P + spillovers
    mut_P = rho_P_total / (1.0 + h_P * rho_P_total)

    dP = P * (r_P - competition_P + mut_P) + mu

    # Capability accumulation
    density = _density_numba(alpha, phi_space, phi_row_sum)
    Cap = np.zeros(SC)
    for j in range(SC):
        for i in range(SP):
            Cap[j] += alpha[j, i] * density[j, i]

    # Country dynamics
    # Mutualism term for countries
    rho_C_total = rho_C + gamma * Cap
    mut_C = rho_C_total / (1.0 + h_C * rho_C_total)

    dC = C * (r_C - d_C - C_CC @ C + mut_C) + mu

    # Adaptive specialisation
    dalpha = np.zeros((SC, SP))
    if nu != 1.0 and G != 0.0:
        for j in range(SC):
            n_active = np.count_nonzero(beta_C[j, :])
            if n_active == 0:
                continue
            for i in range(SP):
                if beta_C[j, i] != 0.0:
                    nu_stab = (1.0 / n_active - alpha[j, i]) * nu
                    dalpha[j, i] = G * (
                        (1.0 - nu) * alpha[j, i] * (
                            beta_C[j, i] * xi[i] - rho_C[j]
                        )
                        + nu_stab
                    )

    return dP, dC, dalpha