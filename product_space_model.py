"""
ProductSpaceModel: extends BaseModel with product space integration.

Implements the full final model:
  - Product dynamics with proximity-dependent competition and knowledge spillovers
  - Country dynamics with capability accumulation via density in product space
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
        N_products: int = 15,
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
        patch_network: bool = False,
        # Product space parameters
        s: float = 0.01,               # Spillover strength
        c: float = 0.02,               # Competition strength for related products
        c_prime: float = 0.005,        # Competition strength for unrelated products
        gamma: float = 1.0,            # Capability accumulation parameter
        kappa: float = 0.05,           # Proximity competition parameter
        sigma: float = 1.0,            # Diminishing returns to effort
        phi_space: np.ndarray = None,  # (SP × SP) proximity matrix; generated if None
        # Product entry parameters
        enable_entry: bool = False,    # Main switch for new-product entry
        entry_threshold: float = 0.1,  # Activation threshold for entry signal
    ):
        self.s = s
        self.c = c
        self.c_prime = c_prime
        self.gamma = gamma
        self.kappa = kappa
        self.sigma = sigma
        self._phi_space_init = phi_space
        self.enable_entry = enable_entry
        self.entry_threshold = entry_threshold

        super().__init__(
            N_products=N_products,
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
            patch_network=patch_network,
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

        # Row means of phi for density calculation (mean proximity per product).
        # Using the mean rather than the sum (like in Hidalgo) ensures density_ij is normalised by the
        # average connectivity of product i, not its total, so Cap_j stays in [0, 1] regardless of SP.
        self.phi_row_sum = self.phi_space.mean(axis=1)



    # Product entry mechanism

    def activate_new_links(self, P, C, alpha):
        """
        Check for and activate new (country, product) links based on
        proximity-weighted production in the product space.

        Called between yearly simulation steps when enable_entry=True.
        Modifies beta_C, beta_P, alpha, adj_matrix, KC, KP in-place.

        Returns the number of newly activated links.
        """
        if not self.enable_entry:
            return 0

        # signal_ji = sum_{i'} phi_ii' * alpha_ji' * beta_C[j,i'] * P_i'
        weighted_phi = self.phi_space * P[np.newaxis, :] # (SP, SP)
        signal = (alpha * self.beta_C) @ weighted_phi # (SC, SP)

        # Only consider currently-inactive pairs
        inactive = self.beta_C == 0.0
        candidates = inactive & (signal > self.entry_threshold)

        n_new = candidates.sum()
        if n_new == 0:
            return 0

        # Set beta values from proximity-weighted averages of active neighbors
        for j in range(self.SC):
            new_products = np.where(candidates[j])[0]
            if len(new_products) == 0:
                continue

            active_j = self.beta_C[j] > 0.0  # (SP,) mask of active products for country j

            for i in new_products:
                phi_i = self.phi_space[i] # Proximity of product i to all products
                weights = phi_i * active_j # Zero out inactive
                w_sum = weights.sum()
                self.beta_C[j, i] = (weights @ self.beta_C[j]) / w_sum
                self.beta_P[j, i] = (weights @ self.beta_P[j]) / w_sum

        # Update network
        self.adj_matrix = (self.beta_C > 0).astype(float)
        self.KC = self.adj_matrix.sum(axis=1).astype(float)
        self.KP = self.adj_matrix.sum(axis=0).astype(float)

        return int(n_new)

    def _after_step(self, P, C, alpha):
        """
        Product entry is handled once per year in simulate(),
        not at every solver output step.
        """
        pass



    # Solver override

    def solve(self, *args, **kwargs):
        kwargs.setdefault("rtol", 1e-4)
        kwargs.setdefault("atol", 1e-7)
        return super().solve(*args, **kwargs)
    
    def _set_sol(self, sol):
        """
        Adapts _set_sol for product space model.
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
        P = z[:self.SP]
        C = z[self.SP:self.N]
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
    # Cap_j = Σ_i α_ij · density_ij: how well-positioned country j is given its current specialisation
    density = _density_numba(alpha, phi_space, phi_row_sum)
    Cap = np.zeros(SC)
    for j in range(SC):
        for i in range(SP):
            Cap[j] += alpha[j, i] * density[j, i]

    # Country dynamics
    # Mutualism term for countries — capability position amplifies mutualistic returns
    # gamma=0: no amplification; gamma*Cap_j: proportional boost from product-space density
    rho_C_total = rho_C * (1.0 + gamma * Cap)
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