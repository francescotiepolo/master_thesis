"""
BaseModel: product-country model adapted from
the original AdaptiveModel (Terpstra 2022, https://github.com/TerpstraS/pollcomm.git).

Variable mapping  (from original):
Plants       →  Products  (SP = 15, index 0..SP-1  in state vector)
Pollinators  →  Countries (SC = 35, index SP..SP+SC-1 in state vector)

The find_critical_points method is a direct port of experiments.hysteresis().
"""

import copy
import numpy as np
import numba
from ode_solver import solve_ode    # Direct port from Terpstra 2022


class BaseModel:

    def __init__(
        self,
        N_products: int = 15,           # Number of products (as plants in original)
        n_countries: int = 35,          # Number of countries (as pollinators in original)
        nestedness: float = 0.6,        # Nestedness of network (0 to 1)
        connectance: float = 0.15,      # Connectance of network (0 to 1)
        forbidden_links: float = 0.3,   # Proportion of forbidden links (0 to 1)
        seed=None,                      # Random seed for reproducibility
        nu: float = 1.0,                # 1 = no adaptation, <1 = adaptive foraging
        G: float = 1.0,                 # Speed of adaptive foraging
        q: float = 0.0,                 # Resource-congestion (supply/demand) exponent
        mu: float = 0.0001,             # Immigration
        beta_trade_off: float = 0.5,    # Eta variable in paper (0 = no trade-off, 1 = strong trade-off)
        feasible: bool = True,          # Whether to enforce feasibility (all species alive at d_C=0) during initialization
        feasible_iters: int = 100,      # Number of iterations to attempt to generate a feasible network
        patch_network: bool = False,    # Skip network generation entirely — for calibration where _patch_model overwrites everything
    ):
        self.SP = N_products
        self.SC = n_countries
        self.N = self.SP + self.SC   # Total species

        self.nestedness = nestedness
        self.connectance = connectance
        self.forbidden_links = forbidden_links
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        self.nu = nu
        self.G = G
        self.q = q
        self.beta_trade_off = beta_trade_off
        self.mu = mu
        self.extinct_threshold = 0.01

        self.feasible = feasible
        self.feasible_iters = feasible_iters
        self.patch_network = patch_network
        self.is_feasible = None

        # Solution storage
        self.t = None
        self.y = None # Shape (SP+SC, n_timepoints) — P and C, NO alpha
        self.y_partial = None # Shape (SC*SP, n_timepoints_partial) — alpha only
        self.y_all_end = None # Full state [P, C, alpha] at final timepoint

        self.generate_feasible_network()



    # Network generation

    def random_network(self, max_iter: int = 1000):
        """
        Generate a random bipartite network (SC × SP).
        """
        N_c, N_p = self.SC, self.SP
        for _ in range(max_iter):
            net = np.zeros((N_c, N_p)) # Adjacency matrix (SC × SP)
            forb = np.zeros((N_c, N_p)) # Forbidden links matrix (SC × SP)
            n_c = int(self.connectance * N_c * N_p) # Number of links to create
            n_f = int(self.forbidden_links * N_c * N_p) # Number of forbidden links

            sel = self.rng.choice(N_c * N_p, size=n_c + n_f, replace=False) # Randomly select positions for links and forbidden links
            i_c = self.rng.choice(sel, size=n_c, replace=False) # Indices for links
            i_f = np.setdiff1d(sel, i_c) # Indices for forbidden links

            i_c = np.column_stack(np.unravel_index(i_c, (N_c, N_p))) # Convert flat indices to 2D indices
            net[tuple(i_c.T)] = 1

            if len(i_f):
                i_f = np.column_stack(np.unravel_index(i_f, (N_c, N_p))) # Convert flat indices to 2D indices
                forb[tuple(i_f.T)] = 1

            # Every species must have at least one interaction
            if net.any(axis=0).all() and net.any(axis=1).all():
                return net, forb

        return self.random_network(max_iter)   # Retry if none found

    def _is_connected(self, M) -> bool:
        """
        BFS connectivity check for bipartite adjacency matrix:
        check if all nodes (countries and products) are reachable from each other.
        """
        N_c, N_p = M.shape
        seen, frontier = set(), {0} # Set of nodes seen so far, set of nodes to explore
        while frontier:
            nxt = set()
            for node in frontier:
                if node not in seen:
                    seen.add(node)
                    # If node is a country, add connected products; if node is a product, add connected countries
                    if node < N_c:
                        nxt.update(N_c + j for j in range(N_p) if M[node, j] > 0)
                    else:
                        nxt.update(i for i in range(N_c) if M[i, node - N_c] > 0)
            frontier = nxt
        return len(seen) == N_c + N_p # True if all nodes are visited (connected)

    def generate_network(self, max_iter: int = int(1e5), max_retries: int = 20):
        """
        Generate a nested bipartite network (SC × SP).
        """
        N_c, N_p = self.SC, self.SP

        for _ in range(max_retries):
            net, forb = self.random_network()

            converged = False
            for _ in range(max_iter):
                if _nestedness_fast(net) >= self.nestedness: # Check nestedness
                    converged = True
                    break

                # If nestedness not reached, perform a random swap of the links
                rows, cols = np.nonzero(net)
                if len(rows) == 0:
                    break
                idx = self.rng.choice(len(rows)) # Randomly select a link to swap
                a, b = int(rows[idx]), int(cols[idx])

                if self.rng.choice([0, 1]) == 0: # Randomly decide to swap a country or a product
                    c = int(self.rng.integers(0, N_c)) # Randomly select a country
                    # Rewiring rule
                    if (
                        net[c, b] == 0 and c != a and forb[c, b] == 0
                        and net[a].sum() > 1 and net[c].sum() > net[a].sum()
                    ):
                        # Swap the link from (a, b) to (c, b)
                        net[c, b] = 1
                        net[a, b] = 0
                else: # Same for product
                    c = int(self.rng.integers(0, N_p))
                    if (
                        net[a, c] == 0 and c != b and forb[a, c] == 0
                        and net[:, b].sum() > 1 and net[:, c].sum() > net[:, b].sum()
                    ):
                        net[a, c] = 1
                        net[a, b] = 0

            if converged and self._is_connected(net):
                break  # Accept this network

        # Accept whatever we have after max_retries (calibration code will overwrite anyway)

        self.adj_matrix = net 
        self.forbidden_network = forb
        self.KC = net.sum(axis=1) # Country degrees (row sums)
        self.KP = net.sum(axis=0) # Product degrees (col sums)



    # Parameter initialisation

    def initialize_parameters(self):
        """
        Sample all ecological parameters.
        """
        # Competition matrices
        self.C_PP = self.rng.uniform(0.01, 0.05, (self.SP, self.SP))
        np.fill_diagonal(self.C_PP, self.rng.uniform(0.8, 1.1, self.SP))

        self.C_CC = self.rng.uniform(0.01, 0.05, (self.SC, self.SC))
        np.fill_diagonal(self.C_CC, self.rng.uniform(0.8, 1.1, self.SC))

        # Saturation factors h
        self.h_P = self.rng.uniform(0.15, 0.3, self.SP)
        self.h_C = self.rng.uniform(0.15, 0.3, self.SC)

        # Intrinsic growth rates
        self.r_P = self.rng.uniform(0.1, 0.35, self.SP)
        self.r_C = self.rng.uniform(0.1, 0.35, self.SC)

        # Initial foraging effort alpha (SC × SP)
        with np.errstate(divide="ignore", invalid="ignore"): # Suppress warnings for zero-degree species (will be set to zero anyway)
            row_sums = self.adj_matrix.sum(axis=1, keepdims=True)
            self.alpha = np.where(
                self.adj_matrix > 0,
                self.adj_matrix / row_sums,
                0.0
            )

        # Beta matrices
        bt = self.beta_trade_off # Eta
        low, high = 0.8, 1.2

        beta_0_P = self.rng.uniform(low, high, (self.SC, self.SP))
        beta_0_C = self.rng.uniform(low, high, (self.SC, self.SP))

        mask = self.adj_matrix > 0                 

        KP_bt = self.KP[np.newaxis, :] ** bt # (1,  SP)
        KC_bt = self.KC[:, np.newaxis] ** bt # (SC, 1)

        with np.errstate(divide="ignore", invalid="ignore"):
            self.beta_P = np.where(mask, beta_0_P / (KP_bt * self.alpha), 0.0)
            self.beta_C = np.where(mask, beta_0_C / (KC_bt * self.alpha), 0.0)



    #  Feasibility

    def generate_feasible_network(self, max_network_tries: int = 10):
        """
        Generate a nested network and sample parameters until all
        species survive at equilibrium (d_C = 0).
        """
        if self.patch_network:
            # Skip network generation — _patch_model will overwrite everything
            return True
        if not self.feasible:
            # Generate network once but skip ODE feasibility check
            self.generate_network()
            self.initialize_parameters()
            return True

        for nt in range(max_network_tries):
            self.generate_network()

            for _ in range(self.feasible_iters):
                self._reset_sol()
                self.initialize_parameters()
                # Quick feasibility solve
                sol = self.solve(
                    1000, n_steps=1000, d_C=0,
                    stop_on_equilibrium=True,
                    stop_on_collapse=True
                )
                self._set_sol(sol)
                if self._is_all_alive()[-1].all(): # If all species are alive at equilibrium, we have a feasible network
                    self.is_feasible = True
                    print(f"Feasible network found")
                    return True
            print(
                f"Network {nt+1}/{max_network_tries}: "
                f"no feasible params after {self.feasible_iters} tries."
            )

        print("WARNING: could not find a feasible network.")
        self.is_feasible = False



    # Solution management

    def _reset_sol(self):
        """
        Reset solution storage.
        """
        self.t = None
        self.y = None
        self.y_partial = None
        self.y_all_end = None

    def _set_sol(self, sol):
        """
        Set solution from results.
        """
        self.t = sol.t
        self.y = sol.y # [P, C] without alpha
        if sol.y_partial is not None and len(sol.y_partial) > 0: # If alpha was saved
            self.y_partial = sol.y_partial # Alpha flattened
            # Join full solution
            self.y_all_end = np.concatenate(
                (self.y[:, -1], self.y_partial[:, -1])
            )
        else:
            self.y_all_end = copy.deepcopy(self.y[:, -1])

    def _is_all_alive(self):
        """
        Check if all species are above the extinction threshold.
        """
        if self.y is None:
            return None
        return self.y > self.extinct_threshold



    # ODE building blocks

    def _phi(self, P, alpha_beta_C_prod):
        """
        Supply/demand ratio for each product.
        """
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.nan_to_num(
                P / (alpha_beta_C_prod ** self.q), copy=False, nan=0.0
            )

    def _mutualism(self, rho, h):
        """
        Holling type-II function: rho / (1 + rho/h).
        """
        return rho / (1.0 + rho/h)



    # ODEs

    def ODEs(self, t: float, z: np.ndarray, d_C: float) -> np.ndarray:
        """
        Full ODE system.

        State vector  z = [P (SP) | C (SC) | alpha (SC×SP, flattened)]
        """
        P = z[:self.SP]
        C = z[self.SP:self.N]
        alpha = z[self.N:].reshape((self.SC, self.SP))

        dP, dC, dalpha = _odes_inner_base(
            np.ascontiguousarray(P), # Ensure contiguous arrays for numba
            np.ascontiguousarray(C),
            np.ascontiguousarray(alpha),
            self.r_P, self.r_C,
            np.ascontiguousarray(self.C_PP),
            np.ascontiguousarray(self.C_CC),
            np.ascontiguousarray(self.beta_C),
            np.ascontiguousarray(self.beta_P),
            self.h_P, self.h_C,
            self.mu, self.nu, self.G, self.q,
            d_C,
        )
        return np.concatenate((dP, dC, dalpha.ravel()))



    # Simulation

    def solve(
        self,
        t_end: float,
        d_C: float = 0.0,
        n_steps: int = int(1e4),
        y0: np.ndarray = None,
        save_period: int = 0,
        stop_on_collapse: bool = False,
        stop_on_equilibrium: bool = False,
        equi_tol: float = 1e-7,
        extinct_threshold: float = 0.01,
        rtol=1e-4,
        atol=1e-7,
        max_solver_steps=None,
        save_trajectory: bool = True,
    ):
        """
        Solve the ODE system using the same solver settings as Terpstra 2022.
        """
        if y0 is None:
            y0 = np.full(self.N, 1.0, dtype=float) # Initial state: all species at abundance 1.0
            y0 = np.concatenate((y0, self.alpha.flatten()))

        # alpha occupies indices [SP+SC  …  SP+SC+SC*SP-1] in the state vector
        alpha_start = self.N
        alpha_end = y0.shape[0] - 1

        save_partial = {
            "ind":         (alpha_start, alpha_end),
            "save_period": save_period,
        }

        sol = solve_ode(
            self.ODEs, (0, t_end), y0, n_steps,
            args=(d_C,),
            save_partial=save_partial,
            rtol=rtol, atol=atol,
            method="LSODA",
            stop_on_collapse=stop_on_collapse,
            N_p=self.SP, N_a=self.SC, # N_a is number of countries, name chosen to match the original ODE solver by Terpstra 2022 in ode_solver.py
            extinct_threshold=extinct_threshold,
            stop_on_equilibrium=stop_on_equilibrium,
            equi_tol=equi_tol,
            max_solver_steps=max_solver_steps,
            save_trajectory=save_trajectory,
        )

        self._reset_sol()
        self._set_sol(sol)
        return sol
    


    def _after_step(self, P, C, alpha):
        """
        Runs after each simulation step in find_critical_points.
        Does nothing here; used to override to update state in the ProductSpaceModel.
        """
        pass

    # Hysteresis study

    def find_critical_points(
        self,
        d_C_min: float = 0.0,
        d_C_max: float = 3.0,
        d_C_step: float = 0.02,
        steps_after_collapse: int = 5,
    ):
        """
        Forward and backward pass over d_C to map the hysteresis loop.

        The forward pass stops `steps_after_collapse` steps after all
        countries have collapsed. The backward pass then starts from
        that same collapsed state.
        """
  
        t_end = int(1e4) # Time steps for initial step
        t_step = int(1e4) # Time steps for subsequent steps

        # Create d_C values for forward pass
        dCs = np.linspace(
            d_C_min, d_C_max,
            int(round((d_C_max - d_C_min) / d_C_step)) + 1 # To have d_C_max included
        )

        # Storage lists
        P_forward_list = []
        C_forward_list = []
        alpha_forward_list = []
        dCs_forward_list = []

        # Initial equilibrium at d_C = 0 to start forward pass
        self.solve(t_end, d_C=0, save_period=0, stop_on_equilibrium=True)
        is_feasible = bool(self._is_all_alive()[-1].all())

        # Forward pass increasing d_C
        print("Forward pass...")
        collapse_counter = 0 # To count steps after collapse for nice figures
        collapsed = False
        for i, d_C in enumerate(dCs):
            print(f"{i+1}/{len(dCs)}  d_C = {d_C:.3f}")
            y0 = np.concatenate((self.y[:, -1], self.y_partial[:, -1]))
            self.solve(t_step, d_C=d_C, y0=y0, save_period=0,
                       stop_on_equilibrium=True)

            P_t = self.y[:self.SP, -1]
            C_t = self.y[self.SP:self.N, -1]
            alpha_t = self.y_partial[:, -1].reshape(self.SC, self.SP)
            self._after_step(P_t, C_t, alpha_t)

            P_forward_list.append(P_t.copy())
            C_forward_list.append(C_t.copy())
            alpha_forward_list.append(alpha_t.copy())
            dCs_forward_list.append(d_C)

            # Check whether all countries have collapsed
            all_collapsed = (
                self.y[self.SP:self.N, -1] < self.extinct_threshold
            ).all()
            if all_collapsed:
                if not collapsed:
                    collapsed = True
                    print(f"--- Full collapse at d_C = {d_C:.3f} ---")
                collapse_counter += 1
                if collapse_counter >= steps_after_collapse:
                    break

        # Convert forward lists to arrays
        dCs_forward = np.array(dCs_forward_list)
        P_forward = np.array(P_forward_list)
        C_forward = np.array(C_forward_list)
        alpha_forward = np.array(alpha_forward_list)

        # Backward pass starts from whatever state the forward pass ended on
        dCs_backward = np.flip(dCs_forward)

        P_backward = np.zeros((len(dCs_backward), self.SP))
        C_backward = np.zeros((len(dCs_backward), self.SC))
        alpha_backward = np.zeros((len(dCs_backward), self.SC, self.SP))

        # Backward pass decreasing d_C
        print("Backward pass...")
        for i, d_C in enumerate(dCs_backward):
            print(f"{i+1}/{len(dCs_backward)}  d_C = {d_C:.3f}")
            y0 = np.concatenate((self.y[:, -1], self.y_partial[:, -1]))
            self.solve(t_step, d_C=d_C, y0=y0, save_period=0,
                       stop_on_equilibrium=True)
            P_t = self.y[:self.SP, -1]
            C_t = self.y[self.SP:self.N, -1]
            alpha_t = self.y_partial[:, -1].reshape(self.SC, self.SP)
            self._after_step(P_t, C_t, alpha_t)

            P_backward[i] = P_t
            C_backward[i] = C_t
            alpha_backward[i] = alpha_t

        # Collapse / recovery threshold
        thresh = self.extinct_threshold

        # Collapse: highest d_C at which all countries are extinct
        collapsed_mask = (C_forward < thresh).any(axis=0)
        if collapsed_mask.any():
            col_idx = np.argmax(C_forward < thresh, axis=0)[collapsed_mask].max() # Get the index of the first occurrence of collapse for each country, then take the max to get the last collapse
            d_collapse = float(dCs_forward[col_idx])
        else:
            d_collapse = d_C_max

        # Recovery: lowest d_C at which any country is alive again
        recovered_mask = (C_backward > thresh).any(axis=0)
        if recovered_mask.any():
            rec_idx = np.argmax(C_backward > thresh, axis=0)[recovered_mask].min() - 1 # Get the index of the first occurrence of recovery for each country, then take the min to get the first recovery (-1 for nice plotting)
            d_recovery = float(dCs_backward[rec_idx])
        else:
            d_recovery = d_C_min

        return {
            "dCs": dCs_forward,
            "is_feasible": is_feasible,
            "d_collapse": d_collapse,
            "d_recovery": d_recovery,
            # Forward
            "d_C_forward": dCs_forward,
            "C_forward": C_forward,
            "P_forward": P_forward,
            "alpha_forward": alpha_forward,
            # Backward
            "d_C_backward": dCs_backward,
            "C_backward": C_backward,
            "P_backward": P_backward,
            "alpha_backward": alpha_backward,
        }



# Numba functions

@numba.njit(cache=True)
def _nestedness_fast(network):
    """
    Numba-accelerated nestedness metric.
    """
    N_c, N_p = network.shape
    nest_c = 0.0 # Country nestedness
    for i in range(N_c - 1): # Compare each pair of countries
        for j in range(i + 1, N_c): 
            ni = network[i].sum() # Number of products country i interacts with
            nj = network[j].sum() # Number of products country j interacts with
            nij = 0
            for k in range(N_p): # Count shared products
                if network[i, k] == 1 and network[j, k] == 1:
                    nij += 1
            if min(ni, nj) > 0:
                nest_c += nij / min(ni, nj) # Proportion of shared interactions relative to the less connected country

    nest_p = 0.0 # Same for product nestedness
    for i in range(N_p - 1):
        for j in range(i + 1, N_p):
            ni = network[:, i].sum()
            nj = network[:, j].sum()
            nij = 0
            for k in range(N_c):
                if network[k, i] == 1 and network[k, j] == 1:
                    nij += 1
            if min(ni, nj) > 0:
                nest_p += nij / min(ni, nj)

    denom = N_c * (N_c - 1) / 2 + N_p * (N_p - 1) / 2 # Total number of pairs (as in original code)
    return (nest_c + nest_p) / denom if denom > 0 else 0.0


@numba.njit(cache=True)
def _nu_term(alpha, beta_C, SC, SP, nu):
    """
    Numba-accelerated stabilising (nu) term in the alpha ODE.
    """
    nu_term = np.zeros((SC, SP))
    for i in numba.prange(SC):
        for j in range(SP):
            if beta_C[i, j] != 0:
                nu_term[i, j] = 1.0 / np.count_nonzero(beta_C[i, :]) - alpha[i, j]
    return nu_term * nu


@numba.njit(cache=True)
def _odes_inner_base(
    P, C, alpha,
    r_P, r_C,
    C_PP, C_CC,
    beta_C, beta_P,
    h_P, h_C,
    mu, nu, G, q,
    d_C,
):
    """
    Numba-compiled ODEs.
    """
    SP = P.shape[0]
    SC = C.shape[0]

    alpha_beta_C = alpha * beta_C
    alpha_beta_C_prod = alpha_beta_C.T @ C

    # Phi_i
    phi = np.empty(SP)
    for i in range(SP):
        x = alpha_beta_C_prod[i]
        if x > 0.0:
            phi[i] = P[i] / (x ** q)
        else:
            phi[i] = 0.0

    # Mutualistic benefit per country
    alpha_beta_phi = alpha_beta_C @ phi
    # Mutualistic benefit per product
    alpha_beta_P_C = (alpha * beta_P).T @ C

    # Product dynamics
    comp_P = C_PP @ P
    dP = np.empty(SP)
    for i in range(SP):
        mut_P  = alpha_beta_P_C[i] / (1.0 + h_P[i] * alpha_beta_P_C[i])
        dP[i]  = P[i] * (r_P[i] + mut_P - comp_P[i]) + mu

    # Country dynamics
    comp_C = C_CC @ C
    dC = np.empty(SC)
    for j in range(SC):
        mut_C  = alpha_beta_phi[j] / (1.0 + h_C[j] * alpha_beta_phi[j])
        dC[j]  = C[j] * (r_C[j] - d_C + mut_C - comp_C[j]) + mu

    # Adaptive foraging dynamics
    dalpha = np.zeros((SC, SP))
    if nu != 1.0 and G != 0.0:
        nu_term = _nu_term(alpha, beta_C, SC, SP, nu)
        for j in range(SC):
            for i in range(SP):
                if beta_C[j, i] != 0.0:
                    dalpha[j, i] = G * (
                        (1.0 - nu) * alpha[j, i] * (
                            beta_C[j, i] * phi[i] - alpha_beta_phi[j]
                        )
                        + nu_term[j, i]
                    )

    return dP, dC, dalpha
