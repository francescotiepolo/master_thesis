import numpy as np
import networkx as nx
from scipy.integrate import solve_ivp

class BaseModel:


    def __init__(self, n_products: int=10, n_countries: int=20, nestedness: float=0.6,
                 connectance: float=0.25, forbidden_links: float=0.1, **params):
        '''
        Initialize the base model.
        Parameters:
        - n_products: Number of products (int)
        - n_countries: Number of countries (int)
        - nestedness: Desired nestedness level (float between 0 and 1)
        - connectance: Desired connectance level (float between 0 and 1)
        - forbidden_links: Fraction of forbidden links (float between 0 and 1)
        - **params: Additional model parameters (dict)
        '''

        self.SP = n_products
        self.SC = n_countries
        self.nestedness = nestedness
        self.connectance = connectance
        self.forbidden_links = forbidden_links

        self.params = {
            'r_P_range': (0.4, 1.0),       # Product intrinsic growth rates
            'r_C_range': (0.4, 1.0),       # Country intrinsic growth rates
            'h': 0.5,                      # Saturation parameter
            'mu': 1e-6,                    # Migration rate
            'eta': 0.5,                    # Specialization reward exponent
            'nu': 0.7,                     # Adaptation strength (1=no adaptation)
            'q': 0.2,                      # Resource congestion
            'C_P': 0.01,                   # Product-product competition
            'C_C': 0.01,                   # Country-country competition
            'd_C': 0.0,                    # Driver of decline (not time dependent)
            'lambda': 0.0,                 # Rate of change of decline
        }
        self.params.update(params)

        # Generate network and initialize
        self.generate_feasible_network()
        self.initialize_parameters()


    def generate_network(self):
        '''
        Generate nested bipartite network
        '''

        # Create adjacency matrix
        adj_matrix = np.zeros((self.SC, self.SP))

        # Generate nested structure
        # Countries
        country_degrees = np.random.power(2, self.SC) # Power-law distribution
        country_degrees = (country_degrees / country_degrees.max()) * self.SP * self.connectance # Normalize and scale by maximum number of possible links
        country_degrees = np.maximum(country_degrees.astype(int), 1) # At least 3 connections

        # Same for products
        product_degrees = np.random.power(2, self.SP)
        product_degrees = (product_degrees / product_degrees.max()) * self.SC * self.connectance
        product_degrees = np.maximum(product_degrees.astype(int), 1)

        for i in range(self.SC):
            # Countries with high degree connect to high-degree products (like in original paper)
            n_connections = min(country_degrees[i], self.SP)
            probs = product_degrees / product_degrees.sum()
            products = np.random.choice(self.SP, size=n_connections, replace=False, p=probs)
            adj_matrix[i, products] = 1

        # Apply forbidden links
        n_forbidden = int(adj_matrix.sum() * self.forbidden_links)
        existing_links = np.argwhere(adj_matrix == 1)            
        if len(existing_links) > 0:
            forbidden_idx = np.random.choice(len(existing_links), size=min(n_forbidden, len(existing_links)), replace=False)
            for idx in forbidden_idx:
                adj_matrix[existing_links[idx][0], existing_links[idx][1]] = 0
            
        self.adj_matrix = adj_matrix
        self.KC = adj_matrix.sum(axis=1) # Country degrees
        self.KP = adj_matrix.sum(axis=0) # Product degrees

    def generate_feasible_network(self, max_param_tries: int=100, max_network_tries: int=10):
        '''
        Generate a feasible network following original paper methodology
        '''

        for network_try in range(max_network_tries):
            self.generate_network()
            
            for param_try in range(max_param_tries):
                self.initialize_parameters()
                
                if self.network_feasible():
                    print(f"Feasible network found"
                          f"(network attempt {network_try+1}/{max_network_tries}, "
                          f"param attempt {param_try+1}/{max_param_tries})")
                    return True
            
            print(f"Network {network_try+1}/{max_network_tries}: "
                  f"No feasible parameters after {max_param_tries} tries")
        
        print(f"WARNING: Could not find feasible network after {max_network_tries} tries")

    def network_feasible(self, extinction_threshold: float=0.01):
        '''
        Check if all species survive at equilibrium with no external stress (d_C=0)
        '''

        # Save current state
        d_C_original = self.params['d_C']
        lambda_original = self.params['lambda']
        
        # Set no stress
        self.params['d_C'] = 0.0
        self.params['lambda'] = 0.0
        
        # Run to equilibrium
        result = self.simulate(t_max=1000)
        
        # Restore parameters
        self.params['d_C'] = d_C_original
        self.params['lambda'] = lambda_original
        
        # Check survival
        products_alive = np.all(result['P'] > extinction_threshold)
        countries_alive = np.all(result['C'] > extinction_threshold)
        
        return products_alive and countries_alive
    
    def initialize_parameters(self):
        '''
        Initialize model parameters based on the generated network
        '''

        # Intrinsic growth rates
        self.r_P = np.random.uniform(*self.params['r_P_range'], size=self.SP)
        self.r_C = np.random.uniform(*self.params['r_C_range'], size=self.SC)


        # Competition matrices (diagonal)
        self.C_PP = np.eye(self.SP) * self.params["C_P"]
        self.C_CC = np.eye(self.SC) * self.params["C_C"]

        # Initialize matching matrix beta
        self.beta_base = self.adj_matrix * np.random.uniform(0.5, 1.0, size=(self.SC, self.SP))

        # Initialize alphas (uniform distribution)
        self.alpha_init = self.adj_matrix.copy().astype(float)
        for i in range (self.SC):
            if self.KC[i] > 0:
                self.alpha_init[i] /= self.alpha_init[i].sum() # Normalize so that sum of alphas for each country is 1

        # Normalize beta
        eta = self.params['eta']
        self.beta_C = np.zeros_like(self.beta_base)
        self.beta_P = np.zeros_like(self.beta_base)

        for i in range(self.SC):
            for j in range(self.SP):
                if self.adj_matrix[i, j] > 0:
                    self.beta_C[i, j] = self.beta_base[i, j] / (self.KP[j] ** eta * self.alpha_init[i, j])
                    self.beta_P[i, j] = self.beta_base[i, j] / (self.KC[i] ** eta * self.alpha_init[i, j])

    def compute_resource_availability(self, P: np.ndarray, C: np.ndarray, alpha: np.ndarray):
        '''
        Compute supply-demand ratio xi for each product
        '''

        q = self.params['q']

        xi = np.zeros(self.SP)
        for i in range(self.SP):
            demand = np.sum(alpha[:, i] * self.beta_C[:, i] * C)
            if demand > 1e-10:
                xi[i] = P[i] / (demand ** q)
            else:
                xi[i] = P[i]

        return xi

    def holling_function(self, rho: np.ndarray):
        '''
        Holling type-II saturation function
        '''

        h = self.params['h']
        return rho / (1 + rho / h)
    
    def mutualistic_benefits(self, P: np.ndarray, C: np.ndarray, alpha: np.ndarray, xi: np.ndarray):
        '''
        Compute mutualistic benefits for countries and products
        
        Returns:
        - rho_C: Country benefits
        - rho_P: Product benefits
        '''

        rho_C = np.sum(alpha * self.beta_C * xi[np.newaxis, :], axis=1)
        rho_P = np.sum(alpha * self.beta_P * C[:, np.newaxis], axis=0)

        return rho_C, rho_P
    
    def ODEs(self, t: float, y: np.ndarray):
        '''
        Compute derivatives for the ODE system
        State vector y = [P_1, ..., P_SP, C_1, ..., C_SC, alpha_11, alpha_12, ...]
        '''

        P = y[:self.SP]
        C = y[self.SP:self.SP + self.SC]
        alpha = y[self.SP + self.SC:].reshape(self.SC, self.SP)

        # Ensure all values are positive
        P = np.maximum(P, 1e-10)
        C = np.maximum(C, 1e-10)
        alpha = np.maximum(alpha, 0)

        # Ensure alphas sum to 1 for each country
        for j in range(self.SC):
            if alpha[j].sum() > 1e-10:
                alpha[j] /= alpha[j].sum()

        xi = self.compute_resource_availability(P, C, alpha)
        rho_C, rho_P = self.mutualistic_benefits(P, C, alpha, xi)

        # Driver of decline
        d_C_t = self.params['d_C'] + self.params['lambda'] * t
        
        # Product dynamics
        dP = np.zeros(self.SP)
        competition_P = self.C_PP @ P 
        dP = P * (self.r_P - competition_P + self.holling_function(rho_P)) + self.params['mu']

        # Country dynamics
        competition_C = self.C_CC @ C
        dC = C * (self.r_C - d_C_t - competition_C + self.holling_function(rho_C)) + self.params['mu']

        # Adaptation dynamics
        nu = self.params['nu']
        dalpha = np.zeros((self.SC, self.SP))

        for i in range(self.SC):
            # Fitness for each product
            fitness = self.beta_C[i] * xi
            mean_fitness = np.sum(alpha[i] * fitness)

            # Number of connected products
            S_P_i = self.adj_matrix[i].sum()

            for j in range(self.SP):
                if self.adj_matrix[i, j] > 0:
                    replicator = alpha[i, j] * (fitness[j] - mean_fitness)
                    stabilizer = (1.0 / S_P_i) - alpha[i, j]
                    dalpha[i, j] = replicator * (1 - nu) + stabilizer * nu

        dy = np.concatenate([dP, dC, dalpha.flatten()])

        return dy
    
    def simulate(self, y0: np.ndarray=None, t_max: float=10, **solver_params):
        '''
        Simulate the ODE system

        Returns:
        - dict with keys 't', 'y', 'P', 'C', 'alpha', 'success'
        '''

        if y0 is None:
            # Initial conditions: all set to 1
            P0 = np.ones(self.SP)
            C0 = np.ones(self.SC)
            alpha0 = self.alpha_init.copy()
            y0 = np.concatenate([P0, C0, alpha0.flatten()])

        # Solve ODE
        sol = solve_ivp(self.ODEs, (0, t_max), y0, method='LSODA', rtol=1e-3, atol=1e-6, **solver_params)
        y_final = sol.y[:, -1]
        P_final = y_final[:self.SP]
        C_final = y_final[self.SP:self.SP + self.SC]
        alpha_final = y_final[self.SP + self.SC:].reshape(self.SC, self.SP)

        return {
            't': sol.t,
            'y': sol.y,
            'P': P_final,
            'C': C_final,
            'alpha': alpha_final,
            'success': sol.success,
            'sol': sol
        }
    
    def find_critical_points(self, d_C_min: float=0.0, d_C_max: float=3.0, step: float=0.05, extinction_threshold: float=0.01, continue_after_collapse: int=10):
        '''
        Find critical points by varying driver of decline d_C

        Returns:
        - dict with 'd_collapse', 'd_recovery', 'd_C_forward, 'C_forward', 'P_forward', ... (and same backward)
        '''

        # Forward pass: increasing d_C until collapse
        d_C_values_forward = np.arange(d_C_min, d_C_max, step)
        C_forward = []
        P_forward = []
        self.params['d_C'] = d_C_min
        self.params['lambda'] = 0
        result = self.simulate()
        y_current = np.concatenate([result['P'], result['C'], result['alpha'].flatten()])

        d_collapse = None
        collapse_counter = 0
        for d_C in d_C_values_forward:
            self.params['d_C'] = d_C
            result = self.simulate(y0=y_current)
            y_current = np.concatenate([result['P'], result['C'], result['alpha'].flatten()])
            C_forward.append(result['C'])
            P_forward.append(result['P'])

            n_alive = np.sum(result['C'] > extinction_threshold)
            if d_collapse is None and n_alive == 0:
                d_collapse = d_C
            
            # Continue for a bit after collapse
            if d_collapse is not None:
                collapse_counter += 1
                if collapse_counter >= continue_after_collapse:
                    break

        if d_collapse is None:
            d_collapse = d_C_max

        # Backward pass: decreasing d_C from collapse point until recovery
        d_C_values_backward = np.arange(d_collapse, d_C_min, -step)
        C_backward = []
        P_backward = []

        d_recovery = None

        for d_C in d_C_values_backward:
            self.params['d_C'] = d_C
            result = self.simulate(y0=y_current)
            y_current = np.concatenate([result['P'], result['C'], result['alpha'].flatten()])
            C_backward.append(result['C'])
            P_backward.append(result['P'])

            n_alive = np.sum(result['C'] > extinction_threshold)
            if d_recovery is None and n_alive > 0:
                d_recovery = d_C
                # Not break here to continue until d_C_min to check for full recovery

        return {
            'd_collapse': d_collapse,
            'd_recovery': d_recovery if d_recovery is not None else d_C_min,
            'd_C_forward': d_C_values_forward[:len(C_forward)],
            'C_forward': np.array(C_forward),
            'P_forward': np.array(P_forward),
            'd_C_backward': d_C_values_backward[:len(C_backward)],
            'C_backward': np.array(C_backward),
            'P_backward': np.array(P_backward)
        }