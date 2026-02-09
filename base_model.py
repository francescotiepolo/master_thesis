import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

class BaseModel:


    def __init__(self, n_products: int=15, n_countries: int=35, nestedness: float=0.6,
                 connectance: float=0.15, forbidden_links: float=0.3, **params):
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
        self.generate_network()
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
        country_degrees = np.maximum(country_degrees.astype(int), 1) # At least one connection

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

    def initialize_parameters(self):
        '''
        Initialize model parameters based on the generated network
        '''

        # Intrinsic growth rates
        self.r_P = np.random.uniform(*self.params['r_P_range'], size=self.SP)
        self.r_C = np.random.uniform(*self.params['r_C_range'], size=self.SC)


        # Competition matrices
        self.C_PP = np.ones((self.SP, self.SP)) * self.params['C_P']
        self.C_CC = np.ones((self.SC, self.SC)) * self.params['C_C']

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
                xi[i] = 0.0

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
        - rho_C : Country benefits
        - rho_P : Product benefits
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
        alpha = np.maximum(alpha, 1e-10)

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
        for i in range(self.SP):
            competition = np.sum(self.C_PP[i] * P)
            mutualism = self.holling_function(rho_P[i])
            dP[i] = P[i] * (self.r_P[i] - competition + mutualism) + self.params['mu']

        # Country dynamics
        dC = np.zeros(self.SC)
        for i in range(self.SC):
            competition = np.sum(self.C_CC[i] * C)
            mutualism = self.holling_function(rho_C[i])
            dC[i] = C[i] * (self.r_C[i] - d_C_t - competition + mutualism) + self.params['mu']

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
    

