# SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT

import numpy as np

#----------------------------------------------------------------------------
# Singleton GBM pricer.

class GBM:
    def __init__(self, X0, mu, sigma, lambda_jump, mu_jump, sigma_jump, flag_jump, seed):
        self.X0 = X0
        self.mu = mu
        self.sigma = sigma
        self.lambda_jump = lambda_jump  # Intensity of jumps (Poisson rate)
        self.mu_jump = mu_jump          # Mean of log-jump size
        self.sigma_jump = sigma_jump    # Volatility of log-jump size
        self.flag_jump = flag_jump
        np.random.seed(seed)  # TODO: Configurable?

    def generate_wiener_process(self, T, N):
        dt = T / N
        dW = np.random.normal(0, np.sqrt(dt), N)
        W = np.cumsum(dW)
        W = np.hstack((0, W))
        return W
    
    def generate_poisson_process(self, T, N):
        dt = T / N
        N_jump = np.random.poisson(self.lambda_jump * dt, N)
        N_jump = np.hstack((0, N_jump))  # Include the initial condition
        return N_jump

    def generate_jump_sizes(self, N):
        J = np.random.normal(self.mu_jump, self.sigma_jump, N)
        J = np.hstack((0, J))  # Include the initial condition
        return J

    def price(self,t, W_tuple, T, N):
        W = np.array(W_tuple)
        time_index = int(t * N / T)
        W_t = W[time_index]
        return self.X0 * np.exp((self.mu - 0.5 * self.sigma**2) * t + self.sigma * W_t)
    
    def price_jumps(self, t, W_tuple, N_jump_tuple, J_tuple, T, N):
        W = np.array(W_tuple)
        N_jump = np.array(N_jump_tuple)
        J = np.array(J_tuple)
        time_index = int(t * N / T)
        W_t = W[time_index]
        S_t = self.X0 * np.exp((self.mu - 0.5 * self.sigma**2) * t + self.sigma * W_t) 
        if  self.flag_jump and N_jump[time_index] > 0:
            S_t *= np.exp(J[time_index])
        return S_t
    
    def price_series(self, *, T, N):
        t = np.linspace(0, T, N + 1)
        W = self.generate_wiener_process(T=T, N=N)
        W_tuple = tuple(W)  # Convert to a hashable type
        Xt = self.X0 * np.exp((self.mu - 0.5 * self.sigma**2) * t + self.sigma * W)
        return t, Xt, W_tuple
    
    def price_series_jumps(self, *, T, N):
        t = np.linspace(0, T, N + 1)
        W = self.generate_wiener_process(T=T, N=N)
        if self.flag_jump:
            N_jump = self.generate_poisson_process(T=T, N=N)
            J = self.generate_jump_sizes(N=N)
            jump_factor = np.exp(np.cumsum(N_jump * J))
            Xt = self.X0 * np.exp((self.mu - 0.5 * self.sigma**2) * t + self.sigma * W) * jump_factor
        else:
            N_jump = np.zeros(N + 1)
            J = np.zeros(N + 1)
            Xt = self.X0 * np.exp((self.mu - 0.5 * self.sigma**2) * t + self.sigma * W)
        W_tuple = tuple(W)
        N_jump_tuple = tuple(N_jump)
        J_tuple = tuple(J)  
        return t, Xt, W_tuple, N_jump_tuple, J_tuple
