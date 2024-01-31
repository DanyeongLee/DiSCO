import numpy as np
import torch


def sigmoid(x):
        return 1 / (torch.exp(-x) + 1)

def sigmoid_schedule(beta_start=1e-7, beta_end=2e-3, n_timesteps=5000):
    betas = torch.linspace(-6, 6, n_timesteps)
    return sigmoid(betas) * (beta_end - beta_start) + beta_start


def linear_schedule(beta_start=1e-7, beta_end=2e-3, n_timesteps=5000):
    return torch.linspace(beta_start, beta_end, n_timesteps)


def quadratic_schedule(beta_start=1e-7, beta_end=2e-3, n_timesteps=5000):
    return torch.linspace(beta_start ** 0.5, beta_end ** 0.5, n_timesteps) ** 2



def symmetric_sigmoid_schedule(beta_start=1e-7, beta_end=2e-3, n_timesteps=5000):
    betas = sigmoid_schedule(beta_start, beta_end, n_timesteps)
    beta0 = betas[:n_timesteps//2]

    return torch.cat([beta0, beta0.flip(0)])


def symmetric_quadratic_schedule(beta_start=1e-4, beta_end=3e-4, n_timesteps=5000):
    betas = torch.linspace(beta_start ** 0.5, beta_end ** 0.5, n_timesteps) ** 2
    beta0 = betas[:n_timesteps//2]
    return torch.cat([beta0, beta0.flip(0)])



def constant_schedule(beta_start=1e-4, beta_end=None, n_timesteps=5000):
    return torch.ones(n_timesteps) * beta_start