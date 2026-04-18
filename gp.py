import numpyro
import numpyro.distributions as dist
import jax.numpy as jnp

def RBFKernel(X, Z, lengthscale, variance):
    sqdist = jnp.sum(X ** 2, 1).reshape(-1, 1) + jnp.sum(Z**2, 1) - 2 * jnp.dot(X, Z.T)
    return variance * jnp.exp(-0.5 / lengthscale ** 2 * sqdist)

def GaussianProcessModel(X, Y = None):
    N = X.shape()