import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
import arviz
import seaborn as sns
import matplotlib.pyplot as plt

key = jax.random.PRNGKey(0)
# Split the key to get two independent subkeys
key1, key2 = jax.random.split(key, 2)

# Generate and transform the distributions
# Formula: (Standard Normal * std_dev) + mean
dist1 = jax.random.normal(key1, (1000,)) * 1.0 + 0.0  # μ=0, σ=1
dist2 = jax.random.normal(key2, (1000,)) * 2.0 + 5.0  # μ=5, σ=2
data = jnp.concatenate([dist1, dist2])
# Checking for correctness
sns.kdeplot(data=data)
plt.show()

def inference(model, burn_in, samples):
    kernel = NUTS(model=model)
    mcmc = MCMC(
        sampler=kernel,
        num_warmup=burn_in,
        num_samples=samples
    )
    mcmc.run(key, data, 2)
    mcmc.print_summary()

def GMM(data, k):
    
    # Mixture weights
    mixture_weights = numpyro.sample('mixture_weights', dist.Dirichlet(jnp.ones(k)))

    # Priors over the K gaussians
    with numpyro.plate('priors', k):
        mu = numpyro.sample('mu', dist.Normal(0, 10))
        sigma = numpyro.sample('sigma', dist.LogNormal(0, 2.5))
    
    with numpyro.plate('observations', len(data)):
        choice = numpyro.sample('choice', dist.Categorical(mixture_weights))
        numpyro.sample('obs', dist.Normal(mu[choice], sigma[choice]), obs=data)


if __name__ == '__main__':
    inference(GMM, 500, 1000)
