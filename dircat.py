import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

import jax
import jax.numpy as jnp

def dice_roll(n, k):
    """Simulates n trials of rolling a k sided die, O(n)"""
    rngkey = jax.random.key(0)
    rolls = jax.random.randint(rngkey, (n,), 0, k)
    return rolls

def DirichletCategoricalModel(observations, k):

    prior = numpyro.sample('prior', dist.Dirichlet(jnp.ones(k)))

    with numpyro.plate('data_plate', len(observations)):
        numpyro.sample('data', dist.Categorical(prior), obs=observations)

def inference(model, burn_in, samples, data):
    rngkey = jax.random.key(0)
    kernel = NUTS(model=model)
    mcmc = MCMC(
        sampler=kernel,
        num_warmup=burn_in,
        num_samples=samples
    )
    mcmc.run(rngkey, data, 12)
    mcmc.print_summary()

trial_data = dice_roll(1000, 11)
inference(DirichletCategoricalModel, 500, 1000, trial_data)