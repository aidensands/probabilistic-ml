import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
import seaborn as sns
import matplotlib.pyplot as plt
import arviz

rngkey = jax.random.key(0)

x = jnp.linspace(0, 100, 2000)
y = (1.5 * x + 40.0) + jax.random.uniform(rngkey, (2000,), minval=-15, maxval=15)


def BayesianRegressionModel(x, y):
    epsilon = numpyro.sample('epsilon', dist.Normal(35.0, 5))
    weights = numpyro.sample('weights', dist.Normal(0.0, 2.5))
    sigma = numpyro.sample('sigma', dist.Exponential(1.0))
    mu = (x * weights) + epsilon
    numpyro.sample('obs', dist.Normal(mu, sigma), obs=y)

kernel = NUTS(BayesianRegressionModel)
mcmc = MCMC(kernel, num_warmup=10, num_samples=2000)
mcmc.run(rng_key=rngkey, x=x, y=y)
mcmc.print_summary()

samples = mcmc.get_samples()
predictive_weight = samples['weights']
predictive_e = samples['epsilon']
predictive_error = samples['sigma']

regression_line = predictive_weight * x + predictive_e


