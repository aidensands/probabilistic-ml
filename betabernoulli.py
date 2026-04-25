import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
import jax.numpy as jnp
import jax.random as random


key = random.PRNGKey(42)
data = jnp.concat([jnp.ones(50), jnp.zeros(50)])
data = random.permutation(key, data)
print(len(data))



def Model(observations):

    # Prior on the probability of getting heads
    heads = numpyro.sample('heads', dist.Beta(1, 1))

    with numpyro.plate('data', len(observations)):
        numpyro.sample('obs', dist.Bernoulli(heads), obs=observations)


kernel = NUTS(model=Model)
mcmc = MCMC(kernel, num_warmup=500, num_samples=1000)
mcmc.run(key, data)
mcmc.print_summary()

numpyro.render_model(model=Model, model_args=(data,), filename='figs/betabernoulli.svg')