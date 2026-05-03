import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive
import jax.numpy as jnp
import jax.random as random
import seaborn as sns
import matplotlib.pyplot as plt

key = random.PRNGKey(42)
data = jnp.concat([jnp.ones(70), jnp.zeros(30)])
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

posterior_samples = mcmc.get_samples()
predictor = Predictive(model=Model, posterior_samples=posterior_samples)
generated_flips = predictor(key, data)
print(generated_flips)


numpyro.render_model(model=Model, model_args=(data,))