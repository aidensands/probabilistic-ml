import jax.numpy as jnp
import jax.random as random
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
import arviz
import seaborn as sns
import matplotlib.pyplot as plt

RNGKey = random.key(0)

# Sample data generation
sample1 = random.normal(RNGKey, (10,))
sample2 = random.normal(RNGKey, (10,)) * 5
data = sample1 + sample2
# Checking for correctness
sns.kdeplot(data=data)
plt.show()