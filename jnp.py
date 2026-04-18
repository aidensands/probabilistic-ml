import numpyro
from numpyro import distributions as dist
from numpyro.infer import MCMC, NUTS
from numpyro.diagnostics import hpdi
import jax.numpy as jnp
from jax import random
from jax.scipy.special import logsumexp
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# Data Import and Cleaning
DATASET_URL = "https://raw.githubusercontent.com/rmcelreath/rethinking/master/data/WaffleDivorce.csv"
dataset = pd.read_csv(DATASET_URL, sep=";")

def standardize(x):
    return (x - x.mean()) / x.std()

dataset["AgeScaled"] = dataset.MedianAgeMarriage.pipe(standardize)
dataset["MarriageScaled"] = dataset.Marriage.pipe(standardize)
dataset["DivorceScaled"] = dataset.Divorce.pipe(standardize)

# This model should be able to take up to two parameters (age and marraige)
def RegressionModel(marraige=None, age=None, divorce=None):
    # Start by defining a distribution for the y-intercept
    a = numpyro.sample('a', dist.Normal(0.0, 0.2)) # The Y-intercept is normally distributed around 0 with small standard deviation
    M = 0.0
    A = 0.0

    if marraige is not None:
        bM = numpyro.sample('bM', dist.Normal(0.0, 0.5))
        M = bM * marraige
    if age is not None:
        bA = numpyro.sample('bA', dist.Normal(0.0, 0.5))
        A = bA * age
    sigma = numpyro.sample('sigma', dist.Exponential(1.0))

    mu = a + M + A

    numpyro.sample('obs', dist.Normal(mu, sigma), obs=divorce)

def plot_regression(x, y_mean, y_hdpi):
    idx = jnp.argsort(x)
    marriage = x[idx]
    mean = y_mean[idx]
    hdpi = y_hdpi[idx]
    divorce = dataset['DivorceScaled'].values[idx]
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 6))
    ax.plot(marriage, mean)
    ax.plot(marriage, divorce, "o")
    ax.fill_between(marriage, hpdi[0], hpdi[1], alpha=0.3, interpolate=True)
    return ax

if __name__ == '__main__':
    RNGKey = random.key(0)
    RNGKey, RNGKey = random.split(RNGKey)
    Kernel = NUTS(RegressionModel)
    mcmc = MCMC(
        sampler=Kernel,
        num_warmup=1000,
        num_samples=2000,
    )

    mcmc.run(RNGKey, marraige=dataset['MarriageScaled'].values, divorce=dataset['DivorceScaled'].values)
    mcmc.print_summary()
    samples_1 = mcmc.get_samples()