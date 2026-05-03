import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpyro
import numpyro.distributions as dist
import seaborn as sns
from numpyro.infer import MCMC, NUTS


def Model(x, y):
    """
    polynomial regression model. learns the parameters of the polynomial function
    """
    # Priors
    c1 = numpyro.sample("c1", dist.Normal(0, 10))
    c2 = numpyro.sample("c2", dist.Normal(0, 10))
    c3 = numpyro.sample("c3", dist.Normal(0, 10))
    sigma = numpyro.sample("sigma", dist.Exponential(1))

    # Observation

    mu = c1 * (x**2) + c2 * x + c3

    with numpyro.plate("observation_plate", len(y)):
        numpyro.sample("observations", dist.Normal(mu, sigma), obs=y)


def inference(model, x, y):
    print("Generating PRNG Key...")
    rngkey = jax.random.PRNGKey(42)
    kernel = NUTS(model=model)
    mcmc = MCMC(sampler=kernel, num_samples=1000, num_warmup=500)
    print("Starting Markov Chain Monte Carlo...")
    mcmc.run(rngkey, x, y)
    mcmc.print_summary()


def generate_data(c1: int, c2: int, c3: int, n_points: int):
    """Generates second degree polynomial regression data with n data points
    data is in the form of c1x^2+c2x+c3
    """
    x = jnp.linspace(-100, 100, n_points)
    y = c1 * (x**2) + c2 * x + c3

    return x, y


def main():
    X, y = generate_data(3, 1, 4, 100)
    sns.scatterplot(x=X, y=y)
    plt.show()
    inference(model=Model, x=X, y=y)


if __name__ == "__main__":
    main()
