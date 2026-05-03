import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
import jax
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
import seaborn as sns
import matplotlib.pyplot as plt


def TRegressionModel(x, y):

    w_prior = numpyro.sample('w1_prior', dist.Normal(0, 10))
    beta_pior = numpyro.sample('beta_prior', dist.Normal(0, 10))
    sigma = numpyro.sample('sigma', dist.Exponential(1))

    observation = w_prior * x + beta_pior

    with numpyro.plate('observations', len(x)):
        numpyro.sample('obs', dist.StudentT(3, observation, sigma), obs=y)

if __name__ == '__main__':
    
    X, y = make_regression(500, 1, noise=10, tail_strength=0.8, random_state=42)

    X = X.squeeze()

    sns.scatterplot(x=X, y=y)
    plt.show()

    rngkey = jax.random.PRNGKey(42)
    kernel = NUTS(TRegressionModel)
    mcmc = MCMC(
        sampler=kernel,
        num_samples=1000,
        num_warmup=500
    )
    mcmc.run(rngkey, X, y)
    mcmc.print_summary()

    X = X.reshape(-1, 1)
    regression_model = LinearRegression()
    regression_model.fit(X, y)

    