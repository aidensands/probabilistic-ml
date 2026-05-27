import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist


def BayesianLogisticModel(X, y = None):
    """
    Numpyro model definition for Bayesian logistic regression. Utilizes Bayesian
    variable selection and is can take as many features as needed.
    """
    num_features = X.shape[1]
    intercept = numpyro.sample('intercept', dist.Normal(0.0, 1.0))
    tau = numpyro.sample('tau', dist.HalfCauchy(1.0))

    with numpyro.plate('coefficient_plate', num_features):
        local_shrinkage = numpyro.sample('local_shrinkage', dist.HalfCauchy(1.0))
        raw_coefs = numpyro.sample('raw_coefs', dist.Normal(0.0, 1.0))
        coefs = numpyro.deterministic('coefs', tau * local_shrinkage * raw_coefs)
    
    logits = intercept + jnp.dot(X, coefs)
    numpyro.sample('obs', dist.Bernoulli(logits=logits), obs=y)