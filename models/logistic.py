import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist


def _class_balanced_weights(y):
    positive_count = jnp.sum(y)
    negative_count = y.shape[0] - positive_count
    positive_weight = negative_count / (positive_count + 1e-8)
    return jnp.where(y > 0, positive_weight, 1.0)


def LogisticBayes(X, y = None):
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
    obs_dist = dist.Bernoulli(logits=logits)

    if y is None:
        numpyro.sample('obs', obs_dist)
    else:
        weights = _class_balanced_weights(y)
        numpyro.factor('weighted_obs', jnp.sum(weights * obs_dist.log_prob(y)))

def HierarchicalLogisticBayes(X, y=None, groups=None, num_groups=None):
    """Hierarcichal Bayesian Logistic Regression Model"""
    N = X.shape[1] # Feature count
    M = X.shape[0]
    
    with numpyro.plate('beta_plate', N):
        beta = numpyro.sample('beta', dist.Normal(0.0, 1.0))

    group_sigma = numpyro.sample('sigma', dist.HalfNormal(1.0))

    with numpyro.plate('group_plate', num_groups):
        alpha = numpyro.sample(f'alpha', dist.Normal(0.0, group_sigma))

    logits = jnp.dot(X, beta) + alpha[groups]
    obs_dist = dist.Bernoulli(logits=logits)

    if y is None:
        numpyro.sample('obs', obs_dist)
    else:
        weights = _class_balanced_weights(y)
        numpyro.factor('weighted_obs', jnp.sum(weights * obs_dist.log_prob(y)))