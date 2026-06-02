import numpyro
import numpyro.distributions as dist
import jax.numpy as jnp

from scripts.utils import ARD_RBFKernel

def SparseGaussianProcess(X, inducing, y=None):
    """
    A Classifier model which uses a Sparse Gaussian Process.
    X: a feature matrix containing feature vectors
    y: a label matrix containing labels
    inducing: a subsampled set of inducing points which are representative of the dataset
    """

    N, D = X.shape
    M = inducing.shape[0]

    variance = numpyro.sample('kernel_variance', dist.LogNormal(0.0, 1.0))
    lengthscales = numpyro.sample('kernel_lengthscales', dist.LogNormal(jnp.zeros(D), jnp.ones(D)))

    K_ZZ = ARD_RBFKernel(inducing, inducing, lengthscales, variance) + jnp.eye(M) * 1e-4
    u = numpyro.sample('u', dist.MultivariateNormal(jnp.zeros(M), covariance_matrix=K_ZZ))

    K_XZ = ARD_RBFKernel(X, inducing, lengthscales, variance)
    K_XX_DIAG = jnp.full(X.shape[0], variance)

    K_ZZ_INV = jnp.linalg.inv(K_ZZ)
    A = jnp.linalg.matmul(K_XZ, K_ZZ_INV)
    mu_f = jnp.linalg.matmul(A, u)
    variance_f = K_XX_DIAG - jnp.sum(A * K_XZ, axis=1)

    f = numpyro.sample('function', dist.Normal(mu_f, jnp.sqrt(variance_f)))

    with numpyro.plate('data', N):
        numpyro.sample('obs', dist.Bernoulli(logits=f), obs=y)
    
