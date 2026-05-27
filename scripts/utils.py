import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, SVI, Trace_ELBO, Predictive
from numpyro.infer.autoguide import AutoNormal
from numpyro.optim import Adam
import equinox as eqx
from sklearn.metrics import (
    classification_report,
    brier_score_loss,
    average_precision_score,
    roc_auc_score,
    precision_recall_curve
)
import seaborn as sns
import matplotlib.pyplot as plt
import secrets


def mcmc_inference(model, X_test, y_test, burn_in, samples, rngkey, chains=1):
    """Begins Markov Chain Monte Carlo approximation of the posterior distribution"""
    kernel = NUTS(
        model=model
    )

    mcmc = MCMC(
        sampler=kernel,
        num_warmup=burn_in,
        num_samples=samples,
        num_chains=chains
    )

    mcmc.run(rng_key=rngkey, X=X_test, y=y_test)
    mcmc.print_summary()

    return mcmc

def svi_inference(model, X_train, y_train, rngkey, steps):
    """A faster posterior exploration using Stochastic Variational Inference"""

    guide = AutoNormal(model=model)
        
    optimizer = Adam(step_size=0.005)
    
    svi = SVI(
        model=model,
        guide=guide,
        optim=optimizer,
        loss=Trace_ELBO()
    )

    svi_results = svi.run(
        rng_key=rngkey,
        num_steps=steps,
        progress_bar=True,
        X=X_train,
        y=y_train
    )

    sns.lineplot(svi_results.losses)
    plt.xlabel('SVI Step')
    plt.ylabel('Trace ELBO Loss')
    plt.title('SVI Convergence')
    plt.show()

    return svi_results, guide

def predict_and_evaluate(model, X_test, y_test, rngkey, mcmc=None, svi_results=None, guide=None):
    
    if mcmc is not None:
        print('Inference Method: Markov Chain Monte Carlo with No U-Turn Sampler')
        posterior_samples = mcmc.get_samples()
        predictor = Predictive(
            model=model,
            posterior_samples=posterior_samples
        )
        posterior_logits = predictor(rng_key=rngkey, X=X_test)
        mean_scores = jnp.mean(posterior_logits['obs'], axis=0)
    elif svi_results is not None and guide is not None:
        print('Inference Method: Stochastic Variational Inference with Adam Optimizer')
        predictor = Predictive(
            model=model,
            guide=guide,
            params=svi_results.params,
            num_samples=1000
        )
        posterior_logits = predictor(rng_key=rngkey, X=X_test)
        mean_scores = jnp.mean(posterior_logits['obs'], axis=0)
    else:
        raise ValueError('Missing MCMC or SVI argument')

    print('Computing optimal threshold')
    precisions, recalls, thresholds = precision_recall_curve(y_test, mean_scores)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)


    optimal_threshold = thresholds[jnp.argmax(f1_scores)]
    print(f'Model threshold is: {optimal_threshold}')
    predictions = (mean_scores > optimal_threshold).astype(int)

    print(f'======================== Model Report ==========================')
    report = classification_report(y_test, predictions)
    print(report)
    roc_auc_report = roc_auc_score(y_test, mean_scores)
    print(f'ROC AUC Score: {roc_auc_report}')
    average_precision = average_precision_score(y_test, mean_scores)
    print(f'Average Precision Score: {average_precision}')
    brier = brier_score_loss(y_test, mean_scores)
    print(f'Brier Score: {brier}')

def generate_keys():
    """Generates a pseudo-random key that is safe from run to run using OS entropy"""
    entropy = secrets.randbits(32)
    master_prgkey = jax.random.PRNGKey(seed=entropy)
    return master_prgkey


def ARD_RBFKernel(x1, x2, lengthscales, variance):
    """kernel function for determining the similarity between two points"""
    squared_distance = jnp.sum(((x1[:, None, :] - x2[:, None, :]) / lengthscales) ** 2, axis=1)
    similarity = variance * jnp.exp(-0.5 * squared_distance)
    return similarity

def get_initial_inducing_points(X, M, key):
    N = X.shape[0]
    index = jax.random.choice(key=key, a=N, shape=(M,), replace=False)
    return X[index]

def sample_nam_params(module, prefix, guide=None):
    nodes, treedef = jax.tree_util.tree_flatten(module)
    sampled_leaves = []

    for i, node in enumerate(nodes):
        if eqx.is_array(node):
            name = f'{prefix}_{i}'

            if guide is not None:
                loc = numpyro.param(
                    f"{name}_loc", 
                    jnp.zeros_like(node)
                )
                scale = numpyro.param(
                    f"{name}_scale", 
                    jnp.ones_like(node) * 0.1,
                    constraint=dist.constraints.positive
                )
                sampled_leaves.append(
                    numpyro.sample(name, dist.Normal(loc, scale))
                )
            else:
                sampled_leaves.append(
                    numpyro.sample(
                        name=name,
                        fn=dist.Normal(0.0, 1.0).expand(node.shape)
                    )
                )
        else:
            sampled_leaves.append(node)
    
    reconstructed_tree = jax.tree_util.tree_unflatten(treedef, sampled_leaves)
    return reconstructed_tree