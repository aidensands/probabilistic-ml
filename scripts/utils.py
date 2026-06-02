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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.compose import ColumnTransformer
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import secrets


def mcmc_inference(model, X_train, y_train, burn_in, samples, rngkey, chains=1, group_codes=None, inducing=None):
    """
    Begins Markov Chain Monte Carlo approximation of the posterior distribution.

    """
    kernel = NUTS(
        model=model
    )

    mcmc = MCMC(
        sampler=kernel,
        num_warmup=burn_in,
        num_samples=samples,
        num_chains=chains
    )

    if group_codes is not None:
        mcmc.run(
            rng_key=rngkey,
            X=X_train,
            y=y_train,
            groups=group_codes,
            num_groups=int(jnp.max(group_codes) + 1)
        )
    elif inducing is not None:
        mcmc.run(
            rng_key=rngkey,
            X=X_train,
            y=y_train,
            inducing=inducing
        )
    else:
        mcmc.run(rng_key=rngkey, X=X_train, y=y_train)

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

def predict_and_evaluate(model, X_test, y_test, rngkey, mcmc=None, svi_results=None, guide=None, groups=None, inducing=None):
    
    if mcmc is not None:
        print('Inference Method: Markov Chain Monte Carlo with No U-Turn Sampler')
        posterior_samples = mcmc.get_samples()
        predictor = Predictive(
            model=model,
            posterior_samples=posterior_samples
        )

        if groups is not None:
            posterior_logits = predictor(rng_key=rngkey, X=X_test, groups=groups, num_groups=int((jnp.max(groups) + 1)))
        elif inducing is not None:
            posterior_logits = predictor(rng_key=rngkey, X=X_test, groups=groups, inducing=inducing)
        else:
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
    """Generation of pseudo-random key using 32 bits of OS entropy"""
    entropy = secrets.randbits(32)
    master_prgkey = jax.random.PRNGKey(seed=entropy)
    return master_prgkey


def ARD_RBFKernel(x1, x2, lengthscales, variance):
    """kernel function for determining the similarity between two points"""
    squared_distance = jnp.sum(((x1[:, None, :] - x2[None, :, :]) / lengthscales) ** 2, axis=-1)
    similarity = variance * jnp.exp(-0.5 * squared_distance)
    return similarity

def get_initial_inducing_points(X, M, key):
    N = X.shape[0]
    index = jax.random.choice(key=key, a=N, shape=(M,), replace=False)
    return X[index]


def preprocess(path):
    """Load csv data and clean/standardize the data
        all binary data is encoded with binary integers
        all categorical data is one hot encoded with the exception of months
        month is encoded using sine cosine encoding
    """

    df = pd.read_csv(path, sep=';')
    scaler = StandardScaler()
    interactor = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    # Encode job categories and save decoder

    categorical_columns = ['job', 'education', 'marital', 'poutcome']
    continuous_columns = ['age', 'balance', 'day', 'campaign', 'previous']
    continuous_indexes = [df.columns.get_loc(c) for c in continuous_columns]

    ct = ColumnTransformer(
        transformers=[('num', scaler, continuous_indexes)],
        remainder='passthrough'
    )

    # Encode labels and defaulting
    binary_mapper = {'no': 0, 'yes': 1}
    df['y'] = df['y'].map(binary_mapper)
    df['default'] = df['default'].map(binary_mapper)
    df['housing'] = df['housing'].map(binary_mapper)
    df['loan'] = df['loan'].map(binary_mapper)

    job_cats = df['job'].unique()
    job_encoder = {job: i for i, job in enumerate(job_cats)}
    job_codes = df['job'].map(job_encoder).to_numpy(dtype=int)
    df = pd.get_dummies(df, columns=categorical_columns, dtype=int)

    # Sine-Cosine Encoding for Month
    month_mapping = {
    "jan": 0, "feb": 1, "mar": 2, "apr": 3, "may": 4, "jun": 5,
    "jul": 6, "aug": 7, "sep": 8, "oct": 9, "nov": 10, "dec": 11
    }
    df['month_idx'] = df['month'].map(month_mapping)
    df['month_sin'] = jnp.sin(2 * jnp.pi * df['month_idx'].to_numpy(dtype=int) / 12)
    df['month_cos'] = jnp.cos(2 * jnp.pi * df['month_idx'].to_numpy(dtype=int) / 12)


    df = df.drop(columns=['pdays', 'month', 'month_idx', 'contact', 'duration'])

    labels = df['y']
    features = df.drop(columns=['y'])

    X_train_raw, X_test_raw, y_train, y_test, g_train, g_test = train_test_split(
        features,
        labels,
        job_codes,
        test_size=0.2,
        shuffle=True,
        stratify=labels,
    )

    X_train = ct.fit_transform(X_train_raw)
    X_test = ct.transform(X_test_raw)

    print(df.head(n=10))

    X_train = jnp.array(X_train)
    X_test = jnp.array(X_test)
    y_train = jnp.array(y_train)
    y_test = jnp.array(y_test)
    g_train = jnp.array(g_train)
    g_test = jnp.array(g_test)

    return X_train, y_train, X_test, y_test, g_train, g_test