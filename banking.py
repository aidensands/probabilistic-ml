import jax
import jax.numpy as jnp
import bartz
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, SVI, Trace_ELBO, Predictive
from numpyro.infer.autoguide import AutoNormal
from numpyro.optim import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score, brier_score_loss, precision_recall_curve, f1_score
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import secrets

def preprocess(path):
    """Load csv data and clean/standardize the data"""

    df = pd.read_csv(path, sep=';')
    scaler = StandardScaler()
    interactor = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    # Encode job categories and save decoder
    df['job'] = df['job'].astype('category')
    job_cat_map = dict(enumerate(df['job'].cat.categories))
    df['job'] = df['job'].cat.codes

    # Encode labels and defaulting
    binary_mapper = {'no': 0, 'yes': 1}
    df['y'] = df['y'].map(binary_mapper)
    df['default'] = df['default'].map(binary_mapper)
    df['housing'] = df['housing'].map(binary_mapper)
    df['loan'] = df['loan'].map(binary_mapper)
    education = {'unknown': 0, 'primary': 1, 'secondary': 2, 'tertiary': 3}
    df['education'] = df['education'].map(education)
    marital_map = {'married': 0, 'single': 1, 'divorced': 2}
    df['marital'] = df['marital'].map(marital_map)

    # Sine-Cosine Encoding for Month
    month_mapping = {
    "jan": 0, "feb": 1, "mar": 2, "apr": 3, "may": 4, "jun": 5,
    "jul": 6, "aug": 7, "sep": 8, "oct": 9, "nov": 10, "dec": 11
    }
    df['month_idx'] = df['month'].map(month_mapping)
    df['month_sin'] = jnp.sin(2 * jnp.pi * df['month_idx'].to_numpy(dtype=int) / 12)
    df['month_cos'] = jnp.cos(2 * jnp.pi * df['month_idx'].to_numpy(dtype=int) / 12)

    df = df.drop(columns=['poutcome', 'duration', 'pdays', 'month', 'month_idx', 'contact'])

    labels = jnp.array(df['y'], dtype=jnp.int32)
    features = jnp.array(df.drop(columns=['y']))

    scaled_features = scaler.fit_transform(features)
    poly_scaled_features = interactor.fit_transform(scaled_features)

    X_train, X_test, y_train, y_test = train_test_split(poly_scaled_features, labels, test_size=0.2,  shuffle=True)
    return X_train, y_train, X_test, y_test

def generate_keys():
    """Generates a pseudo-random key that is safe from run to run using OS entropy"""
    entropy = secrets.randbits(32)
    master_prgkey = jax.random.PRNGKey(seed=entropy)
    return master_prgkey

def BayesianLogisticModel(X, y = None):

    num_features = X.shape[1]
    intercept = numpyro.sample('intercept', dist.Normal(0.0, 1.0))
    tau = numpyro.sample('tau', dist.HalfCauchy(1.0))

    with numpyro.plate('coefficient_plate', num_features):
        local_shrinkage = numpyro.sample('local_shrinkage', dist.HalfCauchy(1.0))
        raw_coefs = numpyro.sample('raw_coefs', dist.Normal(0.0, 1.0))
        coefs = numpyro.deterministic('coefs', tau * local_shrinkage * raw_coefs)
    
    logits = intercept + jnp.dot(X, coefs)
    numpyro.sample('obs', dist.Bernoulli(logits=logits), obs=y)

def inference(model, X, y, burn_in, samples, rngkey, chains=1):
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

    mcmc.run(rng_key=rngkey, X=X, y=y)
    mcmc.print_summary()

    return mcmc

def svi_inference(model, X, y, rngkey, steps):
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
        X=X,
        y=y
    )

    sns.lineplot(svi_results.losses)
    plt.xlabel('SVI Step')
    plt.ylabel('Trace ELBO Loss')
    plt.title('SVI Convergence')
    plt.show()

    return svi_results, guide

def predict_and_evaluate(model, X, y, rngkey, mcmc=None, svi_results=None, guide=None):
    
    if mcmc is not None:
        print('Inference Method: Markov Chain Monte Carlo with No U-Turn Sampler')
        posterior_samples = mcmc.get_samples()
        predictor = Predictive(
            model=model,
            posterior_samples=posterior_samples
        )
        posterior_logits = predictor(rng_key=rngkey, X=X)
        mean_scores = jnp.mean(posterior_logits['obs'], axis=0)
    elif svi_results is not None and guide is not None:
        print('Inference Method: Stochastic Variational Inference with Adam Optimizer')
        predictor = Predictive(
            model=model,
            guide=guide,
            params=svi_results.params,
            num_samples=1000
        )
        posterior_logits = predictor(rng_key=rngkey, X=X)
        mean_scores = jnp.mean(posterior_logits['obs'], axis=0)
    else:
        raise ValueError('Missing MCMC or SVI argument')

    print('Computing optimal threshold')
    precisions, recalls, thresholds = precision_recall_curve(y, mean_scores)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)


    optimal_threshold = thresholds[jnp.argmax(f1_scores)]
    print(f'Model threshold is: {optimal_threshold}')
    predictions = (mean_scores > optimal_threshold).astype(int)

    print(f'======================== Model Report ==========================')
    report = classification_report(y, predictions)
    print(report)
    roc_auc_report = roc_auc_score(y, mean_scores)
    print(f'ROC AUC Score: {roc_auc_report}')
    average_precision = average_precision_score(y, mean_scores)
    print(f'Average Precision Score: {average_precision}')
    brier = brier_score_loss(y, mean_scores)
    print(f'Brier Score: {brier}')

def main():
    X_train, y_train, X_test, y_test = preprocess('data/bank-full.csv')

    prngkey = generate_keys()

    svi_results, guide = svi_inference(
        model=BayesianLogisticModel,
        X=X_train,
        y=y_train,
        rngkey=prngkey,
        steps=50000
    )


    predict_and_evaluate(
        model=BayesianLogisticModel,
        X=X_test,
        y=y_test,
        rngkey=prngkey,
        svi_results=svi_results,
        guide=guide
    )



if __name__ == '__main__':
    main()