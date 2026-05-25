import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, SVI, Trace_ELBO, Predictive
from numpyro.infer.autoguide import AutoNormal
import jax
import jax.numpy as jnp
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier
import arviz as az
import matplotlib.pyplot as plt


def prep_data(path:str, features, label='Outcome'):
    """Loads and scales comma seperated data"""

    df = pd.read_csv(path)
    scaler = StandardScaler()
    X = jnp.array(df[features].fillna(0).values)
    y = jnp.array(df[label].fillna(0).values)
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)

    return X_train, y_train, X_test, y_test

def RFC(X_train, y_train, X_test, y_test):
    rfc = RandomForestClassifier()
    rfc.fit(X_train, y_train)
    preds = rfc.predict(X_test)
    report = classification_report(y_test, preds)
    print('Report for Random Forest Classifier')
    print(report)
    auc_roc = roc_auc_score(y_test, preds)
    print(f'ROC AUC: {auc_roc}')


def LogisticBayes(X, y=None):
    """Logistic Bayesian Model that Utilizes Bayesian Variable Selection to Prune Unimportant Data"""

    features = X.shape[1]
    intercept = numpyro.sample('intercept', dist.Normal(0.0, 10.0))
    tau = numpyro.sample('tau', dist.HalfCauchy(1.0))

    with numpyro.plate('coef_plate', features):
        local_shrinkage = numpyro.sample('local_shrinkage', dist.HalfCauchy(1.0))
        beta = numpyro.sample('beta', dist.Normal(0.0, 5.0))
        coefs = numpyro.deterministic('coefs', beta * tau * local_shrinkage)

    logits = intercept + jnp.dot(X, coefs)
    numpyro.sample('obs', dist.Bernoulli(logits=logits), obs=y)
    

def MCMC_inference(X, y, burn_in, samples, chains=1, key=None):
    kernel = NUTS(model=LogisticBayes)
    mcmc = MCMC(\
        sampler=kernel,
        num_warmup=burn_in,
        num_samples=samples,
        num_chains=chains
    )
    mcmc.run(
        rng_key=key,
        X=X,
        y=y
    )
    mcmc.print_summary()
    return mcmc

def ppc(model, mcmc:MCMC, X_test, key):
    posterior_samples = mcmc.get_samples()
    predictor = Predictive(model=model, posterior_samples=posterior_samples)
    pred_draws = predictor(rng_key=key, X=X_test)
    treedata = az.from_numpyro(mcmc, posterior_predictive=pred_draws)
    az.plot_ppc_pava(treedata)
    plt.show()

def calibration(model, y_test, y_logits):
    prob_true, prob_pred = calibration_curve(y_test, y_logits)
    plt.plot(prob_pred, prob_true)
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives (True Label)')
    plt.title('Probability Calibration Curve')
    plt.show()

def predictive_draws(model, mcmc:MCMC, X_test, key):
    posterior_samples = mcmc.get_samples()
    predictor = Predictive(model=model, posterior_samples=posterior_samples)
    pred_draws = predictor(key, X=X_test)
    scores = jnp.mean(pred_draws['obs'], axis=0)
    print(f'Returning {len(scores)} scores')
    return scores

def score_metrics(probs, y_true, name:str):
    print(f'Report for {name}:')
    y_preds = (probs > 0.5).astype(int)
    results = classification_report(y_true, y_preds)
    print(results)
    roc_auc = roc_auc_score(y_true, probs)
    print(f'ROC AUC: {roc_auc}')

def summary_plots(mcmc:MCMC, feature_names):
    treedata = az.from_numpyro(mcmc, coords={'features':feature_names})
    az.plot_forest(treedata)
    plt.show()

def main():
    features = ['Pregnancies', 'Glucose', 'BloodPressure', 'Insulin', 'BMI', 'DiabetesPedigreeFunction']
    prngkey = jax.random.PRNGKey(0)
    X_train, y_train, X_test, y_test = prep_data('data/diabetes.csv', features=features)
    mcmc_results = MCMC_inference(X_train, y_train, 500, 1000, 1, key=prngkey)
    pred_probs = predictive_draws(
        model=LogisticBayes,
        mcmc=mcmc_results,
        X_test=X_test,
        key=prngkey
    )
    score_metrics(probs=pred_probs, y_true=y_test, name='Numpyro Logistic Bayesian Regression')
    RFC(X_train, y_train, X_test, y_test)
    summary_plots(mcmc_results, feature_names=features)
    ppc(LogisticBayes, mcmc_results, X_test, prngkey)
    calibration(LogisticBayes, y_test, pred_probs)

if __name__ == '__main__':
    main()