import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
import jax
import jax.numpy as jnp
from sklearn.feature_extraction.text import CountVectorizer
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


def data_prep(path):

    df = pd.read_csv(path, sep='\t')
    df.columns = ['ham/spam', 'content']
    labels = list()

    for value in df['ham/spam']:
        if value == 'spam':
            labels.append(1)
        else:
            labels.append(0)

    vectorizer = CountVectorizer(
        max_features=1000,
        binary=True,
        min_df=1
    )

    X = vectorizer.fit_transform(df['content']).toarray()
    X = jnp.array(X)
    Y = jnp.array(labels)

    return X, Y

def NaiveBayesClassifier(features, labels):

    n_emails, n_features = features.shape


    with numpyro.plate('word_plate', n_features):
        p_word_given_spam = numpyro.sample('word|spam', dist.Beta(1,1))
        p_word_given_ham = numpyro.sample('word|ham', dist.Beta(1,1))
    
    p_words = jnp.where(
        labels[:, None] == 1,
        p_word_given_spam,
        p_word_given_ham
    )

    numpyro.sample('obs', dist.Bernoulli(p_words), obs=features)


def inference(model, burn_in, samples, features, labels) -> None:
    """Starts model fitting using Markov Chain Monte Carlo methods, uses the NUTS kernel by default"""
    rngkey = jax.random.PRNGKey(42)
    kernel = NUTS(
        model=model
    )
    mcmc = MCMC(
        sampler=kernel,
        num_samples=samples,
        num_warmup=burn_in
    )
    mcmc.run(rngkey, features, labels)
    mcmc.print_summary()
    


def main():
    feature_counts, labels = data_prep('data/spamhamdata.csv')
    inference(NaiveBayesClassifier, 500, 1000, feature_counts, labels)

if __name__ == '__main__':
    main()