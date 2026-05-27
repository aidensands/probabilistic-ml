import jax
import jax.numpy as jnp
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
import pandas as pd
import functools

# My Models and utilities
from models.logistic import BayesianLogisticModel
from models.nam import BayesianNAM, SubNet
from models.bnn import BayesianNeuralNetwork, NeuralNetwork
from scripts.utils import mcmc_inference, svi_inference, generate_keys, predict_and_evaluate

def preprocess(path, polynomial_interactions=False):
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

    # Encode labels and defaulting
    binary_mapper = {'no': 0, 'yes': 1}
    df['y'] = df['y'].map(binary_mapper)
    df['default'] = df['default'].map(binary_mapper)
    df['housing'] = df['housing'].map(binary_mapper)
    df['loan'] = df['loan'].map(binary_mapper)

    df = pd.get_dummies(df, columns=categorical_columns, dtype=int)

    # Sine-Cosine Encoding for Month
    month_mapping = {
    "jan": 0, "feb": 1, "mar": 2, "apr": 3, "may": 4, "jun": 5,
    "jul": 6, "aug": 7, "sep": 8, "oct": 9, "nov": 10, "dec": 11
    }
    df['month_idx'] = df['month'].map(month_mapping)
    df['month_sin'] = jnp.sin(2 * jnp.pi * df['month_idx'].to_numpy(dtype=int) / 12)
    df['month_cos'] = jnp.cos(2 * jnp.pi * df['month_idx'].to_numpy(dtype=int) / 12)


    df = df.drop(columns=['duration', 'pdays', 'month', 'month_idx', 'contact'])

    labels = df['y']
    features = df.drop(columns=['y'])

    scaled_features = scaler.fit_transform(features)

    if polynomial_interactions:
        poly_scaled_features = interactor.fit_transform(scaled_features)
        X_train, X_test, y_train, y_test = train_test_split(poly_scaled_features, labels, test_size=0.2,  shuffle=True)

    X_train, X_test, y_train, y_test = train_test_split(scaled_features, labels, test_size=0.2,  shuffle=True)

    print(df.head(n=10))

    X_train = jnp.array(X_train)
    X_test = jnp.array(X_test)
    y_train = jnp.array(y_train)
    y_test = jnp.array(y_test)

    return X_train, y_train, X_test, y_test

def main():
    X_train, y_train, X_test, y_test = preprocess('data/bank-full.csv')
    # Debug: print shapes and a tiny sample to catch transposition/dtype issues
    prngkey = generate_keys()

    svi_results, guide = svi_inference(
        model=BayesianLogisticModel,
        X_train=X_train,
        y_train=y_train,
        rngkey=prngkey,
        steps=10000
    )

    predict_and_evaluate(
        model=BayesianLogisticModel,
        X_test=X_test,
        y_test=y_test,
        rngkey=prngkey,
        svi_results=svi_results,
        guide=guide
    )

if __name__ == '__main__':
    main()