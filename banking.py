import argparse
# My Models and utilities
from models.logistic import LogisticBayes, HierarchicalLogisticBayes
from models.sgp import SparseGaussianProcess
from scripts.utils import mcmc_inference, svi_inference, generate_keys, predict_and_evaluate, preprocess, get_initial_inducing_points

def main():
    X_train, y_train, X_test, y_test, g_train, g_test = preprocess('data/bank-full.csv')
    # Debug: print shapes and a tiny sample to catch transposition/dtype issues
    prngkey = generate_keys()

    inducing_points = get_initial_inducing_points(X_train, 100, prngkey)

    mcmc = mcmc_inference(
        model=SparseGaussianProcess,
        X_train=X_train,
        y_train=y_train,
        burn_in=500,
        samples=1000,
        rngkey=prngkey,
        inducing=inducing_points
    )

    predict_and_evaluate(
        model=SparseGaussianProcess,
        X_test=X_test,
        y_test=y_test,
        rngkey=prngkey,
        mcmc=mcmc,
        inducing=inducing_points
    )

if __name__ == '__main__':

    """
        parser = argparse.ArgumentParser(description='A command line utility for probabilistic model demos')
        parser.add_argument('model', help='Model selection flag, allows you to choose what model will be used for classification', choices=['LogisticBayes', 'HierarchichalLogistcBayes'], required=True)
        parser.add_argument('inference-method', help='Inference Selection algorithm between MCMC + NUTS and SVI', choices=['MCMC', 'SVI'], required=True)
    """

    main()