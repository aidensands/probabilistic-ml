# probabilistic-ml

## What is this repo?
I made this during a summer of my undergrad at the University of Connecticut as a way of tracking my progress in learning Bayesian and probabilistic machine learning. Its a good demonstration of several simple to complex models and more than anything its just a showcase of some projects. If you're trying to learn how to use JAX or Numpyro this is also a good demonstration for how to use those libraries.

## Currently Implemented Models:
**Bayesian Logistic Regression**

This is a simple logistic model that can take any number of features from tabular data. It performs well on many datasets but can often be lacking for more complex challenging datasets as you will see in the demonstrations. This model does have a small twist to it, it uses Bayesian variable selection to intuitively cut through irrelevant features that are not informative of the labels.

**Sparse Gaussian Process Classifier**

**Bayesian Neural Network**

This is a logical next step coming after logistic regression. I think its important to preface that uncertainty on a black box model like a neural network is not often useful. Despite this I'm still building one for the love of the game using Equinox as a library that is directly integrated with JAX. Its currently a WIP so there will be more to read here later as I am wrestling with how equinox handles matrix multiplication.


## Benchmarking
The dataset that I chose for this project was the UCI Banking dataset as it presents many challenges for classifiers and data preprocessing. First off only 11% of individuals will actually subscribe making the number of positive examples very low, also creating the possibility for class imbalance. Second is that there are many features that can be encoded in many ways, allowing for some creativity with preprocessing.