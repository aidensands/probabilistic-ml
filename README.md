# probabilistic-ml
A repository containing various probabilistic machine learning models that I learned and implemented in numpryo. Alongside basic model implementations there are also difficult datasets that I try to achieve good model performance on using a variety of models.

## Beta-Bernoulli Model
This is a model I implemented while I was reading a textbook about probabilistic ML. I wanted to implement a model that would infer the probability that an event happens (a success) given a series of observations. The easiest demonstration of this is a coin flip where we infer the probability of heads. We already know this but its a really simple demonstration of how numpyro works and I like it. 


## Bayesian Regression
The first model implemented. Takes some randomly generated correlated points and attempts to a find a line for which the points are most probably generated. There are two versions of this, the first is data that comes from syntheic line data and the other one is a students t distribution that is supposed to be used to detect outliers such as students who cheated on an exam or students who underperformed despite studying heavily. I recently started to update these to only use MCMC and NUTS

## Dirichlet-Categorical Model
This model is really similar to the beta-bernoulli model that was first in the list. In fact its more like a higher dimensional generalization of the model. Rather than inferring the probability of getting heads on an unfair coin, we can have more than just one outcome by using the categorical distribution (multinoulli). As such, its important the the underlying priors are also multi-dimensional which is why we use the Dirichlet distribution as a conjugate prior for the categorical distribution.

## Gaussian Mixture Model
Assuming that our data was generated from many different gaussians, we can treat this sort of like a clustering problem. The difference between this model and a classical KMM classifier is that we will also know how confident the model is with its clustering. I updated this model again recently to explore the entire posterior distribution using Markov Chain Monte Carlo (MCMC) approximation and the No U-Turn Sampler (NUTS). I found it to be very useful to set informative priors using K-Means. Although they were neververy close to the true means they got in the ballpark and MCMC fine tuned it. 


## Hidden Markov Model
This is a classic example that tries to learn the weather from the observed activity of an individual on particular days. 

## PIMA Diabetes Dataset
This is a dataset containing hospital patients that either do or do not have diabetes. Each patient has a number of different medical indicators recorded in the csv. The model that worked for me when classifying this was a simple Bayesian logistic regression model. It utilizes Bayesian variable selection for a marginal performance gain. 

It achieves a ROC AUC of 0.89 which beats the Scikit-learn Random Forest Classifier which gets a ROC AUC of 0.75 with 100 learners and no maximum depth. 


## UCI Banking Dataset 
This is an ongoing project where i'm trying to implement my own models to achieve competetive scores on the UCI Banking dataset where we classify subscribers from non-subscribers. This dataset is challenging because only around 11% of people actually subscribe. There is a lot of preprocessing that needs to be done here as well. There will be multiple approaches to this problem:
### Bayesian Logistic Regression
This model struggles to deal with the challenges of the banking dataset and was only able to achieve a maximum ROC AUC of 0.70. This is with many enhancements to the model. It uses Bayesian variable selection as well as introduces nonlinearities into the dataset. 