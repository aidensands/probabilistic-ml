import torch
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO, Predictive
from pyro.optim import Adam 
from pyro.infer.autoguide import AutoNormal
import matplotlib.pyplot as plt
import seaborn as sns

N = 200
X = torch.linspace(-3, 3, N).squeeze(-1)
y = torch.sin(X.unsqueeze()) + 0.3 * torch.randn(N)

split = 160
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[:split]

def bnn(X, y=None):
    input_dim = X.shape[1]
    hidden_dimensions = 16


    # Network architecture (3 layers with 16 neurons each)

    w1 = pyro.sample('w1', dist.Normal(
        loc=torch.zeros(input_dim, hidden_dimensions),
        scale=torch.ones(input_dim, hidden_dimensions)
    ))

    b1 = pyro.sample('b1', dist.Normal(
        loc=torch.zeros(input_dim, hidden_dimensions),
        scale=torch.ones(hidden_dimensions, hidden_dimensions)
    ))

    w2 = pyro.sample('w2', dist.Normal(
        loc=torch.zeros(hidden_dimensions, hidden_dimensions),
        scale=torch.ones(hidden_dimensions, hidden_dimensions)
    ))

    b2 = pyro.sample('b2', dist.Normal(
        loc=torch.zeros(hidden_dimensions, hidden_dimensions),
        scale=torch.ones(hidden_dimensions, hidden_dimensions)
    ))

    w3 = pyro.sample('w3', dist.Normal(
        loc=torch.zeros(hidden_dimensions, 1),
        scale=torch.ones(hidden_dimensions, 1)
    ))

    b3 = pyro.sample('b3', dist.Normal(
        loc=torch.zeros(1, 1),
        scale=torch.ones(1,1)
    ))

    # Forward Pass
    h1 = torch.tanh(X @ w1 + w2)
    h2 = torch.tanh(h1 @ w2 + b2)
    mu = torch.tanh(h2 @ w3 + b3).unsqueeze(-1)

    sigma = pyro.sample('sigma', dist.Exponential(1.0))

    with pyro.plate('obs', X.shape[0]):
        pyro.sample('obs', dist.Normal(mu, sigma), obs=y)
    
    return mu