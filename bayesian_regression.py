import torch
import pyro 
import pyro.distributions as dist
from pyro.distributions import Normal, LogNormal
from pyro.optim import Adam 
from pyro.infer import MCMC, NUTS
import matplotlib.pyplot as plt
import seaborn as sns

# Generating sample data
data_X = torch.linspace(0, 100, 100, dtype=torch.int32)

true_slope = 1.5
true_y_intercept = 2

# relatively noisy?
data_y = true_slope * data_X + true_y_intercept * 1.75 * torch.randn(100)

def model(X, y):
    w = pyro.sample('w', Normal(loc=torch.tensor([0.0]), scale=torch.tensor([10.0])))
    b = pyro.sample('b', Normal(loc=torch.tensor([0.0]), scale=torch.tensor([10.0])))
    sigma = pyro.sample('sigma', fn=LogNormal(loc=torch.tensor([0.0]), scale=torch.tensor([0.25])))
    mean = X * w + b
    with pyro.plate('data', len(X)):
        pyro.sample('obs', Normal(loc=mean, scale=sigma), obs=y)


# Time to train


kernel = NUTS(model)
mcmc = MCMC(kernel=kernel, num_samples=600, warmup_steps=400)
mcmc.run(data_X, data_y)
mcmc.summary()
graph = pyro.render_model(model=model, model_args=(data_X, data_y))
graph.graph_attr.update(dpi='1000')
graph.format = 'svg'
graph.render('figs/bregression')