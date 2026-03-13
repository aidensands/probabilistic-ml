import torch
import pyro 
import pyro.distributions as dist
from pyro.distributions import Normal, LogNormal
from pyro.optim import Adam 
from pyro.infer import SVI, Trace_ELBO, MCMC, NUTS
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

def guide(X, y):
    w_loc = pyro.param('w_loc', torch.tensor([0.0]))
    w_scale = pyro.param('w_scale', torch.tensor([1.0]), constraint=dist.constraints.positive)

    b_loc = pyro.param('b_loc', torch.tensor([0.0]))
    b_scale = pyro.param('b_scale', torch.tensor([1.0]), constraint=dist.constraints.positive)

    sigma_loc = pyro.param('sigma_loc', torch.tensor([0.0]))
    sigma_scale = pyro.param('sigma_scale', torch.tensor([0.1]), constraint=dist.constraints.positive)

    pyro.sample('w', Normal(loc=w_loc, scale=w_scale))
    pyro.sample('b', Normal(loc=b_loc, scale=b_scale))
    pyro.sample('sigma', LogNormal(loc=sigma_loc, scale=sigma_scale))


# Time to train


svi = SVI(model=model, guide=guide, optim=Adam({'lr':0.005}), loss=Trace_ELBO())
losses = list()
steps = list()

for step in range(10000):
    loss = svi.step(data_X, data_y)
    steps.append(step)
    losses.append(loss)
    if step % 1000 == 0:
        print(f'Epoch: {step} ======================================')
        print(f'TraceELBO Loss: {loss}') 
    
w_loc   = pyro.param("w_loc").item()
w_scale = pyro.param("w_scale").item()
b_loc   = pyro.param("b_loc").item()
b_scale = pyro.param("b_scale").item()

print('Training complete: ')
print(f"\nw: mean={w_loc:.3f}, std={w_scale:.3f}  (true: {true_slope})")
print(f"b: mean={b_loc:.3f}, std={b_scale:.3f}  (true: {true_y_intercept})")

mean_line = data_X * w_loc + b_loc

slope_samples = dist.Normal(loc=w_loc, scale=w_scale).sample((50000,))
intercept_samples = dist.Normal(loc=b_loc, scale=b_scale).sample((50000,))

fig, ax = plt.subplots(2, 2, figsize=(10,8))

ax[0,0].set_title('Pre Fitting Data')
ax[0,0].scatter(data_X, data_y)
ax[0,1].set_title('Fitted Line')
ax[0,1].scatter(data_X, data_y)
ax[0,1].plot(data_X, mean_line, color='red')
ax[1,0].set_title('Posterior Mean of Slope')
sns.histplot(slope_samples, ax=ax[1,0], kde=True)
ax[1,1].set_title('Posterior Mean of Intercept')
sns.histplot(intercept_samples, ax=ax[1,1], kde=True)
plt.show()
plt.savefig('figs/bayesian_regression.png')

plt.plot(steps, losses)
plt.show()
