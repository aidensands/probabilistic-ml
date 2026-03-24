import torch
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, TraceEnum_ELBO, config_enumerate
from pyro.optim import Adam
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

K = 3
T = 1000 

true_means  = torch.tensor([8.0,  4.5,  1.5])   # mean activity per state
true_scales = torch.tensor([1.0,  1.2,  0.8])   # spread in activity


true_trans = torch.tensor([
    [0.70, 0.20, 0.10],   
    [0.30, 0.60, 0.10],   
    [0.05, 0.15, 0.80],   
])
true_init = torch.tensor([0.5, 0.3, 0.2]) 

states, obs = [], []
z = dist.Categorical(true_init).sample()

for t in range(T):
    states.append(z.item())
    activity = dist.Normal(true_means[z], true_scales[z]).sample()
    obs.append(activity.item())
    z = dist.Categorical(true_trans[z]).sample()

data = torch.tensor(obs)
true_states = torch.tensor(states)

state_names = ["Sunny", "Cloudy", "Rainy"]
print("First 10 days:")
for t in range(10):
    print(f"  Day {t+1:3d}: {state_names[true_states[t]]:7s} → activity score {data[t]:.2f}")

@config_enumerate
def Model(scores):
    
    T_observations = scores.shape[0]

    with pyro.plate('weather_states', K):
        means = pyro.sample('means', dist.Normal(loc=5.0, scale=1.0))
        scale = pyro.sample('scales', dist.LogNormal(loc=0.0, scale=1.5))
    
    pi = pyro.sample('pi', dist.Dirichlet(torch.ones(K)))

    with pyro.plate('from_state', K):
        trans = pyro.sample('trans', dist.Dirichlet(torch.ones(K)))

    z = None

    for t in pyro.markov(range(T_observations)):
        if t == 0:
            z = pyro.sample(f'z_{t}', dist.Categorical(pi), infer={"enumerate": "parallel"})
        else:
            z = pyro.sample(f'z_{t}', dist.Categorical(trans[z]), infer={"enumerate": "parallel"})

        pyro.sample(
            f'activity_{t}',
            dist.Normal(means[z], scale[z]),
            obs=scores[t]
        )

def guide(scores):
    # Variational parameters for emission means
    mean_loc   = pyro.param("mean_loc",   torch.tensor([7.0, 4.0, 2.0]))
    mean_scale = pyro.param("mean_scale", torch.ones(K),
                            constraint=dist.constraints.positive)

    # Variational parameters for emission scales
    scale_loc   = pyro.param("scale_loc",   torch.zeros(K))
    scale_scale = pyro.param("scale_scale", torch.ones(K),
                             constraint=dist.constraints.positive)

    # Variational parameters for initial and transition distributions
    pi_conc    = pyro.param("pi_conc",    torch.ones(K),
                            constraint=dist.constraints.positive)
    trans_conc = pyro.param("trans_conc", torch.ones(K, K),
                            constraint=dist.constraints.positive)

    with pyro.plate("weather_states", K):
        pyro.sample("means",  dist.Normal(mean_loc, mean_scale))
        pyro.sample("scales", dist.LogNormal(scale_loc, scale_scale))

    pyro.sample("pi", dist.Dirichlet(pi_conc))

    with pyro.plate("from_state", K):
        pyro.sample("trans", dist.Dirichlet(trans_conc))

def viterbi():
    pass

pyro.clear_param_store()

losses = list()

svi = SVI(
    model=Model,
    guide=guide,
    optim=Adam({'lr':0.005}),
    loss=TraceEnum_ELBO(max_plate_nesting=1)
)

steps = 500


for step in range(steps):
    loss = svi.step(data)
    losses.append(loss)
    if step % 50 == 0:
        print(f'Step {step} ------------------ Loss: {loss}')
    

learned_means  = pyro.param("mean_loc").detach()
learned_trans  = pyro.param("trans_conc").detach()
learned_trans  = learned_trans / learned_trans.sum(-1, keepdim=True)

# Sort states by activity level (high=sunny, low=rainy)
order = learned_means.argsort(descending=True)
labels = ["Sunny (high activity)", "Cloudy (medium)", "Rainy (low activity)"]

print("\nLearned emission means (activity score per weather state):")
for i, idx in enumerate(order):
    print(f"  {labels[i]}: learned={learned_means[idx]:.2f}  "
          f"true={true_means[idx]:.2f}")

x = np.arange(0, steps)
sns.lineplot(x=x, y=losses)
plt.show()