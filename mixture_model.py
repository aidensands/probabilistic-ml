import torch
import pyro
from pyro.distributions import Gamma, LogNormal, Beta, Categorical
from pyro.infer import MCMC, NUTS
from pyro.optim import Adam
import pyarrow.parquet as pq
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

torch.set_default_dtype(torch.float64)

table = pq.read_table('data/SGNex_Hct116_directRNA_replicate1_run1.parquet', columns=['event_length'])
df = table.to_pandas()
sample = df.sample(n=100000)
data = torch.tensor(sample['event_length'].dropna().values, dtype=torch.float64)
data = torch.log(data)


def model(data):
    mixture_weights = pyro.sample('mixture_weights', Beta(3.0, 2.0))

    alpha1 = pyro.sample('alpha1', LogNormal(1.5, 0.4))
    beta1 = pyro.sample('beta1', LogNormal(2.5, 0.4))

    alpha2 = pyro.sample('alpha2', LogNormal(0.5, 0.5))
    beta2 = pyro.sample('beta2', LogNormal(0.5, 0.5))

    pyro.factor('ordering', -10.0 * torch.relu(alpha1 - alpha2))

    with pyro.plate('data', len(data)):
        
        log_prob1 = Gamma(alpha1, beta1).log_prob(data)
        log_prob2 = Gamma(alpha2, beta2).log_prob(data)

        log_mixture = torch.logaddexp(torch.log(mixture_weights) + log_prob1, torch.log(1.0 - mixture_weights) + log_prob2)

        pyro.factor('obs', log_mixture)

if __name__ == '__main__':
    kernel = NUTS(model, target_accept_prob=0.9)
    mcmc = MCMC(
        kernel=kernel, 
        num_samples=2000,
        warmup_steps=1000,
        num_chains=4,
        mp_context="spawn"   # explicit on macOS
    )
    mcmc.run(data)
    samples = mcmc.get_samples() 

    for param in ["mixture_weights", "alpha1", "beta1", "alpha2", "beta2"]:
        s = samples[param]
        print(f"{param}: mean={s.mean():.3f}, std={s.std():.3f}")