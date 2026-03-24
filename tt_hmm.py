import torch
import pyro
import pyro.distributions as dist
from pyro.distributions import DiscreteHMM
from pyro.infer import MCMC, NUTS
from pyro.optim import Adam
import pyarrow.parquet as pq
import pandas as pd
import numpy as np

# File Setup
table = pq.read_table('data/SGNex_Hct116_directRNA_replicate1_run1.parquet',
                       columns=['event_length'])
col = table.column("event_length").combine_chunks()
arr = col.to_numpy(zero_copy_only=False)
observations = torch.as_tensor(arr, dtype=torch.float64)
del table

# Hyperparameters
LATENT_STATES = 2
OBSERVABLE_STATES = 2
SEQ_LENGTH = len(observations)
WARMUP = 500
SAMPLES = 2000

def hmm(observations):
    # Priors for the initial state and the transitions
    initial_probs = pyro.sample('initial_probs', dist.Dirichlet(torch.ones(LATENT_STATES)))
    transition_probs = pyro.sample('transition_probs', dist.Dirichlet(torch.ones(LATENT_STATES, LATENT_STATES)))

    # Priors over the emission paramter
    loc = pyro.sample('loc', dist.Normal(torch.zeros(LATENT_STATES), 0.5).to_event(1))
    scale = pyro.sample('scale', dist.HalfNormal(torch.ones(LATENT_STATES)).to_event(1))

    observation_dist = dist.LogNormal(loc, scale)

    hmm_dist = DiscreteHMM(
        initial_logits=initial_probs.log(),
        transition_logits=transition_probs.log(),
        observation_dist=observation_dist
    )

    pyro.sample('obs', hmm_dist, obs=observations)

def inference(data, warmup, samples):
    kernel = NUTS(hmm)
    mcmc = MCMC(kernel=kernel, warmup_steps=warmup, num_samples=samples)
    print('Starting Markov Chain Monte Carlo...')
    mcmc.run(data)
    return mcmc.get_samples()

inference(observations, WARMUP, SAMPLES)