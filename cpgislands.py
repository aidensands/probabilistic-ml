import torch
import pyro
from pyro.distributions import Categorical, Dirichlet, DiscreteHMM
from pyro.infer import MCMC, NUTS
from Bio import SeqIO
import numpy as np

FASTA_PATH = 'data/ncbi_dataset/data/GCF_000001405.40/chr21.fna'
torch.set_default_dtype(torch.float64)
device = 'cpu'

def prep_data(path):
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}
    seqs = list()

    for record in SeqIO.parse(path, 'fasta'):
        numeric_seq = [mapping.get(base, 4) for base in record.seq]
        seqs.append(numeric_seq)

    data = torch.Tensor(seqs, device=device)
    mask = (data != 4)
    data = data[mask]
    return data

def model(X):
    K = 2 # Island vs No Island
    emission_alphas = torch.Tensor([[10.0, 1.0, 1.0, 10.0],[1.0, 10.0, 10.0, 1.0]])
    # Complete uncertainty of starting state
    initial_probs = pyro.sample('initial_probs', Dirichlet(torch.Tensor([5.0, 1.0])))
    # Non island transitions
    transition_0 = pyro.sample('transition_0', Dirichlet(torch.Tensor([10.0, 1.0])))
    # Island transitions
    transition_1 = pyro.sample('transition_1', Dirichlet(torch.Tensor([1.0, 10.0])))
    # Emittion Probabilities
    emission_probs = pyro.sample('emission_probs', Dirichlet(emission_alphas))

    transition_matrix = torch.stack([transition_0, transition_1])

    obs_dist = Categorical(probs=emission_probs)

    hmm = DiscreteHMM(
        initial_logits=initial_probs,
        transition_logits=transition_matrix,
        observation_dist=obs_dist
    )

    pyro.sample('obs', hmm, obs=X)

def fit_model(model, X):
    sampling_kernel = NUTS(model=model)
    sampling_method = MCMC(sampling_kernel, 500, 1000)
    sampling_method.run(X)
    return sampling_method.summary()

dataset = prep_data(FASTA_PATH).squeeze()
fit_model(model=model, X=dataset)