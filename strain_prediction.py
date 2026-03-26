import torch
import pyro
from pyro.distributions import LogNormal
from pyro.contrib.examples.nextstrain import load_nextstrain_counts
from datetime import datetime

# Configuration and Smoketest
if torch.cuda.is_available():
    device = torch.device('cuda')
    print('Using Cuda')
else:
    device = torch.device('cpu')
    print('Using CPU')

# PyTorch 2.6+ defaults torch.load(weights_only=True), which requires
# explicit allowlisting of non-tensor classes used in trusted checkpoints.
with torch.serialization.safe_globals([datetime]):
    dataset = load_nextstrain_counts()

def summarize(x, name=""):
    if isinstance(x, dict):
        for k, v in sorted(x.items()):
            summarize(v, name + "." + k if name else k)
    elif isinstance(x, torch.Tensor):
        print(f"{name}: {type(x).__name__} of shape {tuple(x.shape)} on {x.device}")
    elif isinstance(x, list):
        print(f"{name}: {type(x).__name__} of length {len(x)}")
    else:
        print(f"{name}: {type(x).__name__}")

def generative_model(dataset):
    features = dataset['features']
    counts = dataset['counts']
    assert features.shape[0] == counts.shape[-1]

    S, M = features.shape
    T, P, S = counts.shape

    time = torch.arange(float(T)) * dataset['time_step_days'] / 5.5
    time -= time.mean()

    strain_plate = pyro.plate('strain', S, dim=-1)
    place_plate = pyro.plate('place', P, dim=-2)
    time_plate = pyro.plate('time', T, dim=-3)

    rate_scale = pyro.sample('rate_scale', LogNormal())
