import numpyro 
from numpyro import distributions as dist
from numpyro.infer import MCMC, NUTS

import jax.numpy as jnp
import jax.random as random

import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
import matplotlib.pyplot as plt
import arviz as az

train_data = pd.read_csv(
    "https://gist.githubusercontent.com/ucals/"
    "2cf9d101992cb1b78c2cdd6e3bac6a4b/raw/"
    "43034c39052dcf97d4b894d2ec1bc3f90f3623d9/"
    "osic_pulmonary_fibrosis.csv"
)

def Model(patient_code, weeks, FVC):

    patient_count = len(np.unique(patient_code))
    
    mu1 = numpyro.sample('mu1', dist.Normal(0, 500))
    sigma1 = numpyro.sample('sigma1', dist.HalfNormal(100))
    mu2 = numpyro.sample('mu2', dist.Normal(0, 5))
    sigma2 = numpyro.sample('sigma2', dist.HalfNormal(2.5))

    with numpyro.plate('plate1', patient_count):
        alpha = numpyro.sample('alpha', dist.Normal(mu1, sigma1))
        beta = numpyro.sample('beta', dist.Normal(mu2, sigma2))

    error = numpyro.sample('error', dist.HalfNormal(100))
    FVC_estimate = alpha[patient_code] + beta[patient_code] * weeks

    with numpyro.plate('plate2', len(patient_code)):
        numpyro.sample('obs', dist.Normal(FVC_estimate, error), obs=FVC)

patient_encoder = LabelEncoder()
train_data['patient_code'] = patient_encoder.fit_transform(train_data['Patient'].values)
FVC_obs = train_data['FVC'].values
weeks = train_data['Weeks'].values
patient_code = train_data['patient_code'].values

if __name__ == '__main__':
    rngkey = random.key(0)
    Kernel = NUTS(Model)
    mcmc = MCMC(
        sampler=Kernel,
        num_warmup=2000,
        num_samples=2000
    )
    mcmc.run(rng_key=rngkey, patient_code=patient_code, weeks=weeks, FVC=FVC_obs)
    mcmc.print_summary()
    posterior_samples = mcmc.get_samples()

data = az.from_numpyro(mcmc)
az.plot_trace(data, compact=True, figsize=(15,25))
plt.show()