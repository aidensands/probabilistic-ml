import torch
import pyro
import pyro.distributions as dist
from pyro.infer import MCMC, NUTS
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

synth_cluster_1 = torch.normal(2, 1.5, size=(500,))
synth_cluster_2 = torch.normal(3.5, 0.5, size=(500,))
synth_clusters = torch.concat((synth_cluster_1, synth_cluster_2))

sns.kdeplot(synth_clusters, fill=True)
plt.title('True Distribution')
plt.show()
K = 2

# New technique: use KMeans to initialize informative priors
def initialize_priors(X, K):

    print('Running Lloyd K-Means...')

    X = X.reshape(-1,1)

    kmeans = KMeans(
        n_clusters=K,
        n_init='auto',
        init='k-means++'
    )
    kmeans.fit(X)
    
    prior_locs = kmeans.cluster_centers_
    print(f'Prior Locs: {prior_locs}')
    print('K-Means: Ok')
    return torch.Tensor(prior_locs).squeeze(-1)


prior_locs = initialize_priors(synth_clusters, K)

def model(X):
    weights = pyro.sample('weights', dist.Dirichlet(torch.ones(K)))

    # Learned: Allow for each component to have its own loc and scale 
    with pyro.plate('components', K):
        locs = pyro.sample('locs', dist.Normal(prior_locs, 3.0))
        scale = pyro.sample('scale', dist.LogNormal(0.0, 2.0))

    with pyro.plate('data', len(X)):
        assignment = pyro.sample('assignment', dist.Categorical(weights))
        pyro.sample('obs', dist.Normal(locs[assignment], scale[assignment]), obs=X)

Kernel = NUTS(model)
mcmc = MCMC(Kernel, 600, 400)
mcmc.run(synth_clusters)
mcmc.summary()
posterior_samples = mcmc.get_samples()

X, Y = posterior_samples['locs'].T

sns.lineplot(X.numpy())
sns.lineplot(Y.numpy())
plt.title('Locs during MCMC')
plt.xlabel('MCMC Step')
plt.ylabel('loc')
plt.show()

mcmc.get_samples()

sns.kdeplot(synth_clusters, fill=True)
