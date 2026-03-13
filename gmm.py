import torch
import pyro
import pyro.distributions as dist
from pyro.optim import Adam 
from pyro.infer import SVI, Trace_ELBO
import matplotlib.pyplot as plt
import seaborn as sns
import pyarrow.parquet as pq

torch.manual_seed(42)
parquet = pq.read_table('data/SGNex_Hct116_directRNA_replicate1_run1.parquet', columns=['event_length'])
df = parquet.to_pandas()
nparray = df.to_numpy()
data = torch.from_numpy(nparray)

K = 2  # number of components
N = len(data)

def model(X):
    with pyro.plate('clusters', K):
        means = pyro.sample('means', dist.Normal(torch.tensor([0.0068]), torch.tensor([0.00001])))

    weights = pyro.sample('weights', dist.Dirichlet(0.5 * torch.ones(K)))

    with pyro.plate('data', N):
        assigment = pyro.sample('assignment', dist.Categorical(weights))
        pyro.sample('observations', dist.Normal(means[assigment], 0.5), obs=X)


def guide(X):
    means_loc = pyro.param('means_loc', torch.randn(K))
    means_scale = pyro.param('means_scale', torch.ones(K), constraint=dist.constraints.positive)
    weights_conc = pyro.param('weights_conc', torch.ones(K), dist.constraints.positive)
    
    with pyro.plate('components', K):
        pyro.sample('means', dist.Normal(means_loc, means_scale))
    
    pyro.sample('weights', dist.Dirichlet(weights_conc))

    assigment_probs = pyro.param('assigment_probs', torch.ones(N,K) / K, constraint=dist.constraints.simplex)

    with pyro.plate('data', N):
        pyro.sample('assignment', dist.Categorical(assigment_probs))

pyro.clear_param_store()


optimizer = Adam({'lr':0.005})
svi = SVI(model, guide, optimizer, Trace_ELBO())

epochs = 10000
losses = list()

for epoch in range(epochs):
    loss = svi.step(data)
    losses.append(loss)
    if epoch % 500 == 0:
        print(f'Iteration {epoch}: Loss {loss}')

means_learned = pyro.param("means_loc")
weights_learned = pyro.param("weights_conc")
weights_learned = weights_learned / weights_learned.sum()  # normalize

print("Learned means:", means_learned)
print("Learned weights:", weights_learned)

sns.histplot(data)
plt.show()

x = torch.linspace(0, 10000, 10000)
sns.lineplot(x=x, y=losses)
plt.show()