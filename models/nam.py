import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import equinox as eqx
import equinox.nn as nn

class SubNet(eqx.Module):

    layers:list

    def __init__(self, hidden_dims, key):
        k1, k2, k3 = jax.random.split(key, num=3)
        self.layers = [
            nn.Linear(1, hidden_dims, key=k1),
            jax.nn.elu,
            nn.Linear(hidden_dims, hidden_dims, key=k2),
            jax.nn.gelu,
            nn.Linear(hidden_dims, out_features=1, key=k3)
        ]

    def __call__(self, x, *args, **kwargs):
        out = x
        for layer in self.layers:
            out = layer(out)

        return out


def sample_subnet(subnet: SubNet, prefix:str):
    """Flatten the neural network into its weights and biases
        we then sample on each node using a normal distribution
    """
    flat, treedef = jax.tree_util.tree_flatten(subnet)
    sampled = []

    for idx, leaf in enumerate(flat):
        # For JAX arrays (weights / biases) sample an array with the same shape
        if isinstance(leaf, jax.Array) or hasattr(leaf, 'shape'):
            shape = getattr(leaf, 'shape', ())
            if shape == ():
                sampled_leaf = numpyro.sample(f'{prefix}_p{idx}', dist.Normal(0.0, 0.5))
            else:
                sampled_leaf = numpyro.sample(
                    f'{prefix}_p{idx}',
                    dist.Normal(0.0, 0.5).expand(shape).to_event(len(shape))
                )
            sampled.append(sampled_leaf)
        else:
            sampled.append(leaf)
            
    # At the end here we are now a subnet again but we now have a neural network of numpryo samples
    return jax.tree_util.tree_unflatten(treedef, sampled)

def apply_subnet(subnet, x):
    return subnet(x)


def BayesianNAM(X, y=None, dummy_subnets=None):
    num_features = X.shape[1]

    intercept = numpyro.sample('intercept', dist.Normal(0.0, 1.0))
    logits = jnp.full(X.shape[0], intercept)

    for i in range(num_features):
        subnet = sample_subnet(dummy_subnets[i], prefix=f'subnet_{i}')
        x_i = X[:, i:i+1]
        contrib = jax.vmap(apply_subnet, in_axes=(None, 0))(subnet, x_i).squeeze(-1)

        logits = logits + contrib

    numpyro.sample('obs', dist.Bernoulli(logits=logits), obs=y)