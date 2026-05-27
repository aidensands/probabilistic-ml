import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import equinox as eqx
import equinox.nn as nn

class NeuralNetwork(eqx.Module):

    layers:list

    def __init__(self, input_dims, hidden_dims, key):
        k1, k2, k3 = jax.random.split(key, 3)

        self.layers = [
            nn.Linear(input_dims, hidden_dims, key=k1),
            jax.nn.tanh,
            nn.Linear(hidden_dims, hidden_dims, key=k2),
            jax.nn.tanh,
            nn.Linear(hidden_dims, 1, key=k3)
        ]

    def __call__(self, x, *args, **kwargs):
        
        for layer in self.layers:
            x = layer(x)
        
        return x.squeeze(-1)
    
def sample_net(module, prefix):
    
    flat, treedef = jax.tree_util.tree_flatten(module)
    sampled_nodes = []

    for i, node in enumerate(flat):
        if isinstance(node, jax.Array):
            sampled_node = numpyro.sample(f'{prefix}_p{i}', dist.Normal(0.0, 1.0).expand(node.shape))
            sampled_nodes.append(sampled_node)
        else:
            sampled_nodes.append(node)

    reconstructed_tree = jax.tree_util.tree_unflatten(treedef, sampled_nodes)
    return reconstructed_tree


def BayesianNeuralNetwork(X, y=None, network=None):
    
    net = sample_net(network, 'net')
    logits = net(X).squeeze(-1)
    numpyro.sample('obs', dist.Bernoulli(logits=logits), obs=y)