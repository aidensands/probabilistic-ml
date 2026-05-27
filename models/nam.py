import jax
import jax.numpy as jnp
import equinox as eqx
import equinox.nn as nn


class SubNet(eqx.Module):

    subnet: nn.MLP

    def __init__(self, in_features:int, out_features:int, neurons_per_layer:int, hidden_layers:int, key:jax.random.PRNGKey):
        self.subnet = nn.MLP(
            in_size=in_features,
            out_size=out_features,
            width_size=neurons_per_layer,
            depth=hidden_layers,
            activation=jax.nn.relu,
            key=key
        )
    
    def __call__(self, x):
        """
        Calls the forward algorithm of the feed forward network
        X: a vector of features
        """
        return self.subnet(x)
    
class NAM(eqx.Module):
    
    # We store a list of feed forward subnets depending on feature type
    continuous_subnets:list
    chrono_subnet:nn.MLP
    # Going to need matrices to store weights that are not given to subnets
    binary_weights:jnp.ndarray
    # Biases also get their own matrix
    biases:jnp.ndarray

    def __init__(self, num_binary:int, num_continuous:int, neurons_per_layer:int, key:jax.random.PRNGKey):
        
        k1, k2, k3 = jax.random.split(key, 3)

        continuous_keys = jax.random.split(k1, num_continuous)

        for i in range(num_continuous):
            self.continuous_subnets.append(SubNet(
                in_features=1,
                out_features=1,
                neurons_per_layer=neurons_per_layer,
                hidden_layers=1,
                key=continuous_keys[i]
            ))

        self.chrono_subnet = SubNet(
            in_features=2,
            out_features=1,
            neurons_per_layer=neurons_per_layer,
            hidden_layers=1,
            key=k2
        )

        self.binary_weights = jnp.zeros(num_binary)
        self.biases = jnp.zeros(1)

    def __call__(self, x_cont, x_chrono, x_binary):
        
        out = self.biases

        # Forward algorithm on continuous features
        for net, feature in zip(x_cont):
            out = out + net(jnp.array([feature]))

        # Forward algorithm on chronological features (sine cosine transformed)
        chrono_weight = self.chrono_subnet(x_chrono)
        out = out + chrono_weight
        
        # Binary indicators are just dot products 
        binary_weights = jnp.dot(binary_weights, x_binary)
        out = out + binary_weights

        return out.squeeze()


