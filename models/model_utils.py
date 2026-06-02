import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import equinox as eqx


def sample_nam_params(module, prefix, guide=None):
    nodes, treedef = jax.tree_util.tree_flatten(module)
    sampled_leaves = []

    for i, node in enumerate(nodes):
        if eqx.is_array(node):
            name = f'{prefix}_{i}'
            if guide is not None:
                loc = numpyro.param(
                    f"{name}_loc", 
                    jnp.zeros_like(node)
                )
                scale = numpyro.param(
                    f"{name}_scale", 
                    jnp.ones_like(node) * 0.1,
                    constraint=dist.constraints.positive
                )
                sampled_leaves.append(
                    numpyro.sample(name, dist.Normal(loc, scale))
                )
            else:
                sampled_leaves.append(
                    numpyro.sample(
                        name=name,
                        fn=dist.Normal(0.0, 1.0).expand(node.shape)
                    )
                )
        else:
            sampled_leaves.append(node)
    
    reconstructed_tree = jax.tree_util.tree_unflatten(treedef, sampled_leaves)
    return reconstructed_tree