import numpyro
from numpyro.distributions import Dirichlet
import jax
import jax.numpy as jnp
import numpy as np

# --- Ground truth parameters ---
true_init_probs    = jnp.array([0.6, 0.4])          # P(start sunny)
true_trans_matrix  = jnp.array([[0.7, 0.3],          # Sunny -> [Sunny, Rainy]
                                 [0.4, 0.6]])          # Rainy -> [Sunny, Rainy]
true_emit_matrix   = jnp.array([[0.6, 0.3, 0.1],     # Sunny -> [Walk, Shop, Clean]
                                 [0.1, 0.4, 0.5]])     # Rainy -> [Walk, Shop, Clean]

def simulate_hmm(key, T=20):
    """Simulate one sequence of length T."""
    k1, k2 = jax.random.split(key)
    states  = []
    obs     = []

    # Sample initial state
    z = int(jax.random.categorical(k1, jnp.log(true_init_probs)))
    states.append(z)
    obs.append(int(jax.random.categorical(k2, jnp.log(true_emit_matrix[z]))))

    for t in range(1, T):
        k1, k2, key = jax.random.split(key, 3)
        z = int(jax.random.categorical(k1, jnp.log(true_trans_matrix[z])))
        states.append(z)
        obs.append(int(jax.random.categorical(k2, jnp.log(true_emit_matrix[z]))))

    return np.array(states), np.array(obs)

# Generate 5 sequences of length 20
key = jax.random.PRNGKey(42)
sequences = []
for i in range(5):
    key, subkey = jax.random.split(key)
    hidden, observed = simulate_hmm(subkey, T=20)
    sequences.append(observed)
    print(f"Seq {i}: {observed}")
    print(f"       hidden: {hidden}\n")

def SimpleHMM(observations, states=2, obs=3):
    
    initial_probs = numpyro.sample('initial_probs', Dirichlet(jnp.ones(states)))

    with numpyro.plate('transition_probs', len(observations)):
        pass
