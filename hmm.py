import numpyro
from numpyro.distributions import Dirichlet, Categorical
from numpyro.infer import MCMC, NUTS
import jax
import jax.numpy as jnp
from jax import lax
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
    hidden, observed = simulate_hmm(subkey, T=1000)
    sequences.append(observed)

def SimpleHMM(observations, states=2, obs=3):
    
    initial_probs = numpyro.sample('initial_probs', Dirichlet(jnp.ones(states)))
    
    with numpyro.plate('transition_plate', states):
        transition_probs = numpyro.sample('transition_probs', Dirichlet(jnp.ones(states)))
    
    with numpyro.plate('emission_plate', states):
        emission_probs = numpyro.sample('emission_probs', Dirichlet(jnp.ones(obs)))

    def forward(previous_state, t):
        current_state = numpyro.sample('current_state', Categorical(transition_probs[previous_state]))
        numpyro.sample('observed_state', Categorical(emission_probs[current_state], obs=observations[t]))

    first_step = numpyro.sample('first_step', Categorical(initial_probs))
    numpyro.sample('first_observation', Categorical(emission_probs[first_step]), obs=observations[0])

    lax.scan(forward, first_step, jnp.arange(1, len(observations)))


kernel = NUTS(SimpleHMM)
mcmc = MCMC(
    sampler=kernel,
    num_warmup=500,
    num_samples=1000
)
mcmc.run(key, observed)