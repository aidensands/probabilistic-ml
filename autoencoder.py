import numpyro
import numpyro.distributions as dist
from numpyro.infer import SVI, Trace_ELBO
from numpyro.optim import Adam
import jax
import jax.numpy as jnp
import numpy as np
import flax.linen as linen
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split


class ProbabilisticAE(linen.Module):
    
    hidden_dims:int = 256
    latent_space_dims:int = 32

    @linen.compact
    def __call__(self, x, *args, **kwds):
        # Encoder
        h = linen.Dense(self.hidden_dims)(x)
        h = linen.relu(h)
        h = linen.Dense(self.hidden_dims // 2)(h)
        h = linen.relu(h)
        z = linen.Dense(self.latent_space_dims)(h)
        # Here we have reeached latent space
        # Decoder
        h = linen.Dense(self.hidden_dims // 2)(z)
        h = linen.relu(h)
        h = linen.Dense(self.hidden_dims)(h)
        h = linen.relu(h)
        h = linen.Dense(self.hidden_dims)

        x_reconstructed = linen.Dense(x.shape[-1])(h)
        x_reconstructed = linen.sigmoid(x_reconstructed)

        return x_reconstructed, z
    

def Model(x, hidden_dims=256, latent_dims=32):
    batch_size = x.shape[0]
    input_dim = x.shape[-1]

    ae = ProbabilisticAE()

    rng_key = numpyro.prng_key()
    x_recon, z = numpyro.module('autoencoder', ae, input_shape=(1, input_dim))(x)

    with numpyro.plate('batch', batch_size):
        numpyro.sample('obs', dist.Bernoulli(x_recon).to_event(1), obs=x)

def guide(x, hidden_dims=256, latent_dims=32):
    pass


def LoadData():
    print('Fetching MNIST Dataset')
    mnist = fetch_openml(
        name='mnist_784',
        version=1,
        as_frame=False,
        parser='auto',
    )

    X, y = mnist.data, mnist.target.astype(int)

    # Normalizing between 0 and 1 (pixel values go up to 255)
    X = X / 255.0

    NORMAL_DIGIT = 1
    ANOMALOUS_DIGIT= 7

    normal_mask = (y == NORMAL_DIGIT)
    test_mask = (y == NORMAL_DIGIT) | (y == ANOMALOUS_DIGIT)
    X_normal = X[normal_mask]

    X_train, X_val = train_test_split(X_normal, test_size=0.2, random_state=42)

    X_test = X[test_mask]
    y_test = y[test_mask]
    y_test_bin = (y_test == NORMAL_DIGIT).astype(int)

    X_train = jnp.asarray(X_train)
    X_val = jnp.asarray(X_val)
    X_test = jnp.asarray(X_test)

    return X_train, X_val, X_test


if __name__ == '__main__':
    
    X_train, X_val, X_test = LoadData()

    # Hyperparameters
HIDDEN_DIMS = 256
LATENT_DIMS = 32
BATCH_SIZE = 512
NUM_EPOCHS = 40
LEARNING_RATE = 1e-3

svi = SVI(
    lambda x: Model(x, hidden_dims=HIDDEN_DIMS, latent_dims=LATENT_DIMS),
    lambda x: guide(x, hidden_dims=HIDDEN_DIMS, latent_dims=LATENT_DIMS),
    Adam(step_size=LEARNING_RATE),
    loss=Trace_ELBO()
)

# Initialize with a dummy batch
rng_key = jax.random.PRNGKey(42)
svi_state = svi.init(rng_key, X_train[:BATCH_SIZE])

num_train = X_train.shape[0]
steps_per_epoch = num_train // BATCH_SIZE

for epoch in range(NUM_EPOCHS):
    perm = np.random.permutation(num_train)
    X_shuffled = X_train[perm]
    
    epoch_loss = 0.0
    for i in range(steps_per_epoch):
        batch = X_shuffled[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
        svi_state, loss = svi.update(svi_state, batch)
        epoch_loss += loss / BATCH_SIZE
    
    epoch_loss /= steps_per_epoch
    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {epoch_loss:.4f}")

# Extract trained parameters
trained_params = svi.get_params(svi_state)
print("Training complete.")