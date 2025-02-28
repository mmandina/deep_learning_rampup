import time
import jax
import jax.numpy as jnp
from jax import grad, jit, lax
import jax.random as random
from jax import devices
print(devices())

#generated using gpt to figure out why mini_batches.py was so much slower than numpy
# Hyperparameters
EPOCHS = 1000
BATCH_SIZE = 64
LEARNING_RATE = 0.01
NUM_SAMPLES = 1000

# Mean Squared Error loss function
@jit
def mse_loss(params, X, y):
    w, b = params
    predictions = X * w + b
    return jnp.mean((predictions - y) ** 2)

# Compute gradients of loss with respect to parameters (w, b)
grad_loss = jit(grad(mse_loss))

# Perform a single mini-batch SGD update
@jit
def sgd_update(params, X_batch, y_batch):
    grads = grad_loss(params, X_batch, y_batch)
    new_params = (params[0] - LEARNING_RATE * grads[0],
                  params[1] - LEARNING_RATE * grads[1])
    return new_params

# Process one epoch: shuffle data and process mini-batches using lax.fori_loop
def process_epoch(params, X, y, key):
    # Shuffle data using JAX's PRNG
    perm = random.permutation(key, jnp.arange(X.shape[0]))
    X_shuffled = X[perm]
    y_shuffled = y[perm]

    num_batches = X.shape[0] // BATCH_SIZE

    # Loop over mini-batches with an XLA-friendly loop
    def body_fun(i, current_params):
        start = i * BATCH_SIZE
        # Use lax.dynamic_slice for dynamic slicing (required inside jit)
        X_batch = lax.dynamic_slice(X_shuffled, (start,), (BATCH_SIZE,))
        y_batch = lax.dynamic_slice(y_shuffled, (start,), (BATCH_SIZE,))
        return sgd_update(current_params, X_batch, y_batch)

    params = lax.fori_loop(0, num_batches, body_fun, params)
    return params

# Train over multiple epochs using lax.scan
def train(params, X, y, key):
    def epoch_step(carry, _):
        params, key = carry
        key, subkey = random.split(key)
        params = process_epoch(params, X, y, subkey)
        return (params, key), None

    (params, _), _ = lax.scan(epoch_step, (params, key), jnp.arange(EPOCHS))
    return params

def stochastic_linear_regression():
    key = random.PRNGKey(42)

    # Generate synthetic dataset: y = 12 * x - 3
    key, subkey = random.split(key)
    X = random.uniform(subkey, shape=(NUM_SAMPLES,), minval=0.0, maxval=1.0)
    y = 12.0 * X - 3.0

    # Initialize parameters (w, b) randomly
    key, subkey1, subkey2 = random.split(key, 3)
    w = random.normal(subkey1, shape=())
    b = random.normal(subkey2, shape=())
    params = (w, b)

    # Time the training loop
    start_time = time.time()
    params = train(params, X, y, key)
    elapsed_time = time.time() - start_time

    print(f"Trained parameters: w = {params[0]:.4f}, b = {params[1]:.4f}")
    print(f"Training time: {elapsed_time:.6f} seconds")


