import time
import jax.numpy as jnp
import jax.random as jrandom
from jax import grad, jit, vmap, lax

# Hyperparameters
EPOCHS = 1000
ALPHA = 0.01
BATCH_SIZE = 64
N_SAMPLES = 1000
KEY = jrandom.PRNGKey(42)  # JAX PRNG for reproducibility

# Loss function (Mean Squared Error)
@jit
def mse_loss(w, b, X, y):
    preds = jnp.dot(X, w) + b
    return jnp.mean((preds - y) ** 2)

# Compute gradients of loss w.r.t. parameters
grad_mse = jit(grad(mse_loss, argnums=(0, 1)))

# SGD Update Function
@jit
def update_params(w, b, batch_X, batch_y):
    grad_w, grad_b = grad_mse(w, b, batch_X, batch_y)
    return w - ALPHA * grad_w, b - ALPHA * grad_b

# Batch Processing with `vmap`
@jit
def process_batches(w, b, X, y):
    """Updates weights using mini-batches with vectorized gradient computation."""
    batch_idxs = jnp.arange(0, len(X), BATCH_SIZE)

    def step(carry, idx):
        w, b = carry
        batch_X = X[idx:idx + BATCH_SIZE]
        batch_y = y[idx:idx + BATCH_SIZE]
        w, b = update_params(w, b, batch_X, batch_y)
        return (w, b), None

    (w, b), _ = lax.scan(step, (w, b), batch_idxs)
    return w, b

# Stochastic Gradient Descent (SGD) Training Loop
@jit
def train(w, b, X, y):
    """Runs SGD for a fixed number of epochs."""
    def epoch_step(carry, _):
        w, b, key = carry
        key, subkey = jrandom.split(key)
        shuffled_indices = jrandom.permutation(subkey, X.shape[0])
        X_shuffled, y_shuffled = X[shuffled_indices], y[shuffled_indices]
        w, b = process_batches(w, b, X_shuffled, y_shuffled)
        return (w, b, key), None

    (w, b, _), _ = lax.scan(epoch_step, (w, b, KEY), jnp.arange(EPOCHS))
    return w, b

# Main Function
def stochastic_linear_regression():
    # Generate synthetic dataset
    X = jrandom.uniform(KEY, shape=(N_SAMPLES,), minval=0, maxval=1)
    y = 12 * X - 3  # True function y = 12x - 3

    # Initialize parameters
    w = jrandom.normal(KEY)
    b = jrandom.normal(KEY)

    # Start timing
    start_time = time.time()
    
    # Train model
    w, b = train(w, b, X, y)

    # End timing
    elapsed_time = time.time() - start_time

    print(f"Final Model: y = {w:.4f} * x + {b:.4f}")
    print(f"Execution Time: {elapsed_time:.6f} seconds")

# Run the optimized JAX implementation
stochastic_linear_regression()
