import time
import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
from jax import grad, jit, random, vmap, devices

# Hyperparameters
EPOCHS = 1000
ALPHA = 0.1
BATCH_SIZE = 64

@jit
def jax_mse(m, b, X, y):
    # Compute elementwise predictions
    preds = m * X + b
    return jnp.mean((preds - y) ** 2)

@jit
def update(params, X, y):
    # Compute gradients for both m and b
    grads = grad(jax_mse, argnums=(0, 1))(params[0], params[1], X, y)
    new_params = (params[0] - ALPHA * grads[0],
                  params[1] - ALPHA * grads[1])
    return new_params

# Vectorized update function: applies 'update' to each batch
batched_update = vmap(update, in_axes=(None, 0, 0), out_axes=0)

@jit
def process_batches(inputs, true_outputs, m, b):
    # Use only full batches
    num_batches = inputs.shape[0] // BATCH_SIZE
    X_batches = inputs[:num_batches * BATCH_SIZE].reshape(num_batches, BATCH_SIZE)
    y_batches = true_outputs[:num_batches * BATCH_SIZE].reshape(num_batches, BATCH_SIZE)
    
    # Apply vectorized update over all batches
    updated_params = batched_update((m, b), X_batches, y_batches)
    
    # Average the updated parameters to get a single update
    new_m = jnp.mean(updated_params[0])
    new_b = jnp.mean(updated_params[1])
    
    return new_m, new_b

def train(inputs, true_outputs, m, b, key):
    loss_history = []
    for epoch in range(EPOCHS):
        # Shuffle the data each epoch
        key, subkey = random.split(key)
        indices = random.permutation(subkey, jnp.arange(len(inputs)))
        inputs, true_outputs = inputs[indices], true_outputs[indices]
        loss = jax_mse(m, b, inputs, true_outputs)
        loss_history.append(loss)
        
        # Debug: Compare full-data gradient to averaged mini-batch gradient
        m, b = process_batches(inputs, true_outputs, m, b)
    return m, b, jnp.array(loss_history)

def schotastic_linear_regression():
    start_time = time.time()
    key = random.PRNGKey(0)
    key, key_input, key_m, key_b = random.split(key, 4)
    
    # Generate synthetic data: y = 12 * x - 3 with x uniformly sampled from [0, 1]
    inputs = random.uniform(key_input, shape=(1000,))
    true_outputs = 12 * inputs - 3
    
    # Initialize parameters randomly
    m = random.normal(key_m, shape=())
    b = random.normal(key_b, shape=(), dtype=jnp.float32)
    
    print("Device(s):", devices())
    
    m, b, loss_history = train(inputs, true_outputs, m, b, key)
    
    elapsed_time = time.time() - start_time
    print(f"After {EPOCHS} epochs: Loss = {loss_history[-1]:.4f}, m = {m:.4f}, b = {b:.4f}")
    print(f"Execution time: {elapsed_time:.4f} seconds")
    
    # plt.figure(figsize=(8, 5))
    # plt.plot(loss_history, label="Training Loss")
    # plt.xlabel("Epoch")
    # plt.ylabel("Loss")
    # plt.title("Training Loss over Epochs")
    # plt.legend()
    # plt.show()

if __name__ == "__main__":
    schotastic_linear_regression()
