import time
import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
from jax import grad,jit,device_put,devices,random
# Hyperparameters
EPOCHS = 1000
ALPHA = 0.01
BATCH_SIZE = 64
@jit
def jax_mse(m, b, X, y):
    preds = jnp.dot(X, m) + b
    return jnp.mean((preds - y) ** 2)
@jit
def update(params, X, y):
    # Specify argnums=(0, 1) to compute gradients for both w and b
    grads = grad(jax_mse, argnums=(0, 1))(params[0], params[1], X, y)
    new_params = (params[0] - ALPHA * grads[0],
                  params[1] - ALPHA * grads[1])
    return new_params
@jit
def process_batches(inputs, true_outputs, m, b):
    for idx in range(0, len(true_outputs), BATCH_SIZE):
        X_batch = inputs[idx:idx+BATCH_SIZE]
        y_batch = true_outputs[idx:idx+BATCH_SIZE]
        m, b = update((m, b), X_batch, y_batch)
    return m, b

def schotastic_linear_regression():
    # Start 
    start_time = time.time()
    key = random.PRNGKey(0)
    inputs = random.uniform(key, shape=(10000,))
    true_outputs = jnp.array((12 * inputs - 3))
    m = random.normal(key, shape=())
    b = random.normal(key, shape=(), dtype=jnp.float32)
    
    loss_history = []
 
    for _ in range(EPOCHS):
        # Shuffle data
        indices = random.permutation(key, jnp.arange(len(inputs)))
        inputs = inputs[indices]
        true_outputs = true_outputs[indices]
        # Compute predictions and loss
        outputs_mse = jax_mse(m, b, inputs, true_outputs)
        loss_history.append(float(outputs_mse))

    # Update parameters with mini-batch processing
        m, b = process_batches(inputs, true_outputs, m, b)
    
    elapsed_time = time.time() - start_time
    print(f"After {EPOCHS} epochs: Loss = {outputs_mse}, m = {m}, b = {b}")
    print(f"Execution time: {elapsed_time:.4f} seconds")
    
    plt.figure(figsize=(8, 5))
    plt.plot(loss_history, label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss over Epochs")
    plt.legend()
    plt.show()
  

if __name__ == "__main__":
    schotastic_linear_regression()
