import time
import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
from jax import grad,jit

# Hyperparameters
EPOCHS = 1000
ALPHA = 0.01
BATCH_SIZE = 64

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

def process_batchs(inputs, true_outputs, outputs):
    # Loop over mini-batches and update parameters using the update function
    for idx in range(0, len(true_outputs), BATCH_SIZE):
        X_batch = inputs[idx:idx+BATCH_SIZE]
        y_batch = true_outputs[idx:idx+BATCH_SIZE]
        new_params = update((outputs['m'], outputs['b']), X_batch, y_batch)
        outputs['m'], outputs['b'] = new_params

def schotastic_linear_regression():
    # Start 
    start_time = time.time()
    
    inputs = np.random.uniform(0, 1, size=[1000])
    true_outputs = 12 * inputs - 3
    m = np.random.randn()
    b = np.random.randn()
    outputs = {'mse': float('inf'), 'm': m, 'b': b}
    
    loss_history = []
 
    for _ in range(EPOCHS):
        # Shuffle data
        indices = np.random.permutation(len(inputs))
        inputs = inputs[indices]
        true_outputs = true_outputs[indices]
        # Compute predictions and loss
        outputs['mse'] = jax_mse(outputs['m'], outputs['b'], inputs, true_outputs)
        loss_history.append(float(outputs['mse']))
        # Update parameters with mini-batch processing
        process_batchs(inputs, true_outputs, outputs)
    
    elapsed_time = time.time() - start_time
    print(f"After {EPOCHS} epochs: Loss = {outputs['mse']}, m = {outputs['m']}, b = {outputs['b']}")
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
