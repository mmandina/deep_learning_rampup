import time
import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
from jax import grad

# Hyperparameters
EPOCHS = 1000
ALPHA = 0.01
BATCH_SIZE = 64

def jax_mse(w, b, X, y):
    preds = jnp.dot(X, w) + b
    return jnp.mean((preds - y) ** 2)

def process_batchs(approximation, true_outputs, inputs, outputs):
    for idx in range(0, len(true_outputs), BATCH_SIZE):
        batch_true = true_outputs[idx:idx+BATCH_SIZE]
        grad_loss = grad(jax_mse, argnums=(0, 1))
        grad_m, grad_b = grad_loss(outputs['m'], outputs['b'], inputs[idx:idx+BATCH_SIZE], batch_true)
        outputs['m'] -= ALPHA * grad_m
        outputs['b'] -= ALPHA * grad_b

def schotastic_linear_regression():
    # Start 
    start_time = time.time()
    
    inputs = np.random.uniform(0, 1, size=[1000])
    true_outputs = 12 * inputs - 3
    m = np.random.randn()
    b = np.random.randn()
    outputs = {'mse': float('inf'), 'm': m, 'b': b}
    
    loss_history = []

    plt.ion()  # Turn on interactive mode
    fig, ax = plt.subplots()
    line, = ax.plot([], [], label="Training Loss")
    ax.set_xlim(0, EPOCHS)
    ax.set_ylim(0, 12) 
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss over Epochs")
    ax.legend()
    
    for epoch in range(EPOCHS):
        # Shuffle data
        indices = np.random.permutation(len(inputs))
        inputs = inputs[indices]
        true_outputs = true_outputs[indices]
        
        # Compute predictions and loss
        approximation = outputs['m'] * inputs + outputs['b']
        outputs['mse'] = jax_mse(outputs['m'], outputs['b'], inputs, true_outputs)
        loss_history.append(float(outputs['mse']))
        
        # Update parameters with mini-batch processing
        process_batchs(approximation, true_outputs, inputs, outputs)
        
        # Update plot every 10 epochs (or at the last epoch)
        if epoch % 10 == 0 or epoch == EPOCHS - 1:
            line.set_data(range(len(loss_history)), loss_history)
            ax.relim()
            ax.autoscale_view()
            plt.draw()
            plt.pause(0.01)
    
    elapsed_time = time.time() - start_time
    print(f"After {EPOCHS} epochs: Loss = {outputs['mse']}, m = {outputs['m']}, b = {outputs['b']}")
    print(f"Execution time: {elapsed_time:.4f} seconds")
    
    plt.ioff()  # Turn off interactive mode
    plt.show()  # Display the final plot

if __name__ == "__main__":
    schotastic_linear_regression()
