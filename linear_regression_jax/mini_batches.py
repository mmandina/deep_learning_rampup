import time

import numpy as np
import main
import jax.numpy as jnp
from jax import grad

EPOCHS = 1000
ALPHA = .01
BATCH_SIZE = 64

def jax_mse(w,b,X,y):
    preds = jnp.dot(X,w)+b
    return jnp.mean((preds-y)**2)
def schotastic_linear_regression():
    start_time = time.time()
    inputs = np.random.uniform(0, 1, size=[1000])
    true_outputs = 12 * inputs - 3
    m = np.random.randn()
    b = np.random.randn()

    outputs = {'mse':float('inf'),'m':m,'b':b}
    for _ in range(EPOCHS):
        indices = np.random.permutation(len(inputs))
        inputs = inputs[indices]
        true_outputs= true_outputs[indices]
        approximation  = outputs['m'] * inputs + outputs['b']
        outputs['mse'] = jax_mse(outputs['m'],outputs['b'],inputs,true_outputs)
        # outputs['mse'] = main.mean_squared_error(true_outputs,approximation)
        process_batchs(approximation,true_outputs,inputs,outputs)
    elapsed_time = time.time() - start_time
    print(f"After {EPOCHS} epochs: Loss = {outputs['mse']}, m = {outputs['m']}, b = {outputs['b']}")
    print(f"Execution time: {elapsed_time:.4f} seconds")


def process_batchs(approximation,true_outputs,inputs,outputs):
    for idx in range(0,len(true_outputs),BATCH_SIZE):
        batch_true = true_outputs[idx:idx+BATCH_SIZE]
        batch_approximation = approximation[idx:idx+BATCH_SIZE]
        grad_loss = grad(jax_mse,argnums=(0,1))
        grad_m,grad_b  = grad_loss(outputs['m'],outputs['b'],inputs[idx:idx+BATCH_SIZE],batch_true)
        outputs['m']-=ALPHA*grad_m
        outputs['b']-=ALPHA*grad_b


schotastic_linear_regression()




