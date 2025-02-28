import numpy as np
import main

EPOCHS = 1000
ALPHA = .01
BATCH_SIZE = 64

def schotastic_linear_regression():

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
        outputs['mse'] = main.mean_squared_error(true_outputs,approximation)
        process_batchs(approximation,true_outputs,inputs,outputs)
    print(f"After {EPOCHS} epochs: Loss = {outputs['mse']}, m = {outputs['m']}, b = {outputs['b']}")

def process_batchs(approximation,true_outputs,inputs,outputs):
    for idx in range(0,len(true_outputs),BATCH_SIZE):
        batch_true = true_outputs[idx:idx+BATCH_SIZE]
        batch_approximation = approximation[idx:idx+BATCH_SIZE]
        error = batch_true - batch_approximation
        grad_m = (-2/BATCH_SIZE) * np.sum(error*inputs[idx:idx+BATCH_SIZE])
        grad_b = (-2/BATCH_SIZE) * np.sum(error)
        outputs['m']-=ALPHA*grad_m
        outputs['b']-=ALPHA*grad_b





