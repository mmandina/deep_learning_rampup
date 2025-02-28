import numpy as np
import main

def linear_regression():

    inputs = np.random.uniform(0, 1, size=[1000])
    true_outputs = 12 * inputs - 3
    m = np.random.randn()
    b = np.random.randn()

    EPOCHS = 10000
    ALPHA = .01
    for _ in range(EPOCHS):
        approximations  = m * inputs + b
        mse = main.mean_squared_error(true_outputs,approximations)
        error = true_outputs - approximations
        grad_m = (-2/len(inputs)) * np.sum(error*inputs)
        grad_b = (-2/len(inputs)) * np.sum(error)
        m=m-ALPHA*grad_m
        b=b-ALPHA*grad_b
    print(f"After {EPOCHS} epochs: Loss = {mse}, m = {m}, b = {b}")