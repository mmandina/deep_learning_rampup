import numpy as np
import mini_batches
import optimized_jax_mini_batches

def main():
    mini_batches.schotastic_linear_regression()
    # optimized_jax_mini_batches.schotastic_linear_regression()



def mean_squared_error(true,predicted):
    squared_errors = np.square(true-predicted)
    return np.mean(squared_errors)


if __name__ == "__main__":
    main()
