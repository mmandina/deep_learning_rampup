import numpy as np
import time
import batched_gd
import sgd
import mini_batches
def main():
    # batched_gd.linear_regression()
    # sgd.stochastic_linear_regression()
    start = time.time()
    mini_batches.stochastic_linear_regression()
    elapsed_time = time.time() - start
    print(f"Execution time: {elapsed_time:.4f} seconds")





def mean_squared_error(true,predicted):
    squared_errors = np.square(true-predicted)
    return np.mean(squared_errors)


if __name__ == "__main__":
    main()
