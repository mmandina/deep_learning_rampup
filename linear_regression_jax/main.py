import numpy as np
import mini_batches
import mini_batches_vectorized
import performance

def main():
   mini_batches_vectorized.stochastic_linear_regression()
   performance.stochastic_linear_regression()

def mean_squared_error(true,predicted):
    squared_errors = np.square(true-predicted)
    return np.mean(squared_errors)


if __name__ == "__main__":
    main()
