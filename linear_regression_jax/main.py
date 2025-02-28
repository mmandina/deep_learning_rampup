import numpy as np
import mini_batches
import mini_batches_vectorized

def main():
    mini_batches.schotastic_linear_regression()
    # below uses learning rate of .1 vice .01, otherwise moves too slowly over 1k epochs
    mini_batches_vectorized.schotastic_linear_regression()

def mean_squared_error(true,predicted):
    squared_errors = np.square(true-predicted)
    return np.mean(squared_errors)


if __name__ == "__main__":
    main()
