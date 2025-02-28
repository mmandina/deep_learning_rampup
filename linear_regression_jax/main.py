import numpy as np


def main():
    pass



def mean_squared_error(true,predicted):
    squared_errors = np.square(true-predicted)
    return np.mean(squared_errors)


if __name__ == "__main__":
    main()
