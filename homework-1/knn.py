from os import path
from time import time
import numpy as np
import math
from preprocess import preprocess_data
from verify import verify


def get_euclidean_distances(X, y):
    return np.linalg.norm(X - y, axis=1)


def get_manhattan_distances(X, y):
    return np.sum(np.absolute(X - y), axis=1)


def knn_classify(X, y, k, distance_func):

    distances = distance_func(X[:, :-1], y)

    results = sorted([(d, X[i][-1], i) for i, d in enumerate(distances)])

    return 1 if sum(r[1] for r in results[:k]) > 0 else -1


def run_knn_with_labels(X, Y, k, f):
    errors = 0
    positive = 0
    start = time()
    for y in Y:
        expected = y[-1]
        result = knn_classify(X, y[:-1], k, f)
        if result == 1:
            positive += 1
        if expected != result:
            errors += 1
    end = time()

    error_rate = errors / len(X) * 100
    positive_rate = positive / len(Y) * 100
    run_time = end - start

    return error_rate, positive_rate, run_time


if __name__ == "__main__":
    # Get the preprocessed training and test data
    this_dir = path.dirname(path.realpath(__file__))
    train_file = path.join(this_dir, "data", "income.train.txt.5k")
    dev_file = path.join(this_dir, "data", "income.dev.txt")

    X, mappings = preprocess_data(train_file)
    Y, _ = preprocess_data(dev_file, mappings)

    print('Using Euclidean distances')
    f = get_euclidean_distances
    #print('Using Manhattan distances')
    #f = get_manhattan_distances

    #Ks = [1, 3, 5, 7, 9, 99, 999, 9999]
    Ks = [6, 7, 8, 9, 10, 11, 12]

    for k in Ks:
        # First get the training stats
        train_error_rate, train_positive_rate, train_time = run_knn_with_labels(X, X, k, f)

        # Then get the dev stats
        dev_error_rate, dev_positive_rate, dev_time = run_knn_with_labels(X, Y, k, f)

        # Print the results
        spaces = " " * (4 - int(math.log10(k)))
        print("k=%d%strn_err %.1f%% (+:%.1f%%)[%.1fs] dev_err %.1f%% (+:%.1f%%)[%.1fs]" % (
           k, spaces,
           train_error_rate, train_positive_rate, train_time,
           dev_error_rate, dev_positive_rate, dev_time
        ))
