from os import path
from time import time
from preprocess import fetch_data, preprocess_data
from knn import knn_classify, get_euclidean_distances

if __name__ == "__main__":

    # Get the data and preprocess it
    this_dir = path.dirname(path.realpath(__file__))
    train_file = path.join(this_dir, "data", "income.train.txt.5k")
    test_file = path.join(this_dir, "data", "income.test.blind")

    X, mappings = preprocess_data(filename=train_file)

    test_data = fetch_data(test_file)
    Y, _ = preprocess_data(data=test_data, mappings=mappings, test=True)

    k = 8

    positive = 0
    start = time()
    results = []
    for y in Y:
        result = knn_classify(X, y, k, get_euclidean_distances)
        if result == 1:
            positive += 1
        results.append(result)
    end = time()

    test_positive_rate = positive / len(Y) * 100
    test_time = end - start

    print("k=%d (+:%.1f%%)[%.1fs]" % (k, test_positive_rate, test_time))

    # Write out to file
    with open(path.join(this_dir, "data", "income.test.predicted"), "w") as out_file:
        for i, d in enumerate(test_data):
            d.append(">50K" if results[i] > 0 else "<=50K")
            out_file.write(','.join(d) + '\n')
