from os.path import join
from pprint import pprint
from operator import itemgetter
from preprocess import preprocess_data, fetch_data
from perceptron import perceptron_train, perceptron_test
from main import run_with_mode


if __name__ == "__main__":
    # Fetch and preprocess the data
    trdata, m = preprocess_data(
        "./data/income.train.txt.5k"
    )
    test_data = fetch_data("./data/income.test.blind")
    tdata, _ = preprocess_data(
        data=test_data,
        mappings=m,
        test=True
    )

    w = None
    y = None
    for e in range(5):
        w, updates, update_p = perceptron_train(trdata[:, :-1], trdata[:, -1], w, 'averaged')

        y, test_positive = perceptron_test(w, tdata)

        print("epoch %d updates %d (%.1f%%) (+:%.1f%%)" %
                (e+1, updates, update_p * 100, test_positive * 100))
    print()

    # Write the results
    with open("./data/income.test.predicted", "w") as out_file:
        for i, d in enumerate(test_data):
            d.append(">50K" if y[i] > 0 else "<=50K")
            out_file.write(','.join(d) + '\n')
