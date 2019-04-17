from os.path import join
from preprocess import preprocess_data
from perceptron import perceptron_train, perceptron_classify, perceptron_validate


def run_with_mode(mode, epochs, trdata, dvdata):
    # Train and classify
    print("Mode: %s" % mode)
    w = None
    for e in range(epochs):

        w, updates, update_p = perceptron_train(trdata[:, :-1], trdata[:, -1], w, mode)

        error_p, positive_p = perceptron_validate(w, dvdata[:, :-1], dvdata[:, -1])

        print("epoch %d updates %d (%.1f%%) dev_err %.1f%% (+:%.1f%%)" %
              (e+1, updates, update_p * 100, error_p * 100, positive_p * 100))
    print()


if __name__ == "__main__":
    # Fetch and preprocess the data
    trdata, mappings = preprocess_data("./data/income.train.txt.5k")
    dvdata, _ = preprocess_data("./data/income.dev.txt", mappings=mappings)

    run_with_mode('vanilla', 5, trdata, dvdata)
    run_with_mode('averaged', 5, trdata, dvdata)
