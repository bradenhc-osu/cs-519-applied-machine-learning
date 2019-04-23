import numpy as np
from perceptron import perceptron_train, perceptron_classify, perceptron_validate


def run_with_mode(mode, epochs, trdata):
    # Train and classify
    print("Mode: %s" % mode)
    w = None
    for e in range(epochs):

        w, updates, update_p = perceptron_train(trdata[:, :-1], trdata[:, -1], w, mode)

        print("epoch %d updates %d (%.1f%%)" %
              (e+1, updates, update_p * 100))
    print()
    print("Final weight vector:", w)


if __name__ == "__main__":
    # Fetch and preprocess the data
    trdata = np.array([[0, 1, 1, 1],
                       [0 , 1, -1, -1],
                       [0, -1, -1, 1],
                       [0, -1, 1, -1]])

    run_with_mode('vanilla', 8, trdata)
