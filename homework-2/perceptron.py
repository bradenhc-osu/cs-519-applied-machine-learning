import numpy as np


def perceptron_train(X, y, w=None, mode='vanilla'):
    w = np.zeros(X.shape[1]) if w is None else w
    if mode == 'vanilla':
        return perceptron_train_vanilla(X, y, w)

    elif mode == 'averaged':
        return perceptron_train_averaged(X, y, w)

    else:
        raise Exception("Invalid mode for perceptron: '%s'" % mode)


def perceptron_train_vanilla(X, y, w):
    updates = 0
    for i, x in enumerate(X):
        if y[i] * w.dot(x) <= 0:
            updates += 1
            w = w + y[i] * x
    return w, updates, updates / len(X)


def perceptron_train_averaged(X, y, w):
    wa = np.zeros(len(w))
    c = 0
    updates = 0
    for i, x in enumerate(X):
        if y[i] * w.dot(x) <= 0:
            updates += 1
            w = w + y[i] * x
            wa = wa + c * y[i] * x
        c += 1
    return c * w - wa, updates, updates / len(X)


def perceptron_classify(w, x, expected=None):
    predicted = 1 if w.dot(x) > 0 else -1
    if expected is not None:
        return predicted, predicted == expected
    else:
        return predicted, None


def perceptron_validate(w, X, y):
    errors = 0
    positive = 0
    for i, x in enumerate(X):
        p, ok = perceptron_classify(w, x, y[i])
        if p > 0:
            positive += 1
        if not ok:
            errors += 1
    return errors / len(X), positive / len(X)
