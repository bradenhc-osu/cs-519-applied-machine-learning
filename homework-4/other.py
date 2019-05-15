import sys
import time
from svector import svector
import numpy as np
from sklearn.svm import LinearSVC
from nltk.corpus import stopwords
from sklearn.neural_network import MLPClassifier

__cache = {}


def read_from(filename):
    if filename in __cache.keys():
        for v in __cache[filename]:
            yield v
    else:
        __cache[filename] = []
        for line in open(filename):
            label, words = line.strip().split("\t")
            label = 1 if label == "+" else -1
            svec = make_vector(words.split())
            v = (label, svec)
            __cache[filename].append(v)
            yield v


def get_pruned_keys(filename, cutoff=1):
    counts = svector()
    if filename in __cache.keys():
        for (_, svec) in __cache[filename]:
            counts += svec
    else:
        __cache[filename] = []
        for line in open(filename):
            label, words = line.strip().split("\t")
            label = 1 if label == "+" else -1
            svec = make_vector(words.split())
            v = (label, svec)
            __cache[filename].append(v)
            counts += svec
    keys = {}
    i = 0
    for k, v in counts.items():
        if v > cutoff and k:
            keys[k] = i
            i += 1
    return keys


def preprocess(filename, keys=None):
    ks = get_pruned_keys(filename, 1) if keys is None else keys
    features = []
    labels = []
    for i, (label, svec) in enumerate(read_from(filename), 1):
        labels.append(label)
        features.append(np.zeros(len(ks)))
        for k, v in svec.items():
            if k in ks:
                features[i - 1][ks[k]] = v
    if keys is None:
        return features, labels, ks
    return features, labels


def make_vector(words):
    v = svector()
    for word in words:
        v[word] += 1
    return v


def train_svm(trainfile, devfile):
    print("Preprocessing data files")
    X_train, y_train, ks = preprocess(trainfile)
    X_dev, y_dev = preprocess(devfile, ks)

    t = time.time()

    print("Begin training")
    clf = MLPClassifier()
    clf.fit(X_train, y_train)

    print("Predicting on dev")
    predictions = clf.predict(X_dev)
    err = 0
    for i, l in enumerate(y_dev):
        err += not l == predictions[i]
    
    dev_err = err / i

    print("dev err %.1f%%, |w|=%d, time: %.1f secs" % (dev_err * 100, len(ks), time.time() - t))

if __name__ == "__main__":
    train_svm(sys.argv[1], sys.argv[2])