"""
Modifications for Python 3 made by Braden Hitchcock
"""
import sys
import time
from operator import itemgetter
from svector import svector

__cache = {}


def read_from(textfile):
    if textfile in __cache.keys():
        for v in __cache[textfile]:
            yield v
    else:
        __cache[textfile] = []
        for line in open(textfile):
            label, words = line.strip().split("\t")
            __cache[textfile].append((1 if label == "+" else -1, make_vector(words.split()), words))
            yield __cache[textfile][-1]


def make_vector(words):
    v = svector()
    v['__bias__'] = 1
    for word in words:
        v[word] += 1
    return v


def test(devfile, model):
    err = 0
    err_pos = []
    err_neg = []
    for i, (label, svec, _) in enumerate(read_from(devfile), 1):  # note 1...|D|
        p = model.dot(svec)
        if label * p <= 0:
            err += 1
            if label > 0:
                err_pos.append((p, i))
            else:
                err_neg.append((p, i))

    return err/i, err_pos, err_neg  # i is |D| now


def train(trainfile, devfile, epochs=5):
    t = time.time()
    best_err = 1.
    model = svector()
    for it in range(1, epochs+1):
        updates = 0
        for i, (label, svec, _) in enumerate(read_from(trainfile), 1):  # label is +1 or -1
            if label * (model.dot(svec)) <= 0:
                updates += 1
                model += label * svec
        dev_err, _, _ = test(devfile, model)
        best_err = min(best_err, dev_err)
        print("epoch %d, update %.1f%%, dev %.1f%%" % (it, updates / i * 100, dev_err * 100))
    print("best dev err %.1f%%, |w|=%d, time: %.1f secs" % (best_err * 100, len(model), time.time() - t))


def train_average(trainfile, devfile, epochs=5):
    t = time.time()
    best_err = 1.
    model = svector()
    for it in range(1, epochs+1):
        updates = 0
        model_avg = svector()
        counts = svector()
        for i, (label, svec, _) in enumerate(read_from(trainfile), 1):  # label is +1 or -1
            if label * (model.dot(svec)) <= 0:
                updates += 1
                model += label * svec
                model_avg += i * label * svec
            counts += svec
        model, model_avg = prune(model, model_avg, counts, 2)
        it_model = i * model - model_avg
        dev_err, err_pos, err_neg = test(devfile, it_model)
        best_err = min(best_err, dev_err)
        print("epoch %d, update %.1f%%, dev %.1f%%" % (it, updates / i * 100, dev_err * 100))
    model = it_model
    print("best dev err %.1f%%, |w|=%d, time: %.1f secs" % (best_err * 100, len(model), time.time() - t))
    print_features(model)
    print_errors(err_pos, err_neg, devfile)
    return model


def prune(model, model_avg, counts, cutoff=1):
    new = svector()
    new_avg = svector()
    for k, v in counts.items():
        if v > cutoff:
            new[k] = model[k]
            new_avg[k] = model_avg[k]
    return new, new_avg


def print_features(model):
    print()
    print("Top 20 Most Negative/Positive Features -----------------")
    for i, (k, v) in enumerate(sorted(model.items(), key=itemgetter(1))):
        if i < 20 or  i > len(model) - 21:
            print(k,v)

def print_errors(err_pos, err_neg, devfile):

    print()
    print("Top 5 Most Falsely Negative Example Numbers ------------")
    for i, (_, j) in enumerate(sorted(err_pos)):
        if i >= 5:
            break
        print(__cache[devfile][j][2])

    print() 
    print("Top 5 Most Falsely Positive Example Numbers ------------")
    for i, (_, j) in enumerate(sorted(err_neg, reverse=True)):
        if i >= 5:
            break
        print(__cache[devfile][j][2])


def predict(testfile, model):
    with open(testfile + ".predicted", "w") as out:
        for (_, svec, words) in read_from(testfile):
            p = "+" if model.dot(svec) > 0 else "-"
            out.write("%s\t%s\n" % (p, words))


if __name__ == "__main__":
    print("Naive Perceptron ---------------------------------------")
    train(sys.argv[1], sys.argv[2], 10)
    print()
    print("Smart Averaged Perceptron ------------------------------")
    model = train_average(sys.argv[1], sys.argv[2], 10)
    print("Testing ------------------------------------------------")
    predict(sys.argv[3], model)
    print("Done")
