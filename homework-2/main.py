from os.path import join
from pprint import pprint
from operator import itemgetter
from preprocess import preprocess_data
from perceptron import perceptron_train, perceptron_classify, perceptron_validate


def print_weights(w, mappings=None, top=-1, bias=True):
    meta = []
    if mappings is not None:
        meta.append('labeled') 
    meta.append('with bias' if bias else 'without bias')
    meta_str = ', '.join(meta)
    if mappings is None:
        print("Weights (%s)" % meta_str)
        pprint(w)
    else:
        feature_names = [
            'age',
            'sector',
            'education',
            'marital-status',
            'occupation',
            'race',
            'gender',
            'hours-per-week',
            'country-of-origin'
        ]
        mws = []
        if bias is True:
            mws += [('bias', w[0])]
        mws += [("%s=%s" % (feature_names[x[0][0]], x[0][1]), y)
                for x, y in zip(mappings.items(), w[1:])]
        smws = sorted(mws, key=itemgetter(1))
        if top > 0:
            top_neg_mws = smws[:top]
            top_pos_mws = smws[len(smws) - top:]
            print("Top %d negative weights (ascending, %s):" % (top, meta_str))
            pprint(top_neg_mws)
            print("Top %d positive weights (ascending, %s):" % (top, meta_str))
            pprint(top_pos_mws)
        else:
            print("Weights (%s)" % meta_str)
            pprint(smws)
    print()


def run_with_mode(mode, epochs, trdata, dvdata):
    print("Mode: %s ----------------------------------------------" % mode)
    w = None
    for e in range(epochs):

        w, updates, update_p = perceptron_train(trdata[:, :-1], trdata[:, -1], w, mode)

        error_p, positive_p = perceptron_validate(w, dvdata[:, :-1], dvdata[:, -1])

        print("epoch %d updates %d (%.1f%%) dev_err %.1f%% (+:%.1f%%)" %
              (e+1, updates, update_p * 100, error_p * 100, positive_p * 100))
    print()
    return w


if __name__ == "__main__":
    # Fetch and preprocess the data
    trdata, m = preprocess_data("./data/income.train.txt.5k")
    dvdata, _ = preprocess_data("./data/income.dev.txt", mappings=m)

    w = run_with_mode('vanilla', 5, trdata, dvdata)
    print_weights(w, mappings=m, top=5, bias=False)
    w = run_with_mode('averaged', 5, trdata, dvdata)
    print_weights(w, mappings=m, top=5, bias=False)

    print_weights(w, mappings=m)
