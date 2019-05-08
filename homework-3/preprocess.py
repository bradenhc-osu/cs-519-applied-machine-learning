import numpy as np
from features import feature_names


def read_data(filename):
    data = []
    with open(filename, "r") as f:
        first = True
        for line in f:
            if first:
                first = False
                continue
            values = [x.strip() for x in line.split(",")]
            data.append(values)
    return np.array(data)


def preprocess(features, mappings=None, binarize=False):

    mappings_provided = False
    if mappings is not None:
        mappings_provided = True
    else:
        mappings = {}

    if binarize:
        map_func = map_binary
        fill_func = fill_binary
    else:
        map_func = map_smart

    extras = {
        "mappings_provided": mappings_provided,
        "num_skipped": 0
    }

    mapped_features = [
        [
            map_func(feature, j, mappings, extras)
            for j, feature in enumerate(row)
        ]
        for i, row in enumerate(features)
    ]

    preprocessed_features = np.zeros((len(features), len(mappings)))

    fill_func(preprocessed_features, mapped_features)

    if mappings_provided:
        return preprocessed_features
    else:
        return preprocessed_features, mappings


def map_binary(f, i, mappings, extras):
    m = (i, f, feature_names[i] + '=' + f)
    if not m in mappings:
        if not extras["mappings_provided"]:
            mappings[m] = len(mappings)
        else:
            extras["num_skipped"] += 1
            return None
    return mappings[m]


def map_smart(f, i, mappings, extras):
    return None


def fill_binary(preprocessed_features, mapped_features):
    for i, row in enumerate(mapped_features):
        for j in row:
            if j is not None:
                preprocessed_features[i][j] = 1


def scale_y(y, reverse=False):
    if reverse:
        return np.exp(y)
    else:
        return np.log(y)
