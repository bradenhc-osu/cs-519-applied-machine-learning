import numpy as np
from features import feature_names
from sklearn.preprocessing import normalize


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
        fill_func = fill_smart

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

    fill_func(preprocessed_features, mapped_features, extras)

    if mappings_provided:
        return preprocessed_features
    else:
        return preprocessed_features, mappings


def map_binary(f, i, mappings, extras):
    m = (i, f, feature_names[i][0] + '=' + f)
    if not m in mappings:
        if not extras["mappings_provided"]:
            mappings[m] = len(mappings)
        else:
            extras["num_skipped"] += 1
            return None
    return mappings[m]


def map_smart(f, i, mappings, extras):
    fn = feature_names[i]
    # Check if it is a numerical feature
    if fn[1]:
        m = (i, '', fn[0])
        if m not in mappings:
            mappings[m] = len(mappings)
        fm = mappings[m]
        if not f.isdigit():
            # By returning 'None' we indicate that there was no number for this numerical feature. We will replace
            # it with the mean later
            return (None, fm)
        else:
            # Convert to a number
            fnum = float(f)

            # We need to capture some 'extra' values as we map so that we can calculate the mean and
            # normalize values later

            # Capture the sum
            sum_key = "sum_for_%d" % fm
            if sum_key in extras.keys():
                extras[sum_key] += fnum
            else:
                extras[sum_key] = fnum

            # Capture the number of features
            count_key = "count_for_%d" % fm
            if count_key in extras.keys():
                extras[count_key] += 1
            else:
                extras[count_key] = 1

            # Capture the minimum value
            min_key = "min_for_%d" % fm
            current_min = extras[min_key] if min_key in extras.keys() else fnum
            extras[min_key] = fnum if fnum < current_min else current_min

            # Capture the maximum value
            max_key = "max_for_%d" % fm
            current_max = extras[max_key] if max_key in extras.keys() else fnum
            extras[max_key] = fnum if fnum > current_max else current_max

            # Return the resulting tuple
            return (fnum, fm)
    else:
        # It isn't a numerical feature, just binarize it
        return map_binary(f, i, mappings, extras)


def fill_binary(preprocessed_features, mapped_features, extras):
    for i, row in enumerate(mapped_features):
        for j in row:
            if j is not None:
                preprocessed_features[i][j] = 1


def fill_smart(preprocessed_features, mapped_features, extras):
    for i, row in enumerate(mapped_features):
        for j in row:
            if j is not None:
                if isinstance(j, tuple):
                    # We are dealing with a numerical feature
                    v = j[0]
                    vi = j[1]
                    if v is None:
                        # Fill with the mean
                        sum_key = "sum_for_%d" % vi
                        count_key = "count_for_%d" % vi
                        preprocessed_features[i][vi] = extras[sum_key] / extras[count_key]
                    else:
                        # Normalize the value
                        fmin = extras["min_for_%d" % vi]
                        fmax = extras["max_for_%d" % vi]
                        if fmax - fmin == 0:
                            preprocessed_features[i][vi] = 0
                        else:
                            preprocessed_features[i][vi] = v
                else:
                    preprocessed_features[i][j] = 1


def scale_y(y, reverse=False):
    if reverse:
        return np.exp(y)
    else:
        return np.log(y)
