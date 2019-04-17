import numpy as np
from os import path


__age_feature_index = 0
__hours_worked_feature_index = 7


def fetch_data(filename):
    data = []
    with open(filename, "r") as f:
        for line in f:
            values = [x.strip() for x in line.split(",")]
            data.append(values)
    return data


def preprocess_data(filename=None, mappings=None, test=False, data=None):

    data = fetch_data(filename) if filename else data

    mappings_provided = True
    if mappings is None:
        mappings = {}
        mappings_provided = False


    def map_data(row_index, feature_index, feature_value):
        nonlocal mappings, mappings_provided, test

        if not test and feature_index == len(data[0]) - 1:
            if feature_value == ">50K":
                return 1
            else:
                return -1

        else:
            f = (feature_index, feature_value)
            if f not in mappings:
                if mappings_provided:
                    return None
                mappings[f] = len(mappings)
            return mappings[f]

    mapped = [[map_data(i, j, v) for j, v in enumerate(row)]
              for i, row in enumerate(data)]

    bindata = np.zeros((len(data), len(mappings) + 1 + (1 if not test else 0)))
    for i, row in enumerate(mapped):
        for j, v in enumerate(row):
            if j == 0:
                bindata[i][j] = 1
            if v is not None:
                if j == len(row) - 1:
                    bindata[i][-1] = v
                else:
                    bindata[i][v] = 1

    return bindata, mappings


if __name__ == '__main__':
    # Preprocess the training data
    this_dir = path.dirname(path.realpath(__file__))
    preprocessed_data, _ = preprocess_data(
        path.join(this_dir, "data", "income.train.txt.5k"))
    print(len(preprocessed_data[500]))
    print(type(preprocessed_data))
    print(preprocessed_data.shape)
