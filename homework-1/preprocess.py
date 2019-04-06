import numpy as np
from os import path


def fetch_data(filename):
    data = []
    with open(filename, "r") as f:
        for line in f:
            values = [x.strip() for x in line.split(",")]
            data.append(values)
    return data


def preprocess_data(filename):

    data = fetch_data(filename)

    mappings = {}

    def map_data(row_index, feature_index, feature_value):
        if feature_index == len(data[0]) - 1:
            if feature_value == ">50K":
                return 1
            else:
                return -1
        else:
            f = (feature_index, feature_value)
            if f not in mappings:
                mappings[f] = len(mappings)
            return mappings[f]

    mapped = [[map_data(i, j, v) for j, v in enumerate(row)] for i, row in enumerate(data)]

    bindata = np.zeros((len(data), len(mappings) + 1))
    for i, row in enumerate(mapped):
        for j, v in enumerate(row):
            if j == len(row) - 1:
                bindata[i][-1] = v
            else:
                bindata[i][v] = 1

    return bindata


if __name__ == '__main__':
    # Preprocess the training data
    this_dir = path.dirname(path.realpath(__file__))
    preprocessed_data = preprocess_data(
        path.join(this_dir, "data", "income.train.txt.5k"))
    print(preprocessed_data[500])
