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


def preprocess_data(filename=None, mappings=None, binarize_numerical=False, test=False, data=None):

    data = fetch_data(filename) if filename else data
    min_age = 0
    max_age = 0
    min_hours = 0
    max_hours = 0

    mappings_provided = True
    if mappings is None:
        mappings = {}
        mappings_provided = False


    def map_data(row_index, feature_index, feature_value):
        nonlocal min_age, max_age, min_hours, max_hours, binarize_numerical, mappings_provided, test

        if not test and feature_index == len(data[0]) - 1:
            if feature_value == ">50K":
                return 1
            else:
                return -1

        elif not binarize_numerical and feature_index == __age_feature_index:
            f = int(feature_value)
            if f < min_age:
                min_age = f
            elif f > max_age:
                max_age = f
            if 'age' not in mappings:
                mappings['age'] = len(mappings)
            return f

        elif not binarize_numerical and feature_index == __hours_worked_feature_index:
            f = int(feature_value)
            if f < min_hours:
                min_hours = f
            elif f > max_hours:
                max_hours = f
            if 'hours' not in mappings:
                mappings['hours'] = len(mappings)
            return f

        else:
            f = (feature_index, feature_value)
            if f not in mappings:
                if mappings_provided:
                    return None
                mappings[f] = len(mappings)
            return mappings[f]

    mapped = [[map_data(i, j, v) for j, v in enumerate(row)]
              for i, row in enumerate(data)]

    bindata = np.zeros((len(data), len(mappings) + (1 if not test else 0)))
    for i, row in enumerate(mapped):
        for j, v in enumerate(row):
            if v is not None:
                if j == len(row) - 1:
                    bindata[i][-1] = v
                elif not binarize_numerical and j == __age_feature_index:
                    bindata[i][j] = (v - min_age) / (max_age - min_age)
                elif not binarize_numerical and j == __hours_worked_feature_index:
                    bindata[i][j] = (v - min_hours) / (max_hours - min_hours)
                else:
                    bindata[i][v] = 1

    return bindata, mappings


if __name__ == '__main__':
    # Preprocess the training data
    this_dir = path.dirname(path.realpath(__file__))
    preprocessed_data = preprocess_data(
        path.join(this_dir, "data", "income.train.txt.5k"), True)
    print(len(preprocessed_data[500]))
    print(type(preprocessed_data))
