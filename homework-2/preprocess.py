import numpy as np
from os import path

__i_age = 0
__i_sector = 1
__i_education = 2
__i_marital_status = 3
__i_occupation = 4
__i_race = 5
__i_gender = 6
__i_hours_per_week = 7
__i_country_of_origin = 8
__i_combo = 9


def fetch_data(filename):
    data = []
    with open(filename, "r") as f:
        for line in f:
            values = [x.strip() for x in line.split(",")]
            data.append(values)
    return data


def preprocess_data(filename=None, mappings=None, test=False, data=None, normalize_numerical=False, enable_combos=False):

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
        nonlocal mappings, mappings_provided, test, min_age, max_age, min_hours, max_hours, normalize_numerical

        if not test and feature_index == len(data[0]) - 1:
            if feature_value == ">50K":
                return 1
            else:
                return -1

        if normalize_numerical and feature_index == __i_age:
            f = int(feature_value)
            if f < min_age:
                min_age = f
            elif f > max_age:
                max_age = f

        if normalize_numerical and feature_index == __i_hours_per_week:
            f = int(feature_value)
            if f < min_hours:
                min_hours = f
            elif f > max_hours:
                max_hours = f

        f = (feature_index, feature_value)
        if f not in mappings:
            if mappings_provided:
                return None
            mappings[f] = len(mappings)
        return mappings[f]

    mapped = [[map_data(i, j, v) for j, v in enumerate(row)]
              for i, row in enumerate(data)]

    if normalize_numerical:
        mappings[(__i_age, 'numerical_age')] = len(mappings)
        mappings[(__i_hours_per_week, 'numerical_hours')] = len(mappings)

    if enable_combos:
        mappings[(__i_combo, 'race=White and sector=Private')] = len(mappings)

    bindata = np.zeros((len(data), len(mappings) + 1 + (1 if not test else 0)))
    for i, row in enumerate(mapped):
        for j, v in enumerate(row):
            if j == 0:
                bindata[i][j] = 1
            elif v is not None:
                if j == len(row) - 1 and not test:
                    bindata[i][-1] = v
                elif normalize_numerical and j == len(row) - 3:
                    bindata[i][-3] = (int(data[i][__i_age]) - min_age) / (max_age - min_age) - 1
                elif normalize_numerical and j == len(row) - 2:
                    bindata[i][-2] = (int(data[i][__i_hours_per_week]) - min_age) / (max_age - min_age) - 1
                elif enable_combos:
                    if j == len(row) - 2:
                        x = bindata[i][get_mapped_index_from(mappings, __i_race, 'White')]
                        y = bindata[i][get_mapped_index_from(mappings, __i_sector, 'Private')]
                        bindata[i][-2] = 1 if x == 1 and y == 1 else 0
                else:
                    bindata[i][v + 1] = 1

    return bindata, mappings


def get_mapped_index_from(mappings, index, value):
    for k, v in mappings.items():
        if k[0] == index and k[1] == value:
            return v
