import numpy
from os import path


def fetch_data(filename):
    data = []
    with open(filename, "r") as f:
        for line in f:
            values = [x.strip() for x in line.split(",")]
            data.append(values)
    return data

def preprocess_data(filename):
    # Fetch the string data from the file
    data = fetch_data(filename)
    print(data)
    return None


if __name__ == '__main__':
    # Preprocess the training data
    this_dir = path.dirname(path.realpath(__file__))
    preprocess_data(path.join(this_dir, "data", "income.train.txt.5k"))
