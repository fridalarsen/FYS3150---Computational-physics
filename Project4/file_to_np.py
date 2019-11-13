import numpy as np

def file_to_array(filename):
    """
    Function for reading a file and converting the data into a numpy array.
    Arguments:
        filename (string): Name of file containing data of interest.
    """
    data = []
    with open(filename, "r") as f:
        for line in f.readlines():
            data.append(float(line))
    return np.array(data)
