import math

import numpy as np
from sklearn.exceptions import NotFittedError

from visualize import DELIMITER, DTYPE, ENCODING, COLUMNS


def get_data(file):
    return np.genfromtxt(file, delimiter=DELIMITER, dtype=DTYPE, names=True, encoding=ENCODING)


def is_valid_log(file):
    fields = file.dtype.fields.keys()

    return all(key in fields for key in COLUMNS)


def find_largest_factor(x):
    i = math.ceil(math.sqrt(x))
    while i >= 1:
        if x % i == 0:
            break
        i -= 1
    if x / i > x * (3 / 4):
        i = round(x / (3 * 4))

    return int(i)


def set_visible(patches, value):
    for patch in patches:
        patch.set_visible(value)


def is_empty_model(clf):
    try:
        clf.predict([[0, 0, 0]])
        return False
    except NotFittedError:
        return True
