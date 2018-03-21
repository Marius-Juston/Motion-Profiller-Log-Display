import math
import os

import numpy as np
from sklearn.exceptions import NotFittedError

COLUMNS = (
    "Time", "xActual", "yActual", "angleActual", "xTarget", "yTarget", "angleTarget", "XTE", "lagE", "angleE",
    "pLeft", "pRight", "pathNumber", "motionState"
)

MODEL_FILE_NAME = "model.pkl"
MODEL_DATA_FILE_NAME = "data.npz"
open_path = "{0:s}\*.csv".format(os.path.expanduser("~"))
ACCELERATING = 0
DECELERATING = 1
OUTLIER = -1
DTYPE = tuple("U10" if i == len(COLUMNS) - 1 else np.float32 for i in range(len(COLUMNS)))


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
