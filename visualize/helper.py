import math
from datetime import datetime

import numpy as np
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from skimage import measure
from sklearn.exceptions import NotFittedError

from visualize import DELIMITER, DTYPE, ENCODING, COLUMNS


def get_data(file):
    return np.genfromtxt(file, delimiter=DELIMITER, dtype=DTYPE, names=True, encoding=ENCODING)


def plot_hyperplane(clf, ax):
    # get the separating hyperplane
    interval = .1
    interval = int(1 / interval)

    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    z_min, z_max = ax.get_zlim()

    # create grid to evaluate model
    xx = np.linspace(x_min, x_max, interval)
    yy = np.linspace(y_min, y_max, interval)
    zz = np.linspace(z_min, z_max, interval)
    yy, xx, zz = np.meshgrid(yy, xx, zz)

    z = clf.decision_function(np.c_[xx.ravel(), yy.ravel(), zz.ravel()])
    z = z.reshape(xx.shape)

    verteces, faces, _, _ = measure.marching_cubes(z, 0)
    # Scale and transform to actual size of the interesting volume
    verteces = verteces * [x_max - x_min, y_max - y_min, z_max - z_min] / interval
    verteces = verteces + [x_min, y_min, z_min]
    # and create a mesh to display
    # mesh = Poly3DCollection(verteces[faces],
    #                         facecolor='orange', alpha=0.3)
    mesh = Line3DCollection(verteces[faces],
                            facecolor='orange', alpha=0.3)
    ax.add_collection3d(mesh)

    return mesh


def get_velocity(time, file_data, actual=True):
    previous_data = np.roll(file_data, 1)

    if actual:
        velocity = np.sqrt(
            (file_data["xActual"] - previous_data["xActual"]) ** 2 + (
                    file_data["yActual"] - previous_data["yActual"]) ** 2) / (time - previous_data["Time"])
    else:
        velocity = np.sqrt(
            (file_data["xTarget"] - previous_data["xTarget"]) ** 2 + (
                    file_data["yTarget"] - previous_data["yTarget"]) ** 2) / (time - previous_data["Time"])

    velocity[0] = 0

    return velocity


def get_acceleration(time, velocity):
    previous_velocity = np.roll(velocity, 1)
    previous_time = np.roll(time, 1)
    acceleration = (velocity - previous_velocity) / (time - previous_time)
    acceleration[0] = 0

    return acceleration


def get_coordinates_at_center(x_current, y_current, height, width, angle):
    angle = np.math.atan2((height / 2), (width / 2)) + np.deg2rad(angle)

    d = np.sqrt((width / 2) ** 2 + (height / 2) ** 2)
    sin = np.cos(angle)
    x = ((d * sin) if sin != 0 else 0)
    x = x_current - x
    cos = np.sin(angle)
    y = ((d * cos) if cos != 0 else 0)
    y = y_current - y

    return x, y


def is_valid_log(file):
    fields = file.dtype.fields.keys()

    return all(key in fields for key in COLUMNS)


def sort_files(csv_files):
    """
    Returns a sorted list of keys. The dates will be sorted first and with latest first then
    the non date objects are added and sorted alphabetically
    :param csv_files: the dictionary to have the keys sorted
    :return: a list with the keys for the dictionary sorted
    """
    rest = []
    result = []

    for file in csv_files.keys():
        if isinstance(file, datetime):
            result.append(file)
        else:
            rest.append(file)

    result = sorted(result, reverse=True)
    result.extend(sorted(rest))

    return result


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
