# coding=utf-8
import math
from datetime import datetime

import numpy as np
from matplotlib.axes import Axes
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from skimage import measure
from sklearn.exceptions import NotFittedError

from visualize import DELIMITER, DTYPE, ENCODING, COLUMNS


def rotate_points_around_point(points: np.ndarray, angle: float, point: iter = (0, 0)) -> np.ndarray:
    """
Rotates an array of points a certain angle in radians around a point
    :param points: the points to rotate
    :param angle: the angle in radians the points should be rotated by
    :param point: the point to rotate around
    :return: the array of points that have been rotated
    """
    rotated_points = points[:]

    rotated_points -= point
    rotated_points = np.dot(points, [[np.cos(angle), -np.sin(angle)],
                                     [np.sin(angle), np.cos(angle)]])
    rotated_points += point

    return rotated_points


def get_data(file) -> np.ndarray:
    """
Retrieves the data from the csv log file.
    :param file: the file to retrieve the data from
    :return: the data in an dictionary with as keys the first row in the csv file
    """
    return np.genfromtxt(file, delimiter=DELIMITER, dtype=DTYPE, names=True, encoding=ENCODING)


def plot_hyperplane(clf, ax, interval=.1):
    """
Plots the hyperplane of the model in an axes
    :param clf: the classifier to use to find the hyperplane
    :param ax: the axes to plot the hyperplane into
    :param interval: the precision of the the hyperplane rendering.
    :return: the mesh of the created hyperplane that was added to the axes
    """
    # get the separating hyperplane
    interval = int(1 / interval)

    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    z_min, z_max = ax.get_zlim()

    # create grid to evaluate model
    xx = np.linspace(x_min, x_max, interval)
    yy = np.linspace(y_min, y_max, interval)
    zz = np.linspace(z_min, z_max, interval)
    yy, xx, zz = np.meshgrid(yy, xx, zz)

    if hasattr(clf, "decision_function"):
        z = clf.decision_function(np.c_[xx.ravel(), yy.ravel(), zz.ravel()])
    else:
        z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel(), zz.ravel()])[:, 1]
    z = z.reshape(xx.shape)

    verteces, faces, _, _ = measure.marching_cubes(z, 0)
    # Scale and transform to actual size of the interesting volume
    verteces = verteces * [x_max - x_min, y_max - y_min, z_max - z_min] / interval
    verteces += [x_min, y_min, z_min]
    # and create a mesh to display
    # mesh = Poly3DCollection(verteces[faces],
    #                         facecolor='orange', alpha=0.3)
    mesh = Line3DCollection(verteces[faces],
                            facecolor='orange', alpha=0.3)
    ax.add_collection3d(mesh)

    return mesh


def get_velocity(time: np.ndarray, file_data: np.ndarray, actual: bool = True) -> np.ndarray:
    """
Returns an array of the velocities given the time and the x,y coordinates
    :param time: the time array
    :param file_data: the data where the x and y coordinates are stored
    :param actual: if returns the "xActual", "yActual" velocity or the "xTarget", "yTarget" velocity
    :return: an array of velocities
    """
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


def get_acceleration(time: np.ndarray, velocity: np.ndarray) -> np.ndarray:
    """
Returns an array of the velocities given the time and the x,y coordinates
    :param time: the time array
    :param velocity: the velocity array to ue to find the acceleration
    :return: an array with the accelerations
    """
    previous_velocity = np.roll(velocity, 1)
    previous_time = np.roll(time, 1)
    acceleration = (velocity - previous_velocity) / (time - previous_time)
    acceleration[0] = 0

    return acceleration


def get_coordinates_at_center(x_current, y_current, height, width, angle):
    """
Returns the x,y coordinates the object should be at to have its position at its center given its angle
    :param x_current: the current x coordinate
    :param y_current: the current y coordinate
    :param height: the height of the object
    :param width: the width of the object
    :param angle: the angle the object is at
    :return: the x,y coordinate to be at the center
    """
    angle = np.math.atan2((height / 2), (width / 2)) + np.deg2rad(angle)

    d = np.sqrt((width / 2) ** 2 + (height / 2) ** 2)
    sin = np.cos(angle)
    x = ((d * sin) if sin != 0 else 0)
    x = x_current - x
    cos = np.sin(angle)
    y = ((d * cos) if cos != 0 else 0)
    y = y_current - y

    return x, y


def is_valid_log(file: np.ndarray, headers: iter = COLUMNS) -> bool:
    """
Checks if the log is valid (has the needed columns headers)
    :param file: the file to check if it has all of the required headers
    :param headers: the headers needed
    :return: True if it has all the headers False otherwise
    """
    fields = file.dtype.fields.keys()

    return all(key in fields for key in headers)


def contains_key(file: np.ndarray, key: str) -> bool:
    """
Looks in the file to check if it has the key inside of it
    :param file: the file to check
    :param key: the key to look for
    :return: True if the file contains the key False otherwise
    """
    return key in file.dtype.fields.keys()


def sort_files(csv_files: dict) -> iter:
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
    """
Finds the largest factor of x
    :param x: the number to look for the largest factorial
    :return: the largest factorial in x
    """
    i = math.ceil(math.sqrt(x))
    while i >= 1:
        if x % i == 0:
            break
        i -= 1
    if x / i > x * (3 / 4):
        i = round(x / (3 * 4))

    return int(i)


def set_visible(patches: iter, value: bool) -> None:
    """
Sets the matplotlib patch visibility to be either on or off.
    :param patches: the list of matplotlib patches
    :param value: True for being visible False otherwise
    """
    for patch in patches:
        patch.set_visible(value)


def is_empty_model(clf):
    """
Checks if a model has been fitted yet or not
    :param clf: the model to check if it has been fitted yet or not
    :return: True if it has not yet been fitted, False otherwise
    """
    try:
        clf.predict([[0]])
        return False
    except NotFittedError:
        return True


def get_range_middle(data: np.ndarray) -> (float, float):
    """
Returns the range and the middle of the data ((max + min) / 2)
    :param data: the array to find the data from
    :return: the range (max - min) and the center of the data ((max + min) / 2)
    """
    min_value = data.min()
    max_value = data.max()
    data_range = max_value - min_value
    return data_range, (max_value + min_value) / 2


def view_subplot_legends(*args: Axes) -> None:
    """
Shows the legends for the Axes
    :param args: the axes to show the legends
    """
    for subplot in args:
        handles, labels = subplot.get_legend_handles_labels()
        subplot.legend(handles, labels)
