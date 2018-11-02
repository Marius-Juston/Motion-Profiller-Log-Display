# coding=utf-8
import math
from datetime import datetime
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from skimage import measure
from sklearn.exceptions import NotFittedError

from visualize import DTYPE, ENCODING, DELIMITERS, COLUMNS


def find_linear_best_fit_line(x, y):
    """
Finds the line of best fit
    :param x: the x data
    :param y: the y data
    :return: the coefficient of coefficient and the intercept
    """
    m, b = np.polyfit(x, y, 1)

    return m, b


def plot_subplots(features, translation, subplots, labels=None):
    """
This is used for if the features contain more than 2 dimensions and wants to be decomposed into the combinations of its dimensions
    :param features: The features to use
    :param translation: the translation from column index to column label name
    :param subplots: the subplots to plot the dimension combination on
    :param labels: the color array
    :return: the column combination in order
    """
    variable_combinations = combinations(list(_ for _ in range(features.shape[1])), 2)

    for combination, subplot in zip(variable_combinations, subplots):
        subplot.scatter(features[:, combination[1]], features[:, combination[0]], c=labels)
        subplot.set_xlabel(translation[combination[1]])
        subplot.set_ylabel(translation[combination[0]])

    return variable_combinations


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

    for delimiter in DELIMITERS:
        try:
            return np.genfromtxt(file, delimiter=delimiter, dtype=DTYPE, names=True, encoding=ENCODING)
        except (ValueError, IndexError):
            pass


def needed_axes(clf=None, ax=None):
    """
Method in case ax is None it will find what type of Axes is need in order to plot the features if there are 3 dimensions to the features then it will return a Axes3D otherwise a simple 2D Axes
    :param clf: the model to look at what what type of features are needed. This model needs to have been fitted already
    :param ax: the axes to check if is a 3D axes or not
    :return: True if the axes is an instance of Axes3D False otherwise and the Axes instance itself.
    """
    is_3d = False

    if ax is None:
        try:
            if clf is None:
                return False, plt.gca()

            clf.predict([[0, 0, 0]])
            is_3d = True
            ax = plt.gca(projection="3d")
        except ValueError:
            is_3d = False
            ax = plt.gca()

    elif isinstance(ax, Axes3D):
        is_3d = True

    return is_3d, ax


def plot_hyperplane(clf, ax: Axes = None, interval: float = .05, alpha=.3,
                    colors=('r', 'b')) -> Line3DCollection:
    """
Plots the hyperplane of the model in an axes
    :param clf: the classifier to use to find the hyperplane
    :param ax: the axes to plot the hyperplane into
    :param interval: the precision of the the hyperplane rendering.
    :param alpha:
    :param colors:
    :return: the mesh of the created hyperplane that was added to the axes
    """

    is_3d, ax = needed_axes(clf, ax)

    interval = int(1 / interval)

    # get the separating hyperplane
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()

    # create grid to evaluate model
    xx = np.linspace(x_min, x_max, interval)
    yy = np.linspace(y_min, y_max, interval)

    if is_3d:
        z_min, z_max = ax.get_zlim()

        zz = np.linspace(z_min, z_max, interval)

        yy, xx, zz = np.meshgrid(yy, xx, zz)

        if hasattr(clf, "decision_function"):
            z = clf.decision_function(np.c_[xx.ravel(), yy.ravel(), zz.ravel()])
        elif hasattr(clf, "predict_proba"):
            z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel(), zz.ravel()])[:, 1]
        else:
            raise ValueError(
                "The model passed in does not contain either the decision_function or the predict_proba functions.")

        z = z.reshape(xx.shape)

        vertices, faces, _, _ = measure.marching_cubes_lewiner(z, 0)
        # Scale and transform to actual size of the interesting volume
        vertices = vertices * [x_max - x_min, y_max - y_min, z_max - z_min] / interval
        vertices += [x_min, y_min, z_min]
        # and create a mesh to display
        mesh = Line3DCollection(vertices[faces],
                                facecolor=colors, alpha=alpha)

        ax.add_collection3d(mesh)

        return mesh
    else:
        xx, yy = np.meshgrid(xx,
                             yy)
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        return ax.contourf(xx, yy, Z, 10, colors=colors, alpha=alpha)


def plot_fitting_plane(clf, ax, number: int = 50, color=None):
    """
Plots the line of best fit that was fitted by the model on the specific axes
    :param clf:
    :param ax:
    :param number:
    :param color:
    :return:
    """
    x_min, x_max = ax.get_xlim()

    if isinstance(ax, Axes3D):
        y_min, y_max = ax.get_ylim()
        yy, xx = np.meshgrid(np.linspace(y_min, y_max, number), np.linspace(x_min, x_max, number))
        z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        z = z.reshape(xx.shape)

        return ax.plot_wireframe(xx, yy, z, facecolors=color)
    else:
        x = np.linspace(x_min, x_max, number)
        y = clf.predict(x)

        return ax.plot(x, y, c=color)


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
    if file is not None:
        fields = file.dtype.fields.keys()

        return all(key in fields for key in headers)
    return False


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


def is_empty_model(clf) -> bool:
    """
Checks if a model has been fitted yet or not
    :param clf: the model to check if it has been fitted yet or not
    :return: True if it has not yet been fitted, False otherwise
    """
    try:
        clf.predict([[0, 0, 0]])
        return False
    except NotFittedError:
        return True


def get_features(file_data):
    """
Return the features to use when finding the constants
    :param file_data: the log data to use to extract the features from
    :return: the features with a dictionary of each column name
    """
    average_power = (file_data["pLeft"] + file_data["pRight"]) / 2.0

    time = file_data["Time"]
    previous_data = np.roll(file_data, 1)
    velocity = np.sqrt(
        (file_data["xActual"] - previous_data["xActual"]) ** 2 + (
                file_data["yActual"] - previous_data["yActual"]) ** 2) / (time - previous_data["Time"])

    velocity[0] = 0

    # x = np.concatenate((np.vstack(average_power), np.vstack(velocity), np.vstack(time)), 1)
    x = np.hstack((average_power.reshape(-1, 1), velocity.reshape(-1, 1), time.reshape(-1, 1)))
    return x, {0: 'Average Power', 1: 'Velocity', 2: 'Time'}


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


def is_straight_line(file_data):
    x = file_data["xTarget"]
    y = file_data["yTarget"]

    x1 = x[0]
    x2 = x[-1]

    x_sub = (x2 - x1)
    if x_sub == 0:
        return np.alltrue(x == x1)

    y1 = y[0]
    y2 = y[-1]
    y_sub = (y2 - y1)

    # y = m*x + b
    m = y_sub / x_sub
    b = -((y_sub * x1) / x_sub) + y1

    return np.alltrue(y == m * x + b)


def get_xy_limited(intercept, coef, x_lim, y_lim):
    x = (y_lim - intercept) / coef

    x[x > x_lim.max()] = x_lim.max()
    x[x < x_lim.min()] = x_lim.min()

    y = x * coef + intercept

    return x, y


def retrieve_parameters(clf) -> dict:
    """
Retrieves the available parameters that can be changed in the model with as key the variable name and as value the datatype
    :param clf: the model to retrieve the parameters from
    :return: the list of parameters that can be changed in the model
    """
    parameters = {}

    for parameter_name, parameter_default_value in clf.get_params().items():
        if parameter_default_value is not None:
            parameter_type = type(parameter_default_value)

            if any([parameter_type is available_type for available_type in (int, float, str, bool)]):
                parameters[parameter_name] = parameter_default_value

    return parameters
