import os

import easygui
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from sklearn.externals import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import OneClassSVM

import visualize
from visualize import MODEL_FILE
from visualize.helper import is_empty_model, is_valid_log, get_data, get_features, plot_hyperplane, plot_subplots, \
    find_linear_best_fit_line, contains_key, is_straight_line, get_xy_limited


def manipulate_features(features: np.ndarray, file_data: np.ndarray, find_and_remove_outliers=False,
                        show_outliers=False, master_plot=None) -> (
        np.ndarray, np.ndarray):
    """
Return the features manipulated in a way as to make the algorithm for separating the data more accurate.
    :param features: the features to use
    :param file_data: the log file's data
    :return: the manipulated features array, the outliers of the data set and the data scaler
    """

    if contains_key(file_data, "motionState"):
        moving_mask = file_data["motionState"] == "MOVING"
        features = features[moving_mask]
        file_data = file_data[moving_mask]

    new_features = None
    scalers = {}
    if contains_key(file_data, "pathNumber"):

        for i in range(file_data["pathNumber"].min(), file_data["pathNumber"].max() + 1):
            min_max_scaler = MinMaxScaler()

            path_number = file_data["pathNumber"] == i
            scalers[min_max_scaler] = path_number

            features_at_path = features[path_number]

            half = features_at_path.shape[0] // 2
            coefficient, _ = find_linear_best_fit_line(features_at_path[:half, 2], features_at_path[:half, 0])

            if coefficient < 0:
                features_at_path[:, 0] *= - 1

            features_at_path = min_max_scaler.fit_transform(features_at_path)
            outliers_free_features = features_at_path

            if new_features is None:
                new_features = outliers_free_features
            else:
                new_features = np.concatenate((new_features, outliers_free_features), 0)
    else:
        min_max_scaler = MinMaxScaler()
        scalers[min_max_scaler] = np.full(features.shape[0], True)
        new_features = min_max_scaler.fit_transform(features)

    if find_and_remove_outliers:
        outlier_detector = OneClassSVM(gamma=10)  # Seems to work best

        outlier_detector.fit(new_features)
        outlier_prediction = outlier_detector.predict(new_features)
        outliers = new_features[outlier_prediction == -1]
        new_features = new_features[outlier_prediction == 1]

        features = reverse_scalling(new_features, scalers, outlier_prediction)

        if show_outliers:
            plot_hyperplane(outlier_detector, master_plot, interval=.04, colors="orange")

        return new_features, outliers, features
    else:
        features = reverse_scalling(new_features, scalers)
        return new_features, features


def reverse_scalling(features, scalers, outlier_prediction=None):
    features = np.copy(features)

    for scaler, index in zip(scalers.keys(), scalers.values()):
        if outlier_prediction is not None:
            index = index[outlier_prediction == 1]

        features[index] = scaler.inverse_transform(features[index])

    return features


class ConstantViewer(object):
    """
Class meant to visualize the constants of a log file for the Motion Profiler of Walton Robotics
    """

    def __init__(self, clf, show_outliers: bool = False) -> None:
        """
        :param clf: the model to use to separate the data
        :param show_outliers: True if black dots should be placed in the 3d plot to represent the outliers False otherwise
        """
        super().__init__()

        self.showing = False
        self.show_outliers = show_outliers
        self.clf = clf
        self.fig = plt.figure("Scaled 3d  data")

        fig_manager = plt.get_current_fig_manager()
        fig_manager.window.showMaximized()

        self.gs = GridSpec(3, 4, self.fig)

        self.master_plot = self.fig.add_subplot(self.gs[:3, :3], projection='3d')
        self.time_velocity = self.fig.add_subplot(self.gs[0, -1])
        self.time_power = self.fig.add_subplot(self.gs[1, -1])
        self.power_velocity = self.fig.add_subplot(self.gs[2, -1])

    def show(self):
        """
    Shows the figure
        """

        if not self.showing:
            self.gs.tight_layout(self.fig)

            self.fig.show()
            self.showing = True

    def close_all(self):
        """
    Closes the figure
        """
        if self.showing:
            self.showing = False
            plt.close(self.fig)

    def plot_3d_plot(self, features, headers, labels):
        """
    PLots the features in a 3d plot including the hyperplane that separates the data
        :param features: the features to use to plot in the graph
        :param headers: the axis titles
        :param labels: the color of each data point
        """
        self.master_plot.scatter(features[:, 0], features[:, 1], features[:, 2], c=labels)
        self.master_plot.set_xlabel(headers[0])
        self.master_plot.set_ylabel(headers[1])
        self.master_plot.set_zlabel(headers[2])

        plot_hyperplane(self.clf, self.master_plot, colors='orange')

    def graph(self, file_data):
        """
    Graphs the features from the log file. Creates a 3D graph with time, average power to motors and velocity as axises.
    It also decomposes the dimensions into individual 2D graphs.
        :param file_data: the log file to use to extract the data from
        """
        self.clear_graphs()

        features, headers = get_features(file_data)

        new_scaled_features, outliers, features = manipulate_features(features, file_data,
                                                                      find_and_remove_outliers=True,
                                                                      show_outliers=self.show_outliers,
                                                                      master_plot=self.master_plot)
        # features = scaler.inverse_transform(new_scaled_features)

        labels = self.clf.predict(new_scaled_features)
        color_labels = list(map(lambda x: 'r' if x == 0 else 'b', labels))

        self.plot_3d_plot(new_scaled_features, headers, color_labels)

        if self.show_outliers:
            self.master_plot.scatter(outliers[:, 0], outliers[:, 1], outliers[:, 2], c="black")

        self.show_constants_graph(features, file_data, labels, c=color_labels)

        plot_subplots(new_scaled_features, headers, (self.time_velocity, self.time_power, self.power_velocity),
                      color_labels)

        plt.draw()

    def clear_graphs(self):
        """
    Clears all the axes
        """
        for ax in (self.master_plot, self.time_velocity, self.time_power, self.power_velocity):
            ax.cla()

    def show_grid(self):
        """
    Shows the grids for the major ticks in the plot.
        """
        for ax in (self.time_velocity, self.time_power, self.power_velocity):
            ax.grid(True)

    def show_constants_graph(self, features, file_data, labels, c=None):
        """
    Creates an addition figure that will display the a graph with the constants on it and also the lines of best fit of
    the accelerating portion of it, the decelerating portion of it and the average of both of those lines
        :param features: the features to use to show the graph and find the constants from
        :param file_data: the whole file data
        :param labels: the labels to say if a data point is accelerating or not
        :param c: the color to plot the points
        :return the constants for the motion profiling kV, kK and kAcc
        """
        if is_straight_line(file_data):
            easygui.msgbox("It was determined that the robot was trying to go straight. "
                           "As an ongoing feature the program will be able detect kLag, etc... "
                           "however for the instance this features has not been added")

        figure = plt.figure("Constants graph")
        constants_plot = figure.gca()
        constants_plot.set_xlabel("Velocity")
        constants_plot.set_ylabel("Average Power")

        fig_manager = plt.get_current_fig_manager()
        fig_manager.window.showMaximized()

        x = features[:, 1]
        y = features[:, 0]

        constants_plot.scatter(x, y, c=labels if c is None else c)

        acceleration_mask = labels == visualize.ACCELERATING
        coef_accelerating, intercept_accelerating = find_linear_best_fit_line(x[acceleration_mask],
                                                                              y[acceleration_mask])
        deceleration_mask = labels == visualize.DECELERATING
        coef_decelerating, intercept_decelerating = find_linear_best_fit_line(x[deceleration_mask],
                                                                              y[deceleration_mask])

        x_lim = np.array(constants_plot.get_xlim())
        y_lim = np.array(constants_plot.get_ylim())

        x, y = get_xy_limited(intercept_accelerating, coef_accelerating, x_lim, y_lim)
        constants_plot.plot(x, y)
        x, y = get_xy_limited(intercept_decelerating, coef_decelerating, x_lim, y_lim)
        constants_plot.plot(x, y)
        # constants_plot.plot(x_lim, coef_accelerating * x_lim + intercept_accelerating)
        # constants_plot.plot(x_lim, coef_decelerating * x_lim + intercept_decelerating)

        average_coef = (coef_accelerating + coef_decelerating) / 2
        average_intercept = (intercept_accelerating + intercept_decelerating) / 2
        # constants_plot.plot(x_lim, average_coef * x_lim + average_intercept)

        x, y = get_xy_limited(average_intercept, average_coef, x_lim, y_lim)
        constants_plot.plot(x, y)

        acceleration_coefficient = (coef_accelerating - average_coef)
        acceleration_intercept = (intercept_accelerating - average_intercept)
        k_acc = ((x.max() + x.min()) / 2) * acceleration_coefficient + acceleration_intercept

        bbox_props = dict(boxstyle="round,pad=0.3", fc="cyan", ec="b", lw=2)
        constants_plot.text(x_lim[0], y_lim[1],
                            "kV: {}\nkK: {}\nkAcc: {}".format(average_coef, average_intercept, k_acc), ha="left",
                            va="top", bbox=bbox_props)

        return average_coef, average_intercept, k_acc


def find_constants(open_path):
    """
This is the main loop which runs until the user no selects any file. Retrieves the saved model for separating the data.
    :param open_path: the default location to start your search
    :return: the ending location the folder search was looking at
    """

    if not os.path.exists(MODEL_FILE):
        easygui.msgbox("There are no models to use to classify the data. Please train algorithm first.")
        return

    clf = joblib.load(MODEL_FILE)

    if is_empty_model(clf):
        easygui.msgbox("The model has not been fitted yet. Please fit data to the model.")
        return

    while True:
        file = easygui.fileopenbox('Please locate csv file', 'Specify File', default=open_path, filetypes='*.csv')

        if file:
            open_path = "{0:s}\*.csv".format(os.path.dirname(file))

            file_data = get_data(file)

            legacy_log = is_valid_log(file_data, visualize.LEGACY_COLUMNS)
            current_log = is_valid_log(file_data)

            if legacy_log or current_log:
                if legacy_log and not current_log:
                    easygui.msgbox("Because this log is missing information that makes it optimal "
                                   "for manipulating the data efficiently results may be inaccurate")

                # TODO make it so that when closing the figure using the GUI it reopens normally
                plot = ConstantViewer(clf)
                plot.graph(file_data)
                plot.show()
            else:

                easygui.msgbox(
                    "The file {0:s} is not a valid file.".format(os.path.basename(file)))
        else:
            break

    plt.close("all")
    return open_path


if __name__ == '__main__':
    find_constants(open_path=visualize.open_path)
