import os

import easygui
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from sklearn.externals import joblib

import visualize
from visualize import MODEL_FILE
from visualize.feature_manipulator import manipulate_features, find_and_remove_outliers
from visualize.helper import plot_hyperplane, plot_subplots, is_empty_model, get_data, is_valid_log, get_xy_limited, \
    get_features, is_straight_line, find_linear_best_fit_line
from visualize.new_selector import remove_outliers


class ConstantViewer(object):
    """
Class meant to visualize the constants of a log file for the Motion Profiler of Walton Robotics
    """

    def __init__(self, clf, automatically_find_remove_outliers: bool = False,
                 manually_find_remove_outliers: bool = True) -> None:
        """
        :param clf: the model to use to separate the data
        :param automatically_find_remove_outliers: True if black dots should be placed in the 3d plot to represent the outliers False otherwise
        """
        super().__init__()

        self.manually_find_remove_outliers = manually_find_remove_outliers
        self.showing = False
        self.show_outliers = automatically_find_remove_outliers
        self.clf = clf

        self.fig = None
        self.gs = None
        self.master_plot = None
        self.time_power = None
        self.time_velocity = None
        self.power_velocity = None

        self.file_data = None
        self.headers = None
        self.new_scaled_features = None
        self.features = None
        self.outliers = None
        self.labels = None
        self.color_labels = None
        self.outlier_detector = None

    def show(self):
        """
    Shows the figure
        """

        if not self.showing:
            self.fig = plt.figure("Scaled 3d  data")

            fig_manager = plt.get_current_fig_manager()
            fig_manager.window.showMaximized()

            self.gs = GridSpec(3, 4, self.fig)

            self.master_plot = self.fig.add_subplot(self.gs[:3, :3], projection='3d')
            self.time_velocity = self.fig.add_subplot(self.gs[0, -1])
            self.time_power = self.fig.add_subplot(self.gs[1, -1])
            self.power_velocity = self.fig.add_subplot(self.gs[2, -1])

            self.gs.tight_layout(self.fig)
            self.clear_graphs()

            self.plot_3d_plot(self.new_scaled_features, self.headers, self.color_labels)

            if self.show_outliers:
                self.master_plot.scatter(self.outliers[:, 0], self.outliers[:, 1], self.outliers[:, 2], c="black")
                plot_hyperplane(self.outlier_detector, self.master_plot, interval=.04, colors="orange")

            self.show_constants_graph(self.features, self.file_data, self.labels, c=self.color_labels)

            plot_subplots(self.new_scaled_features, self.headers,
                          (self.time_velocity, self.time_power, self.power_velocity),
                          self.color_labels)

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

        self.file_data = file_data
        self.features, self.headers = get_features(file_data)

        # FIXME make it so that the outliers can be visualized as well
        self.new_scaled_features, self.features = manipulate_features(self.features, file_data)
        # features = scaler.inverse_transform(new_scaled_features)

        if self.show_outliers:
            self.new_scaled_features, self.outliers, self.outlier_detector = find_and_remove_outliers(
                self.new_scaled_features)

        if self.manually_find_remove_outliers:
            selector = remove_outliers(self.new_scaled_features)

            self.new_scaled_features = self.new_scaled_features[selector.indexes]
            self.features = self.features[selector.indexes]

        self.labels = self.clf.predict(self.new_scaled_features)
        self.color_labels = list(map(lambda x: 'r' if x == 0 else 'b', self.labels))

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
