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
    find_linear_best_fit_line


class ConstantViewer(object):

    def __init__(self, clf, show_outliers: bool = False) -> None:
        super().__init__()

        self.show_outliers = show_outliers
        self.clf = clf
        self.fig = plt.figure("Scaled 3d  data")

        fig_manager = plt.get_current_fig_manager()
        fig_manager.window.showMaximized()

        self.gs = GridSpec(3, 4, self.fig)
        self.showing = False
        # Axes3D()
        self.master_plot = self.fig.add_subplot(self.gs[:3, :3], projection='3d')
        self.time_velocity = self.fig.add_subplot(self.gs[0, -1])
        self.time_power = self.fig.add_subplot(self.gs[1, -1])
        self.power_velocity = self.fig.add_subplot(self.gs[2, -1])

    def show(self):
        if not self.showing:
            self.gs.tight_layout(self.fig)

            self.fig.show()
            # plt.show()
            self.showing = True

    def close_all(self):
        # self.fig.hide()
        if self.showing:
            self.showing = False
            plt.close(self.fig)

    def plot_3d_plot(self, features, headers, labels):
        self.master_plot.scatter(features[:, 0], features[:, 1], features[:, 2], c=labels)
        self.master_plot.set_xlabel(headers[0])
        self.master_plot.set_ylabel(headers[1])
        self.master_plot.set_zlabel(headers[2])

        plot_hyperplane(self.clf, self.master_plot, colors='orange')

    def manipulate_features(self, features: np.ndarray, file_data: np.ndarray) -> (np.ndarray, np.ndarray):
        min_max_scaler = MinMaxScaler()

        moving_mask = file_data["motionState"] == "MOVING"
        features = features[moving_mask]
        file_data = file_data[moving_mask]

        new_features = None

        # for i in range(5, 6):
        for i in range(file_data["pathNumber"].min(), file_data["pathNumber"].max() + 1):
            path_number = file_data["pathNumber"] == i

            features_at_path = features[path_number]

            half = features_at_path.shape[0] // 2
            coefficient, _ = find_linear_best_fit_line(features_at_path[:half, 2], features_at_path[:half, 0])

            if coefficient < 0:
                features_at_path[:, 0] -= features_at_path[:, 0].max()
                features_at_path[:, 0] *= - 1

            features_at_path = min_max_scaler.fit_transform(features_at_path)
            outliers_free_features = features_at_path

            # outlier_detector.fit(features_at_path)
            # outlier_prediction = outlier_detector.predict(features_at_path)
            # outliers_free_features = features_at_path[outlier_prediction == 1]
            #
            # if outliers is None:
            #     outliers = features_at_path[outlier_prediction == -1]
            # else:
            #     outliers= np.concatenate((outliers, features_at_path[outlier_prediction == -1]), 0)

            if new_features is None:
                new_features = outliers_free_features
            else:
                new_features = np.concatenate((new_features, outliers_free_features), 0)

        # outlier_detector = IsolationForest(contamination=.1, n_jobs=-1)
        outlier_detector = OneClassSVM(gamma=10)  # Seems to work best
        # outlier_detector = OneClassSVM()
        # outlier_detector = OneClassSVM(random_state=0)

        outlier_detector.fit(new_features)
        outlier_prediction = outlier_detector.predict(new_features)
        outliers = new_features[outlier_prediction == -1]
        new_features = new_features[outlier_prediction == 1]

        if self.show_outliers:
            plot_hyperplane(outlier_detector, self.master_plot, interval=.04, colors="orange")

        return new_features, outliers, min_max_scaler

    def graph(self, file_data):
        self.clear_graphs()

        features, headers = get_features(file_data)

        new_features, outliers, scaler = self.manipulate_features(features, file_data)
        features = scaler.inverse_transform(new_features)

        # labels = cluster.DBSCAN(.1, n_jobs=-1).fit_predict(features) # Works
        labels = self.clf.predict(new_features)
        color_labels = list(map(lambda x: 'r' if x == 0 else 'b', labels))

        self.plot_3d_plot(new_features, headers, color_labels)

        if self.show_outliers:
            self.master_plot.scatter(outliers[:, 0], outliers[:, 1], outliers[:, 2], c="black")

        self.show_constants_graph(features, labels, c=color_labels)

        plot_subplots(new_features, headers, (self.time_velocity, self.time_power, self.power_velocity), color_labels)

        # self.show_grid()
        plt.draw()

    def clear_graphs(self):
        for ax in (self.master_plot, self.time_velocity, self.time_power, self.power_velocity):
            ax.cla()

    def show_grid(self):
        """
    Shows the grids for the major ticks in the plot.
        """
        for ax in (self.time_velocity, self.time_power, self.power_velocity):
            ax.grid(True)

    def show_constants_graph(self, features, labels, c=None):
        figure = plt.figure("Constants graph")
        constants_plot = figure.gca()
        constants_plot.set_xlabel("Velocity")
        constants_plot.set_ylabel("Average Power")

        fig_manager = plt.get_current_fig_manager()
        fig_manager.window.showMaximized()

        x = features[:, 1]
        y = features[:, 0]

        constants_plot.scatter(x, y, c=labels if c is None else c)

        coef_accelerating, intercept_accelerating = find_linear_best_fit_line(x[labels == 0], y[labels == 0])
        coef_decelerating, intercept_decelerating = find_linear_best_fit_line(x[labels == 1], y[labels == 1])

        x_lim = np.array(constants_plot.get_xlim())
        y_lim = np.array(constants_plot.get_ylim())

        constants_plot.plot(x_lim, coef_accelerating * x_lim + intercept_accelerating)
        constants_plot.plot(x_lim, coef_decelerating * x_lim + intercept_decelerating)

        average_coef = (coef_accelerating + coef_decelerating) / 2
        average_intercept = (intercept_accelerating + intercept_decelerating) / 2
        constants_plot.plot(x_lim, average_coef * x_lim + average_intercept)

        acceleration_coefficient = (coef_accelerating - average_coef)
        acceleration_intercept = (intercept_accelerating - average_intercept)
        k_acc = ((x.max() + x.min()) / 2) * acceleration_coefficient + acceleration_intercept

        bbox_props = dict(boxstyle="round,pad=0.3", fc="cyan", ec="b", lw=2)
        constants_plot.text(x_lim[0], y_lim[1],
                            "kV: {}\nkK: {}\nkAcc: {}".format(average_coef, average_intercept, k_acc), ha="left",
                            va="top", bbox=bbox_props)


def find_constants(open_path):
    if not os.path.exists(MODEL_FILE):
        easygui.msgbox("There are no models to use to classify the data. Please train algorithm first.")
        return

    clf = joblib.load(MODEL_FILE)

    if is_empty_model(clf):
        easygui.msgbox("The model has not been fitted yet. Please fit data to the model.")
        return

    # plt.show()

    while True:
        file = easygui.fileopenbox('Please locate csv file', 'Specify File', default=open_path, filetypes='*.csv')

        if file:
            open_path = "{0:s}\*.csv".format(os.path.dirname(file))

            file_data = get_data(file)

            if is_valid_log(file_data):
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
