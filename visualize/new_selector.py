import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.path import Path
from matplotlib.widgets import LassoSelector
from mpl_toolkits.mplot3d import Axes3D

from visualize.new_constant_viewer import manipulate_features


class Graphs(object):

    def __init__(self, features: np.array, title="2D Graphs", suptitle="") -> None:
        rows = 1
        columns = 3

        self._fig = plt.figure(title)
        self._fig.suptitle(suptitle)

        gs = GridSpec(rows, columns, self._fig)

        # TODO x = motor power, y velocity, z time

        self._velocity_time = self._fig.add_subplot(gs[0, 0])
        self._power_velocity = self._fig.add_subplot(gs[0, 1])
        self._power_time = self._fig.add_subplot(gs[0, 2])

        velocity_time_graph = self._velocity_time.scatter(features[:, 1], features[:, 2])
        power_time_graph = self._power_time.scatter(features[:, 0], features[:, 2])
        power_velocity_graph = self._power_velocity.scatter(features[:, 0], features[:, 1])

        self.flat_graphs = {self._velocity_time: velocity_time_graph, self._power_time: power_time_graph,
                            self._power_velocity: power_velocity_graph}

        self._combination_figure = plt.figure("3D Combination")

        self.all_features_axes = Axes3D(self._combination_figure)

        self.all_feature_graph = self.all_features_axes.scatter(features[:, 0], features[:, 1], features[:, 2])

        self._initialize_plots()

    def _initialize_plots(self):
        self._velocity_time.set_xlabel("Velocity")
        self._velocity_time.set_ylabel("Time")

        self._power_velocity.set_xlabel("Velocity")
        self._power_velocity.set_ylabel("Average power")

        self._power_time.set_ylabel("Time")
        self._power_time.set_xlabel("Average power")

        self.all_features_axes.set_xlabel("Average Power")
        self.all_features_axes.set_ylabel("Velocity")
        self.all_features_axes.set_zlabel("Time")

    def close_figures(self):
        plt.close(self._fig)
        plt.close(self._combination_figure)

    def add_key_press_handler(self, handler):
        self._fig.canvas.mpl_connect("key_press_event", handler)
        self._combination_figure.canvas.mpl_connect("key_press_event", handler)

    def show(self):
        self._fig.show()
        self._combination_figure.show()
        plt.show()


class LassoManager(object):

    def __init__(self, graphs: Graphs, alpha_other=0.05, close_on_exit=True) -> None:
        self.close_on_exit = close_on_exit
        self.alpha_other = alpha_other
        self._graphs = graphs

        self._lassos = {}
        self._canvas = {}
        self._collection = {}
        self._fc = {}
        self._xys = {}

        self.indexes = None

        self.full_array = None

        number_of_points = []

        self.selected_color = np.array([0, 0, 1, 1])
        self.unselected_color = np.array([1, 0, 0, alpha_other])

        for ax in graphs.flat_graphs:
            self._canvas[ax] = ax.figure.canvas

            self._collection[ax] = graphs.flat_graphs[ax]
            self._xys[ax] = self._collection[ax].get_offsets()
            Npts = self._xys[ax].shape[0]

            number_of_points.append(Npts)

            # Ensure that we have separate colors for each object
            self._fc[ax] = self._collection[ax].get_facecolors()

            if len(self._fc[ax]) == 0:
                raise ValueError('Collection must have a facecolor')
            elif len(self._fc[ax]) == 1:
                # self._fc[ax] = np.tile(self._fc[ax], (Npts, 1))
                self._fc[ax] = np.tile(self._fc[ax], (Npts, 1))

            self._lassos[ax] = LassoSelection(ax, onselect=self.on_select)

        if number_of_points[1:] != number_of_points[:-1]:
            raise ValueError('Collections must have the same number of points')

        self.full_array = np.arange(0, number_of_points[0])
        self.indexes = self.full_array

        graphs.add_key_press_handler(self.on_key_press)

    def on_key_press(self, event):
        if event.key == "enter":
            self.disconnect()

            if self.close_on_exit:
                self._graphs.close_figures()

    def on_select(self, event, verts):
        ax = event.inaxes

        print(event.dblclick)

        path = Path(verts)

        ind = np.nonzero(path.contains_points(self._xys[ax]))[0]

        if event.button == 1:
            if ind.size == 0:
                self.indexes = self.full_array
            else:
                self.add_points(ind)

        elif event.button == 3:
            self.remove_points(ind)

        self.update_indexes()

    def update_indexes(self):

        for ax in self._lassos:
            # self.fc[ax][:, -1] = self.alpha_other
            # self.fc[ax][self.indexes, -1] = 1

            self._fc[ax][:] = self.unselected_color
            self._fc[ax][self.indexes] = self.selected_color

            self._collection[ax].set_facecolors(self._fc[ax])
            self._canvas[ax].draw_idle()

        # demo = np.zeros(selector._full_array.shape[0])
        # demo[selector.indexes] = 1

        # clf = OneClassSVM()
        # clf.fit(features, demo)

        # plot_hyperplane(clf, self._graphs._features_graph, colors='orange')

        self._graphs.all_feature_graph._facecolor3d = list(self._fc.values())[0]
        self._graphs.all_feature_graph._edgecolor3d = list(self._fc.values())[0]
        self._graphs.all_feature_graph.figure.canvas.draw_idle()

    def disconnect(self):
        for ax in self._lassos:
            self._lassos[ax].disconnect_events()
            self._fc[ax][:, -1] = 1
            self._collection[ax].set_facecolors(self._fc[ax])
            self._canvas[ax].draw_idle()

    def add_points(self, ind):
        self.indexes = np.union1d(self.indexes, ind)

    def remove_points(self, ind):
        self.indexes = np.setdiff1d(self.indexes, ind, assume_unique=True)

    def get_labels(self):
        labels = np.zeros(self.full_array.shape[0])
        labels[self.indexes] = 1

        return labels


class LassoSelection(LassoSelector):
    def __init__(self, ax, onselect):
        super().__init__(ax, onselect)

    def _release(self, event):
        if self.verts is not None:
            self.verts.append(self._get_data(event))
            self.onselect(event, self.verts)
        self.line.set_data([[], []])
        self.line.set_visible(False)
        self.verts = None


def remove_outliers(all_features):
    graphs = Graphs(all_features, title="Outlier Selection",
                    suptitle="Please unselect the outliers. Press [Enter] to confirm.")
    selector = LassoManager(graphs)
    graphs.show()
    plt.show()

    return selector


def manipulate_features_and_remove_outliers(all_features, all_data):
    new_scaled_features, features = manipulate_features(all_features, all_data)

    all_features = new_scaled_features

    graphs = Graphs(all_features, title="Manipulated Features Outlier Selection",
                    suptitle="Please unselect the outliers. Press [Enter] to confirm.")
    selector = LassoManager(graphs)
    graphs.show()

    return new_scaled_features, features, selector


def select_accelerating_vs_decelerating(all_features):
    graphs = Graphs(all_features, title="Acceleration vs Deceleration Selection",
                    suptitle="Selected=accelerating, Unselected=decelerating. Press [Enter] to confirm.")
    selector = LassoManager(graphs)

    graphs.show()
    return selector
