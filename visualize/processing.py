import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.preprocessing import MinMaxScaler

from visualize.selection import PointSelectors


class SelectionProcess(object):
    def __init__(self, file_data, features, fig):
        self._features = features
        self._file_data = file_data
        self.gs = None
        self.fig = fig
        self._manipulated_features = None

    def show(self):
        self.gs.tight_layout(self.fig)

        fig_manager = plt.get_current_fig_manager()
        fig_manager.window.showMaximized()

        plt.show()


class OutlierAndScalingSelection(SelectionProcess):

    def __init__(self, file_data, features):
        super().__init__(file_data, features, plt.figure("Outlier selector and data scaling"))

        rows = 1
        columns = 3

        self.gs = GridSpec(rows, columns, self.fig)

        # TODO x = motor power, y velocity, z time

        self.velocity_time = self.fig.add_subplot(self.gs[0, 0])
        self.power_velocity = self.fig.add_subplot(self.gs[0, 1])
        self.power_time = self.fig.add_subplot(self.gs[0, 2])

        self.initialize_plots()

        self.initialize_point_selection(features)

    def initialize_plots(self):
        self.velocity_time.set_xlabel("Time")
        self.velocity_time.set_ylabel("Velocity")

        self.power_velocity.set_xlabel("Velocity")
        self.power_velocity.set_ylabel("Average power")

        self.power_time.set_xlabel("Time")
        self.power_time.set_ylabel("Average power")

    def initialize_point_selection(self, features):
        self.velocity_time_graph = self.velocity_time.scatter(features[:, 1], features[:, 2])
        self.power_time_graph = self.power_time.scatter(features[:, 0], features[:, 2])
        self.power_velocity_graph = self.power_velocity.scatter(features[:, 0], features[:, 1])

        self.point_selector = PointSelectors((
            self.velocity_time_graph, self.power_time_graph, self.power_velocity_graph,
        ),
            self.on_select_scaled)

    def on_select_unscaled(self, indexes):
        pass

    def on_select_scaled(self, indexes):
        print(indexes.shape[0])

        self.point_selector.remove_scatter_plots(
            (self.velocity_time_graph, self.power_time_graph, self.power_velocity_graph))

        features = self._features[indexes]
        if features.shape[0] == self.point_selector.number_of_points:
            features = self._features

        features = MinMaxScaler().fit_transform(features)

        self.velocity_time_graph = self.velocity_time.scatter(features[:, 1], features[:, 2])
        self.power_time_graph = self.power_time.scatter(features[:, 0], features[:, 2])
        self.power_velocity_graph = self.power_velocity.scatter(features[:, 0], features[:, 1])

        self.point_selector.add_scatter_plots(
            (self.velocity_time_graph, self.power_time_graph, self.power_velocity_graph))

        # plt.draw()
        # self.fig.canvas.draw_idle()
        # self._manipulated_features = self._features[indexes]
