"""
1 3d graph with 3 other graphs with the axes against each other. There you can use lasso
selection on one of the graphs and it will cause the other graphs to also see the points
as being selected
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path
from matplotlib.widgets import LassoSelector
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import make_classification


class SelectFromCollection(object):
    def __init__(self, ax, collection, alpha_other=0.3):
        self.canvas = ax.figure.canvas
        self.collection = collection
        self.alpha_other = alpha_other

        self.xys = collection.get_offsets()
        self.Npts = len(self.xys)

        # Ensure that we have separate colors for each object
        self.fc = collection.get_facecolors()
        if len(self.fc) == 0:
            raise ValueError('Collection must have a facecolor')
        elif len(self.fc) == 1:
            self.fc = np.tile(self.fc, (self.Npts, 1))

        self.lasso = LassoSelector(ax, onselect=self.onselect)
        self.ind = []

    def onselect(self, verts):
        path = Path(verts)
        self.ind = np.nonzero(path.contains_points(self.xys))[0]
        self.fc[:, -1] = self.alpha_other
        self.fc[self.ind, -1] = 1
        self.collection.set_facecolors(self.fc)
        self.canvas.draw_idle()

    def disconnect(self):
        self.lasso.disconnect_events()
        self.fc[:, -1] = 1
        self.collection.set_facecolors(self.fc)
        self.canvas.draw_idle()


def create_3_plot_figure(title, data_3d):
    outlier_selection = plt.figure(title)
    a = outlier_selection.add_subplot(131)
    b = outlier_selection.add_subplot(132)
    c = outlier_selection.add_subplot(133)
    add_data((a, b, c), X)


def add_data(axis, data, color=None):
    for axis, plot in zip([(0, 1), (0, 2), (1, 2)], axis):
        plot.scatter(data[:, axis[0]], data[:, axis[1]], c=color)

        if axis[0] == 0:
            plot.set_xlabel("Helloo")
        elif axis[0] == 1:
            plot.set_xlabel("cheese")
        elif axis[0] == 2:
            plot.set_xlabel("pancake")

        if axis[1] == 0:
            plot.set_ylabel("Helloo")
        elif axis[1] == 1:
            plot.set_ylabel("cheese")
        elif axis[1] == 2:
            plot.set_ylabel("pancake")


X, Y = make_classification(n_features=3, random_state=100, n_redundant=1)

complete_data = plt.figure("Complete data")
ax3 = Axes3D(complete_data)
ax3.scatter(X[:, 0], X[:, 1], X[:, 2], c=Y)

separated_data = plt.figure("Data separation selection")
a = separated_data.add_subplot(131)
b = separated_data.add_subplot(132)
c = separated_data.add_subplot(133)
add_data((a, b, c), X, Y)

plt.show()
