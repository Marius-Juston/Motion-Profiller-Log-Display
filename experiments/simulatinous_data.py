import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import make_regression
from sklearn.preprocessing import MinMaxScaler

from visualize.selection import PointSelectors

if __name__ == '__main__':
    number_of_points = 100

    # xy = np.random.rand(number_of_points, 3)

    xy = make_regression(number_of_points, 2, 3)
    # xy = np.concatenate(xy, 1)
    xy = np.hstack((xy[0], xy[1].reshape(-1, 1)))
    xy = MinMaxScaler().fit_transform(xy)

    columns = (0, 1, 2)

    ax = plt.subplot(131)
    scat = ax.scatter(xy[:, columns[0]], xy[:, columns[1]])
    ax.set_aspect("equal")
    ax.set_xlabel("Column 0")
    ax.set_ylabel("Column 1")

    ax2 = plt.subplot(132)
    scat2 = ax2.scatter(xy[:, columns[1]], xy[:, columns[2]])
    ax2.set_aspect("equal")
    ax2.set_xlabel("Column 1")
    ax2.set_ylabel("Column 2")

    ax3 = plt.subplot(133)
    scat3 = ax3.scatter(xy[:, columns[0]], xy[:, columns[2]])
    ax3.set_aspect("equal")
    ax3.set_xlabel("Column 0")
    ax3.set_ylabel("Column 2")

    fig2 = plt.figure("Cheese")
    ax4 = Axes3D(fig2)
    scat4 = ax4.scatter(xy[:, 0], xy[:, 1], xy[:, 2])
    ax4.set_xlabel("Column 0")
    ax4.set_ylabel("Column 1")
    ax4.set_zlabel("Column 2")

    selector = None

    plots = (scat, scat2, scat3)


    def reset_plots(indexes):
        global scat, scat2, scat3, plots, selector

        selector.remove_scatter_plots(plots)

        scat = ax.scatter(xy[:, columns[0]], xy[:, columns[1]])
        scat2 = ax2.scatter(xy[:, columns[1]], xy[:, columns[2]])
        scat3 = ax3.scatter(xy[:, columns[0]], xy[:, columns[2]])

        plots = (scat, scat2, scat3)
        selector.add_scatter_plots(plots)


    selector = PointSelectors(plots, reset_plots)

    plt.show()
