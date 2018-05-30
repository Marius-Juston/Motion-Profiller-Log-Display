import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path
from matplotlib.widgets import LassoSelector
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Path3DCollection
from sklearn.datasets import make_regression


def set_facecolors(axs, fcs, indx, alpha_other):
    for scat, fc in zip(axs, fcs):
        if isinstance(scat, Path3DCollection):
            fc[:, -1] = alpha_other
            fc[indx, -1] = 1

            scat._facecolor3d = fc
            scat._edgecolor3d = fc

        else:
            fc[:, -1] = alpha_other
            fc[indx, -1] = 1
            scat.set_facecolors(fc)


class LassoSelectors(LassoSelector):
    fcs = None
    scatter_plots = None
    indx = None

    def __init__(self, main_ax, offset, alpha_other=.1, mouse_button_add=1, mouse_button_remove=3):
        super().__init__(main_ax, self.onselect)
        self.mouse_button_remove = mouse_button_remove
        self.mouse_button_add = mouse_button_add
        self.alpha_other = alpha_other
        self.offset = offset

    def onpress(self, event):
        self._press(event)

    def _release(self, event):
        if self.verts is not None:
            self.verts.append(self._get_data(event))
            self.onselect(self.verts, self.offset, event.button)
        self.line.set_data([[], []])
        self.line.set_visible(False)
        self.verts = None

    def onselect(self, verts, offset, mouse_button_pressed):
        path = Path(verts)

        point_index = np.nonzero(path.contains_points(offset))[0]

        if point_index.shape[0] != 0:
            if mouse_button_pressed == self.mouse_button_add:
                if LassoSelectors.indx is None:
                    LassoSelectors.indx = point_index
                else:
                    LassoSelectors.indx = np.append(LassoSelectors.indx, point_index)
            elif LassoSelectors.indx is not None and mouse_button_pressed == self.mouse_button_remove:
                LassoSelectors.indx = np.array([*set(LassoSelectors.indx).difference(point_index)])

            set_facecolors(LassoSelectors.scatter_plots, LassoSelectors.fcs, LassoSelectors.indx, self.alpha_other)
        else:
            set_facecolors(LassoSelectors.scatter_plots, LassoSelectors.fcs, LassoSelectors.indx, 1)
            LassoSelectors.indx = None

        LassoSelectors.update_facecolors()

    @staticmethod
    def update_facecolors():
        for plot in LassoSelectors.scatter_plots:
            plot.figure.canvas.draw_idle()


if __name__ == '__main__':

    number_of_points = 100

    # xy = np.random.rand(number_of_points, 3)

    xy = make_regression(number_of_points, 2, 3)
    # xy = np.concatenate(xy, 1)
    xy = np.hstack((xy[0], xy[1].reshape(-1, 1)))

    columns = (0, 1, 2)

    ax = plt.subplot(311)
    scat = ax.scatter(xy[:, columns[0]], xy[:, columns[1]])
    ax.set_xlabel("Column 0")
    ax.set_ylabel("Column 1")

    ax2 = plt.subplot(312)
    scat2 = ax2.scatter(xy[:, columns[1]], xy[:, columns[2]])
    ax2.set_xlabel("Column 1")
    ax2.set_ylabel("Column 2")

    ax3 = plt.subplot(313)
    scat3 = ax3.scatter(xy[:, columns[0]], xy[:, columns[2]])
    ax3.set_xlabel("Column 0")
    ax3.set_ylabel("Column 2")

    fig2 = plt.figure("Cheese")
    ax4 = Axes3D(fig2)
    scat4 = ax4.scatter(xy[:, 0], xy[:, 1], xy[:, 2])
    ax4.set_xlabel("Column 0")
    ax4.set_ylabel("Column 1")
    ax4.set_zlabel("Column 2")

    offsets1 = scat.get_offsets()
    offsets2 = scat2.get_offsets()
    offsets3 = scat3.get_offsets()
    offsets4 = scat4.get_offsets()


    def update_facecolors(scatters):

        for scatter in scatters:
            fc = scatter.get_facecolors()

            if len(fc) == 0:
                raise ValueError('Collection must have a facecolor')
            elif len(fc) == 1:
                fc = np.tile(fc, (number_of_points, 1))

            yield fc


    LassoSelectors.scatter_plots = (scat, scat2, scat3, scat4)
    LassoSelectors.fcs = tuple(update_facecolors(LassoSelectors.scatter_plots))

    lasso = LassoSelectors(ax, offsets1)
    lasso2 = LassoSelectors(ax2, offsets2)
    lasso3 = LassoSelectors(ax3, offsets3)

    information = {ax: (lasso, "1"), ax2: (lasso2, "2"), ax3: (lasso3, "3")}


    def accept(event):
        if event.inaxes is not None:
            print(information[event.inaxes][0].indx)


    ax.figure.canvas.mpl_connect("button_release_event", accept)

    plt.show()
