import numpy as np
from matplotlib.path import Path
from matplotlib.widgets import LassoSelector
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Path3DCollection


def set_facecolors(axs, fcs, indx, alpha_other):
    for scat, fc in zip(axs, fcs):
        fc[:, -1] = alpha_other

        if indx is not None and indx.shape[0] != 0:
            fc[indx, -1] = 1

        if isinstance(scat, Path3DCollection):
            scat._facecolor3d = fc
            scat._edgecolor3d = fc

        else:
            scat.set_facecolors(fc)


def create_facecolors(scatters, number_of_points):
    for scatter in scatters:
        fc = scatter.get_facecolors()

        if len(fc) == 0:
            raise ValueError('Collection must have a facecolor')
        elif len(fc) == 1:
            fc = np.tile(fc, (number_of_points, 1))

        yield fc


class LassoPointSelector(LassoSelector):
    def __init__(self, scatter_plot, alpha_other=.1, mouse_button_add=1, mouse_button_remove=3):
        super().__init__(scatter_plot.axes, self.on_select)
        # SEts constantst
        self.mouse_button_remove = mouse_button_remove
        self.mouse_button_add = mouse_button_add
        self.alpha_other = alpha_other
        self.offset = scatter_plot.get_offsets()
        self.indexes = None

    # for the release motion event trigger
    def _release(self, event):
        if self.verts is not None:
            self.verts.append(self._get_data(event))
            # Custom on select method uses 3 arguments
            self.on_select(self.verts, self.offset, event)
        self.line.set_data([[], []])
        self.line.set_visible(False)
        self.verts = None

    def on_select(self, vertices, offset, event):
        # gets the button pressed
        mouse_button_pressed = event.button

        path = Path(vertices)

        # Finds the indexces that are encircled
        point_index = np.nonzero(path.contains_points(offset))[0]

        # If there are points selected
        if point_index.shape[0] != 0:

            if mouse_button_pressed == self.mouse_button_add:
                if self.indexes is None:
                    self.indexes = point_index
                else:
                    self.indexes = np.unique(np.append(self.indexes, point_index))
            elif self.indexes is not None and mouse_button_pressed == self.mouse_button_remove:
                self.indexes = np.array([*set(self.indexes).difference(point_index)],
                                        dtype=np.int64)


class PointSelectors(object):
    cids = {}
    number_of_points = 0

    def __init__(self, scatter_plots, on_release, alpha_other=.1):
        self.selectors = dict()

        self.add_scatter_plots(scatter_plots, alpha_other)
        self.on_release = on_release

        # LassoPointSelector.set_scatter_plots(scatter_plots)

    def remove_scatter_plots(self, scatter_plots):
        for scatter_plot in scatter_plots:
            ax = scatter_plot.axes

            if ax in PointSelectors.cids:
                ax.figure.canvas.mpl_disconnect(PointSelectors.cids[ax])
            del self.selectors[ax]

            scatter_plot.remove()
            ax.clear()

            LassoPointSelector._scatter_plots.remove(scatter_plot)

        # LassoPointSelector.set_scatter_plots(LassoPointSelector._scatter_plots)

    def add_scatter_plots(self, scatter_plots, alpha_other=.1):
        for scatter_plot in scatter_plots:
            ax = scatter_plot.axes
            if not isinstance(ax, Axes3D) and ax not in self.selectors:
                self.selectors[ax] = LassoPointSelector(scatter_plot, alpha_other=alpha_other)

        # must be done afterwards otherise it will casue the relsae event of the lasss selection to be called
        # after thus not allowing us to retrieve the latest index selections
        for ax in self.selectors.keys():
            PointSelectors.cids[ax] = ax.figure.canvas.mpl_connect("button_release_event", self._on_release)

        plots = LassoPointSelector._scatter_plots

        for scatter in scatter_plots:
            plots.add(scatter)
        LassoPointSelector.set_scatter_plots(plots)

        PointSelectors.number_of_points = LassoPointSelector.number_of_points

    def _on_release(self, event):
        ax = event.inaxes
        if ax is not None and ax in self.selectors:
            self.on_release(LassoPointSelector.indexes)
