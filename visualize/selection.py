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


class LassoPointSelector(LassoSelector):
    __fcs = None
    indexes = None
    __canvases = set()

    __scatter_plots = None
    __select_all_points = None

    @staticmethod
    def __create_facecolors(scatters, number_of_points):

        for scatter in scatters:
            fc = scatter.get_facecolors()

            if len(fc) == 0:
                raise ValueError('Collection must have a facecolor')
            elif len(fc) == 1:
                fc = np.tile(fc, (number_of_points, 1))

            yield fc

    @classmethod
    def set_scatter_plots(cls, scatter_plots, number_of_points):
        cls.__scatter_plots = scatter_plots

        if len(scatter_plots) > 0:

            for plot in scatter_plots:
                cls.__canvases.add(plot.figure.canvas)

            cls.__fcs = tuple(cls.__create_facecolors(cls.__scatter_plots, number_of_points))

            cls.__select_all_points = np.arange(0, number_of_points)
            cls.indexes = cls.__select_all_points

    def __init__(self, scatter_plot, alpha_other=.1, mouse_button_add=1, mouse_button_remove=3):
        super().__init__(scatter_plot.axes, self.on_select)
        self.mouse_button_remove = mouse_button_remove
        self.mouse_button_add = mouse_button_add
        self.alpha_other = alpha_other
        self.offset = scatter_plot.get_offsets()

    def _release(self, event):
        if self.verts is not None:
            self.verts.append(self._get_data(event))
            self.on_select(self.verts, self.offset, event.button)
        self.line.set_data([[], []])
        self.line.set_visible(False)
        self.verts = None

    def on_select(self, vertices, offset, mouse_button_pressed):
        path = Path(vertices)

        point_index = np.nonzero(path.contains_points(offset))[0]

        if point_index.shape[0] != 0:
            if mouse_button_pressed == self.mouse_button_add:
                if LassoPointSelector.indexes is None:
                    LassoPointSelector.indexes = point_index
                else:
                    LassoPointSelector.indexes = np.append(LassoPointSelector.indexes, point_index)
            elif LassoPointSelector.indexes is not None and mouse_button_pressed == self.mouse_button_remove:
                LassoPointSelector.indexes = np.array([*set(LassoPointSelector.indexes).difference(point_index)],
                                                      dtype=np.int64)

            set_facecolors(LassoPointSelector.__scatter_plots, LassoPointSelector.__fcs, LassoPointSelector.indexes,
                           self.alpha_other)
        else:
            set_facecolors(LassoPointSelector.__scatter_plots, LassoPointSelector.__fcs, LassoPointSelector.indexes, 1)
            LassoPointSelector.indexes = LassoPointSelector.__select_all_points

        LassoPointSelector.update_figures()

    @staticmethod
    def update_figures():
        for canvas in LassoPointSelector.__canvases:
            canvas.draw_idle()


class PointSelectors(object):
    def __init__(self, scatter_plots, data_size):
        self.selectors = dict()

        for scatter_plot in scatter_plots:
            ax = scatter_plot.axes
            if not isinstance(ax, Axes3D):
                self.selectors[ax] = LassoPointSelector(scatter_plot)

        # must be done afterwards otherise it will casue the relsae event of the lasss selection to be called
        # after thus not allowing us to retrieve the latest index selections
        for ax in self.selectors.keys():
            ax.figure.canvas.mpl_connect("button_release_event", self._on_release)

        LassoPointSelector.set_scatter_plots(scatter_plots, data_size)

    def _on_release(self, event):
        ax = event.inaxes
        if ax is not None and ax in self.selectors:
            self.on_release(self.selectors[ax].indexes)

    def on_release(self, selections):
        print(selections)
