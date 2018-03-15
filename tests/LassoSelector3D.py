from matplotlib.widgets import LassoSelector
from mpl_toolkits.mplot3d.art3d import Line3D


class LassoSelector3D(LassoSelector):
    """
    Selection curve of an arbitrary shape.

    For the selector to remain responsive you must keep a reference to it.

    The selected path can be used in conjunction with `~.Path.contains_point`
    to select data points from an image.

    In contrast to `Lasso`, `LassoSelector` is written with an interface
    similar to `RectangleSelector` and `SpanSelector`, and will continue to
    interact with the axes until disconnected.

    Example usage::

        ax = subplot(111)
        ax.plot(x,y)

        def onselect(verts):
            print(verts)
        lasso = LassoSelector(ax, onselect)

    Parameters
    ----------
    ax : :class:`~matplotlib.axes.Axes`
        The parent axes for the widget.
    onselect : function
        Whenever the lasso is released, the *onselect* function is called and
        passed the vertices of the selected path.
    button : List[Int], optional
        A list of integers indicating which mouse buttons should be used for
        rectangle selection. You can also specify a single integer if only a
        single button is desired.  Default is ``None``, which does not limit
        which button can be used.

        Note, typically:

        - 1 = left mouse button
        - 2 = center mouse button (scroll wheel)
        - 3 = right mouse button

    """

    def __init__(self, ax, onselect=None, useblit=True, lineprops=None,
                 button=None):
        LassoSelector.__init__(self, ax, onselect, useblit=useblit,
                               button=button)

        self.verts = None

        if lineprops is None:
            lineprops = dict()
        if useblit:
            lineprops['animated'] = True
        self.line = Line3D([], [], [], **lineprops)
        self.line.set_visible(False)
        self.ax.add_line(self.line)
        self.artists = [self.line]

    def onpress(self, event):
        self.press(event)

    def _press(self, event):
        self.verts = [self._get_data(event)]
        self.line.set_visible(True)

    def onrelease(self, event):
        self.release(event)

    def _release(self, event):
        if self.verts is not None:
            self.verts.append(self._get_data(event))
            self.onselect(self.verts)
        self.line.set_data([[], [], []])
        self.line.set_visible(False)
        self.verts = None

    def _onmove(self, event):
        if self.verts is None:
            return
        self.verts.append(self._get_data(event))

        self.line.set_data(list(zip(*self.verts)))

        self.update()
