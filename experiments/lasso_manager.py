import matplotlib.pyplot as plt
import numpy as np
from matplotlib import path
from matplotlib.collections import RegularPolyCollection
from matplotlib.colors import colorConverter
from matplotlib.widgets import Lasso
from numpy import nonzero
from numpy.random import rand


class Datum(object):
    colorin = colorConverter.to_rgba('red')
    colorout = colorConverter.to_rgba('blue')

    def __init__(self, x, y, include=False):
        self.x = x
        self.y = y
        if include:
            self.color = self.colorin
        else:
            self.color = self.colorout


class LassoManager(object):
    def __init__(self, ax, data):
        self.axes = ax
        self.canvas = ax.figure.canvas
        self.data = data

        self.Nxy = len(data)

        facecolors = [d.color for d in data]
        self.xys = [(d.x, d.y) for d in data]
        self.ind = []
        fig = ax.figure
        self.collection = RegularPolyCollection(
            fig.dpi, 6, sizes=(100,),
            facecolors=facecolors,
            offsets=self.xys,
            transOffset=ax.transData)

        ax.add_collection(self.collection)

        self.cid = self.canvas.mpl_connect('button_press_event', self.onpress)

    def callback(self, verts):
        facecolors = self.collection.get_facecolors()
        p = path.Path(verts)
        ind = p.contains_points(self.xys)
        self.ind = nonzero([p.contains_point(xy) for xy in self.xys])[0]
        for i in range(len(self.xys)):
            if ind[i]:
                facecolors[i] = colorConverter.to_rgba('red')
            #                print ind
            else:
                facecolors[i] = colorConverter.to_rgba('blue')

        self.canvas.draw_idle()
        self.canvas.widgetlock.release(self.lasso)
        del self.lasso

    def onpress(self, event):
        if self.canvas.widgetlock.locked():
            return
        if event.inaxes is None:
            return
        self.lasso = Lasso(event.inaxes,
                           (event.xdata, event.ydata),
                           self.callback)
        # acquire a lock on the widget drawing
        self.canvas.widgetlock(self.lasso)


data = np.random.rand(5, 5)
fig, ax = plt.subplots()
# No, no need for collection
ax.imshow(data, aspect='auto', origin='lower', picker=True)
data = [Datum(*xy) for xy in rand(10, 2)]
lman = LassoManager(ax, data)
plt.show()
