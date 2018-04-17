import matplotlib.pyplot as plt
import numpy as np

xy = np.random.rand(10, 3)

columns = (0, 1)

ax = plt.subplot(121)
scat = ax.scatter(xy[:, columns[0]], xy
[:, columns[1]])

ax2 = plt.subplot(122)
scat2 = ax2.scatter(xy[:, columns[0]], xy[:, columns[1]])
plt.ion()
plt.show()


def update_plots(scatter_plots, xy, columns):
    offsets = scat.get_offsets()

    xy = xy[np.all(xy[:, columns] == offsets, 1)]

    for plot in scatter_plots:
        plot.set_offsets(xy[:, columns])

        # plot.autoscale()
    plt.draw()


update_plots((scat, scat2), xy, columns)

plt.ioff()
plt.show()
