import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.widgets import Button
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
gs = GridSpec(6, 4, fig)

# Axes3D()
master_plot = fig.add_subplot(gs[:-1, :-1], projection='3d')
master_plot.set_aspect("equal")

time_velocity = fig.add_subplot(gs[:2, -1])
time_power = fig.add_subplot(gs[2:4, -1])
power_velocity = fig.add_subplot(gs[4:6, -1])
train_button = fig.add_subplot(gs[-1, :-1])

button = Button(train_button,"Train")

gs.tight_layout(fig)

figManager = plt.get_current_fig_manager()
figManager.window.showMaximized()

plt.show()
