import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
gs = GridSpec(3, 4, fig)

# Axes3D()
master_plot = fig.add_subplot(gs[:3, :3], projection='3d')
time_velocity = fig.add_subplot(gs[0, -1])
time_power = fig.add_subplot(gs[1, -1])
power_velocity = fig.add_subplot(gs[2, -1])

gs.tight_layout(fig)

figManager = plt.get_current_fig_manager()
figManager.window.showMaximized()

plt.show()
