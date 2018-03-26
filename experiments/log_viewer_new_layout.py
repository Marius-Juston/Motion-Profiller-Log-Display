import matplotlib.pyplot as plt

# fig = plt.figure("Path")
# path = fig.gca()
#
# fig = plt.figure("Numbers")
# errors, powers, velocity = fig.subplots(3)
# plt.show()
from matplotlib.gridspec import GridSpec
from matplotlib.widgets import Button

fig = plt.figure()
gs = GridSpec(4, 6, fig)
path = fig.add_subplot(gs[:3, :3])
velocity = fig.add_subplot(gs[0, 3:])
errors = fig.add_subplot(gs[1, 3:])
powers = fig.add_subplot(gs[2, 3:])

previous_plot = fig.add_subplot(gs[3, :3])
prev_button = Button(previous_plot, "Previous")
next_plot = fig.add_subplot(gs[3, 3:])
next_button = Button(next_plot, "Next")

gs.tight_layout(fig)

figManager = plt.get_current_fig_manager()
figManager.window.showMaximized()

plt.show()
