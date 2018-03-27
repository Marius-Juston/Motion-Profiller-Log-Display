import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

fig = plt.figure()
gs = GridSpec(5, 8, fig)
path = fig.add_subplot(gs[:-1, :4])
powers = fig.add_subplot(gs[:-1, 4:])
text_1 = fig.add_subplot(gs[-1, :])
text_1.set_visible(False)
fig.text(5, 5, 'matplotlib', ha='left', va='top')

gs.tight_layout(fig)

figManager = plt.get_current_fig_manager()
figManager.window.showMaximized()

plt.show()
