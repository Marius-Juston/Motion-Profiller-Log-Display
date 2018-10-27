import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from legacy.selection import PointSelectors

fig = plt.figure("2D Data")

rows = 2
columns = 3

number_of_column_elemnts = 2
number_of_row_elements = 3

half_column = columns // number_of_row_elements

row_division = rows // number_of_column_elemnts
gs = GridSpec(rows, columns, fig)

if columns < rows:

    fig.add_subplot(gs[:row_division, :half_column]).set_aspect("equal")
    fig.add_subplot(gs[row_division: row_division * 2, :half_column]).set_aspect("equal")
    fig.add_subplot(gs[row_division * 2:, :half_column]).set_aspect("equal")

    fig.add_subplot(gs[0:row_division, half_column:]).set_aspect("equal")
    fig.add_subplot(gs[row_division: row_division * 2, half_column:]).set_aspect("equal")
    fig.add_subplot(gs[row_division * 2:, half_column:]).set_aspect("equal")
else:

    fig.add_subplot(gs[:row_division, :half_column]).set_aspect("equal")
    fig.add_subplot(gs[:row_division, half_column:half_column * 2]).set_aspect("equal")
    fig.add_subplot(gs[:row_division, half_column * 2:]).set_aspect("equal")

    fig.add_subplot(gs[row_division:, :half_column]).set_aspect("equal")
    fig.add_subplot(gs[row_division:, half_column:half_column * 2]).set_aspect("equal")
    ax = fig.add_subplot(gs[row_division:, half_column * 2:])
    ax.set_aspect("equal")
    s = ax.scatter([1, 2, 3], [1, 2, 4])

    sp = PointSelectors((s,), lambda x: print(x))

# fig = plt.figure("Total Data 3D")
# master_plot = Axes3D(fig)

gs.tight_layout(fig)

figManager = plt.get_current_fig_manager()
figManager.window.showMaximized()

plt.show()
