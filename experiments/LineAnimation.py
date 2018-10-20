import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec

fig = plt.gcf()
gs = GridSpec(3, 1, fig)

velocity = fig.add_subplot(gs[0])
errors = fig.add_subplot(gs[1])
powers = fig.add_subplot(gs[2])

line = velocity.axvline(x=0)
line2 = errors.axvline(x=0)
line3 = powers.axvline(x=0)


def update_line(i):
    x = 1 / (i + 1)

    line.set_xdata(x)
    line2.set_xdata(x)
    line3.set_xdata(x)
    print("Hello")

    return [line, line2, line3]


animation = FuncAnimation(fig, update_line,
                          blit=True,
                          repeat=False)

plt.show()
