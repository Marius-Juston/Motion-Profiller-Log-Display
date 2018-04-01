import math

import matplotlib.pyplot as plt
# fig = plt.figure("Path")
# path = fig.gca()
#
# fig = plt.figure("Numbers")
# errors, powers, velocity = fig.subplots(3)
# plt.show()
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyArrow
from matplotlib.widgets import Button

from visualize.helper import rotate_points_around_point

fig = plt.figure()
gs = GridSpec(4, 6, fig)
path = fig.add_subplot(gs[:3, :3])
velocity = fig.add_subplot(gs[0, 3:])
errors = fig.add_subplot(gs[1, 3:])
powers = fig.add_subplot(gs[2, 3:])

# slider = Slider(fig.add_subplot(gs[3, :]), "Hello", 0, 1)

previous_plot = fig.add_subplot(gs[3, :3])
prev_button = Button(previous_plot, "Previous")
next_plot = fig.add_subplot(gs[3, 3:])
next_button = Button(next_plot, "Next")
next_button.on_clicked(next_button)

plt.draw()

gs.tight_layout(fig)

figManager = plt.get_current_fig_manager()


class DirectionalArrow(FancyArrow):

    def __init__(self, x, y, angle, total_length, in_degrees=False, width=0.001, head_width=None,
                 shape='full', overhang=0, head_starts_at_zero=False, **kwargs):
        if in_degrees:
            angle = math.radians(angle)
        self.angle = angle
        self.length = total_length
        dx = math.cos(angle) * total_length
        dy = math.sin(angle) * total_length
        super().__init__(x, y, dx, dy, width=width, length_includes_head=True,
                         head_width=head_width, head_length=total_length / 2, shape=shape, overhang=overhang,
                         head_starts_at_zero=head_starts_at_zero, **kwargs)

        # self.x =

        self.center_xy = [x + dx / 2, y + dy / 2]

        self.set_center_xy((x, y))

    def set_angle(self, angle, in_degrees=False):
        path.axhline(self.center_xy[1])
        path.axvline(self.center_xy[0])

        if in_degrees:
            angle = math.radians(angle)
        rotate_angle = self.angle - angle

        rotated_points = rotate_points_around_point(self.get_xy(), self.center_xy, rotate_angle)
        self.set_xy(rotated_points)

        self.angle = angle

    def set_center_xy(self, xy):
        self.set_xy(self.get_xy() + (xy[0] - self.center_xy[0], xy[1] - self.center_xy[1]))
        self.center_xy = xy


# arrow = DirectionalArrow(x * 2, y, 0, width / 2, head_length=width / 2, width=width / 4, color="red")

# arrow.set_xy(arrow.get_xy() - (width / 2, 0))

# col = PatchCollection((rectangle, arrow), match_original=True)

# path.add_collection(col)

# rectangle = Rectangle((x, y), width, height, color="blue")

figManager.window.showMaximized()
width = .5
height = .4
x = .5
y = .5
arrow = DirectionalArrow(x, y, 90, height, in_degrees=True,
                         width=width / 4, color="red")

arrow.set_center_xy((.3, .4))
arrow.set_angle(180, in_degrees=True)
path.add_patch(arrow)

path.minorticks_on()
path.grid(True, which="both")
plt.show()
