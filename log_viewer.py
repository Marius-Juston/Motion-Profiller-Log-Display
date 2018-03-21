import os
from datetime import datetime

import easygui
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec, patches, animation

import helper
from helper import set_visible, is_valid_log

plt.rc_context()


def get_coordinates_at_center(x_current, y_current, height, width, angle):
    angle = np.math.atan2((height / 2), (width / 2)) + np.deg2rad(angle)

    d = np.sqrt((width / 2) ** 2 + (height / 2) ** 2)
    sin = np.cos(angle)
    x = ((d * sin) if sin != 0 else 0)
    x = x_current - x
    cos = np.sin(angle)
    y = ((d * cos) if cos != 0 else 0)
    y = y_current - y

    return x, y


class AnimationDisplay(object):
    def __init__(self, fig, association):
        self.association = association
        self.fig = fig
        fig.canvas.callbacks.connect('button_press_event', self)
        self.patches = {}
        self.animations = {}

    def set_interval(self, time, ax):
        """
        :param time: Time in seconds
        :return:
        """
        self.animations[ax].event_source.interval = time * 1000

    def animate(self, i, delta_times, file_data, ax):
        # FIXME minimal issue of knowing what time delay to use. Time difference is unnoticeable only minor issue
        self.set_interval(delta_times[i], ax)
        patch_actual, patch_target = self.patches[ax]

        patch_actual.set_xy(
            get_coordinates_at_center(file_data["xActual"][i], file_data["yActual"][i], patch_actual.get_height(),
                                      patch_actual.get_width(), patch_actual.angle))
        patch_actual.angle = np.rad2deg(file_data["angleActual"][i])

        patch_target.set_xy(
            get_coordinates_at_center(file_data["xTarget"][i], file_data["yTarget"][i], patch_target.get_height(),
                                      patch_target.get_width(), patch_target.angle))
        patch_target.angle = np.rad2deg(file_data["angleTarget"][i])

        if i == file_data.shape[0] - 1:
            patch_actual.set_visible(False)
            patch_target.set_visible(False)
        return patch_actual, patch_target,

    def __call__(self, event):

        ax = event.inaxes

        # TODO: Add event for when the event.button == 3 display the graphs associated with the g
        # raph clicked on on a new figure

        if ax is not None and ax in self.association and event.dblclick:
            file_data = self.association[ax]

            delta_times = file_data["Time"]
            delta_times = np.roll(delta_times, -1, 0) - delta_times
            delta_times[-1] = 0

            if ax not in self.patches:
                # TODO fix bug where there are two patches on the animation. One is moving
                #  the other is not. Only want one.

                patch_actual = patches.Rectangle((0, 0), width=.78, height=.8, angle=0,
                                                 fc='y', color="red")

                patch_target = patches.Rectangle((0, 0), width=.78, height=.8, angle=0,
                                                 fc='y', color="blue")
                self.patches[ax] = [ax.add_patch(patch_actual), ax.add_patch(patch_target)]
            else:
                set_visible(self.patches[ax], True)
                self.animations[ax].event_source.stop()

            self.animations[ax] = animation.FuncAnimation(self.fig, self.animate,
                                                          frames=len(delta_times),
                                                          interval=0,  # FIXME start at 0?
                                                          fargs=(delta_times, file_data, ax),
                                                          blit=True, repeat=False)


def sort_files(csv_files):
    """
    Returns a sorted list of keys. The dates will be sorted first and with latest first then
    the non date objects are added and sorted alphabetically
    :param csv_files: the dictionary to have the keys sorted
    :return: a list with the keys for the dictionary sorted
    """
    rest = []
    result = []

    for file in csv_files.keys():
        if isinstance(file, datetime):
            result.append(file)
        else:
            rest.append(file)

    result = sorted(result, reverse=True)
    result.extend(sorted(rest))

    return result


def get_axis(grid_spec, fig, index):
    path = fig.add_subplot(grid_spec[0:2, index * 2: index * 2 + 2])
    errors = fig.add_subplot(grid_spec[-1, index * 2])
    powers = fig.add_subplot(grid_spec[-1, index * 2 + 1])

    return path, errors, powers


def plot_graphs(csv_files):
    # fig, axs = plt.subplots(ncols=len(csv_files), nrows=3)
    fig = plt.figure()
    gs = gridspec.GridSpec(3, 2 * len(csv_files))
    association = {}

    for i, date in enumerate(sort_files(csv_files)):
        path, errors, powers = get_axis(gs, fig, i)
        current_file = csv_files[date]
        association[path] = current_file

        # TODO make the path plot bigger than the other two plots because it is more important
        path.set_title(date)
        path.plot(current_file["xActual"], current_file["yActual"], c="r", label="Actual")
        path.scatter(current_file["xActual"][[0, -1]], current_file["yActual"][[0, -1]], c="r")

        path.plot(current_file["xTarget"], current_file["yTarget"], c="b", label="Target")
        path.scatter(current_file["xTarget"][[0, -1]], current_file["yTarget"][[0, -1]], c="b")

        path.grid(True)
        handles, labels = path.get_legend_handles_labels()
        path.legend(handles, labels)

        # Uncomment this code to have the x and y scales the same
        min_dimension = min(current_file["xActual"].min(), current_file["xTarget"].min(), current_file["yActual"].min(),
                            current_file["yTarget"].min())
        max_dimension = max(current_file["xActual"].max(), current_file["xTarget"].max(), current_file["yActual"].max(),
                            current_file["yTarget"].max())

        path.set_xlim(xmin=-.1 + min_dimension, xmax=.1 + max_dimension)
        path.set_ylim(ymin=-.1 + min_dimension, ymax=.1 + max_dimension)
        path.set_aspect("equal", "box")

        # Uncomment this code to have arrows on the target line that shows the angle the robot should be in
        # size = len(current_file["xActual"])
        # print(size)
        # print(find_largest_factor(size))
        # for index in range(0, size, find_largest_factor(size)):
        #     angle = current_file["angleActual"][index]
        #
        #     length = .00000000000001
        #
        #     if angle == 0:
        #         angle = .000000000001
        #
        #     x = length / np.math.cos(angle)
        #     y = length / np.math.sin(angle)
        #
        #     path.arrow(current_file["xActual"][index], current_file["yActual"][index], y, x, fc="k", ec="k",
        #                head_width=0.05, head_length=0.1)

        errors.plot(current_file["Time"], current_file["XTE"], c="r", label="Cross Track Error")
        errors.plot(current_file["Time"], current_file["lagE"], c="g", label="Lag Error")
        errors.plot(current_file["Time"], current_file["angleE"], c="b", label="Angle Error")
        errors.grid(True)
        handles, labels = errors.get_legend_handles_labels()
        errors.legend(handles, labels)
        # errors.set_xlim(0)

        powers.plot(current_file["Time"], current_file["pLeft"], c="r", label="Left Power")
        powers.plot(current_file["Time"], current_file["pRight"], c="b", label="Right Power")
        powers.grid(True)
        handles, labels = powers.get_legend_handles_labels()
        powers.legend(handles, labels)
        # powers.set_xlim(0)

        # TODO look at the velocities over time

    gs.tight_layout(fig)

    AnimationDisplay(fig, association)

    plt.show()


def main(open_path):
    while True:
        files = easygui.fileopenbox('Please locate csv files', 'Specify File', default=open_path, filetypes='*.csv',
                                    multiple=True)

        if files:
            open_path = "{0:s}\*.csv".format(os.path.dirname(files[0]))

            csv_files = {}

            for file in files:
                file_data = helper.get_data(file)

                if is_valid_log(file_data):
                    try:
                        name = datetime.strptime(os.path.basename(file), "%Y-%m-%d %H-%M-%S.csv")
                    except ValueError:
                        name = os.path.basename(file).split(".")[0]
                    csv_files[name] = file_data
                else:
                    easygui.msgbox(
                        "The file {0:s} is not a valid file it will be removed.".format(os.path.basename(file)))

            plot_graphs(csv_files)

        else:
            return open_path


if __name__ == '__main__':
    main(helper.open_path)
