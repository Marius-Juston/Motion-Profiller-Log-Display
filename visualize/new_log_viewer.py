import os
from datetime import datetime

import easygui
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation, patches
from matplotlib.gridspec import GridSpec
from matplotlib.widgets import Button

import visualize
from visualize.helper import get_data, is_valid_log, sort_files, get_velocity, get_coordinates_at_center


# TODO add a slider so that you can interactively zoom through time.
# TODO start button or some other mechanism to continue animation from current slider position
# TODO make system more efficient by creating the axis only once and setting the data on the graphs only instead
# TODO show a bar in the smaller graphs to show where it is in time right now
# TODO make 'h' button popup key shortcuts
class Plot(object):

    def __init__(self, files, show_buttons=False, robot_width=.78, robot_height=.8) -> None:
        super().__init__()
        self.robot_width = robot_width
        self.robot_height = robot_height
        self.show_buttons = show_buttons
        self.files = files
        self.sorted_names = sort_files(files)

        self.fig = plt.figure(figsize=(10, 5))

        if len(self.sorted_names) > 1:
            self.fig.canvas.mpl_connect('key_press_event', self.handle_key_event)
        fig_manager = plt.get_current_fig_manager()
        fig_manager.window.showMaximized()

        if len(files) == 1 or not show_buttons:
            self.grid_rows = 3
            self.grid_column = 6
        else:
            self.grid_rows = 4
            self.grid_column = 6

        self.grid = GridSpec(self.grid_rows, self.grid_column)

        self.current_plot_index = 0
        self.animation = None
        self.plot_index(self.current_plot_index)

    def handle_key_event(self, event):
        if event.key == "right":
            self.next_figure(1)
        if event.key == "left":
            self.next_figure(-1)

    def create_axis(self):  # TODO should this only be done one?
        paths = self.fig.add_subplot(self.grid[:3, :3])
        paths.grid(True)

        velocities = self.fig.add_subplot(self.grid[0, 3:])
        velocities.set_xlabel("Time (sec)")
        velocities.set_ylabel("Velocity m/s")
        velocities.grid(True)

        errors = self.fig.add_subplot(self.grid[1, 3:])
        errors.set_xlabel("Time (sec)")
        errors.grid(True)

        powers = self.fig.add_subplot(self.grid[2, 3:])
        powers.set_xlabel("Time (sec)")
        powers.grid(True)

        return paths, velocities, errors, powers

    def next_figure(self, increment):
        new_plot_index = max(min((self.current_plot_index + increment), (len(self.sorted_names) - 1)), 0)

        if new_plot_index != self.current_plot_index:
            self.fig.clear()  # FIXME Should the figure be cleared should the buttons be removed and the data changed?

            self.current_plot_index = new_plot_index
            self.plot_index(self.current_plot_index)

    def create_buttons(self, plot_index):
        # FIXME make the button actually be able to be pressed (ome reason it does not work)
        if len(self.sorted_names) > 1 and self.show_buttons:
            if plot_index == 0:
                next_button_axis = self.fig.add_subplot(self.grid[self.grid_rows - 1, :])
                next_button = Button(next_button_axis, self.sorted_names[plot_index + 1])
                next_button.on_clicked(self.next_plot)
            elif plot_index == len(self.sorted_names) - 1:
                previous_button_axis = self.fig.add_subplot(self.grid[self.grid_rows - 1, :])
                previous_button = Button(previous_button_axis, self.sorted_names[plot_index - 1])
                previous_button.on_clicked(self.previous_plot)
            else:
                next_button_axis = self.fig.add_subplot(self.grid[self.grid_rows - 1, :int(self.grid_column / 2)])
                next_button = Button(next_button_axis, self.sorted_names[plot_index + 1])
                next_button.on_clicked(self.next_plot)

                previous_button_axis = self.fig.add_subplot(self.grid[self.grid_rows - 1, int(self.grid_column / 2):])
                previous_button = Button(previous_button_axis, self.sorted_names[plot_index - 1])
                previous_button.on_clicked(self.previous_plot)

    def next_plot(self):
        self.next_figure(1)

    def previous_plot(self):
        self.next_figure(-1)

    def plot_index(self, plot_index):
        self.fig.canvas.set_window_title(str(self.sorted_names[plot_index]))

        paths, velocities, errors, powers = self.create_axis()
        self.create_buttons(plot_index)

        current_file = self.files[self.sorted_names[plot_index]]

        self.place_path(paths, current_file)
        self.place_velocities(velocities, current_file)
        self.place_errors(errors, current_file)
        self.place_powers(powers, current_file)

        self.view_subplot_legends(paths, velocities, errors, powers)
        self.distinguish_paths(current_file, velocities, errors, powers)

        if self.animation is not None:
            self.animation.stop_animation()

        self.animation = RobotMovement(paths, current_file, robot_width=self.robot_width,
                                       robot_height=self.robot_height)
        plt.draw()

    def show(self):
        self.grid.tight_layout(self.fig)
        plt.show()

    def close_all(self):
        plt.close("all")

    def place_powers(self, powers, current_file):
        time = current_file["Time"]
        time -= time.min()

        right_power = current_file["pRight"]
        left_power = current_file["pLeft"]

        powers.plot(time, right_power, c="blue", label="Right power")
        powers.plot(time, left_power, c="red", label="Left power")

        powers.set_xlim(0, time.max())

    def place_errors(self, errors, current_file):
        time = current_file["Time"]
        time -= time.min()

        lag_error = current_file["lagE"]
        cross_track_error = current_file["XTE"]
        angle_error = current_file["angleE"]

        errors.plot(time, lag_error, c="red", label="Lag error")
        errors.plot(time, cross_track_error, c="blue", label="Cross track error")
        errors.plot(time, angle_error, c="green", label="Angle error")

        errors.set_xlim(0, time.max())

    def view_subplot_legends(self, *args):
        for subplot in args:
            handles, labels = subplot.get_legend_handles_labels()
            subplot.legend(handles, labels)

    def place_velocities(self, velocities, current_file):
        time = current_file["Time"]
        time -= time.min()

        actual_velocity = get_velocity(time, current_file)
        target_velocity = get_velocity(time, current_file, actual=False)

        # plt.text(.94, .92, "Max: {0:.4f}".format(velocity.max()), ha='center', va='center',
        #          transform=velocities.transAxes)
        velocities.plot(time, actual_velocity, c="red", label="Actual max: {0:.4f}".format(actual_velocity.max()))
        velocities.plot(time, target_velocity, c="blue",
                        label="Target max: {0:.4f}".format(target_velocity.max()))
        velocities.set_xlim(0, time.max())

    def get_range_median(self, data):
        min_value = data.min()
        max_value = data.max()
        data_range = max_value - min_value
        return data_range, (max_value + min_value) / 2

    def place_path(self, paths, current_file):
        x_target = current_file["xTarget"]
        max_range, max_x_center = self.get_range_median(x_target)

        y_target = current_file["yTarget"]
        range_1, max_y_center = self.get_range_median(y_target)
        max_range = max(max_range, range_1)

        x_actual = current_file["xActual"]

        range_1, x_center = self.get_range_median(x_actual)

        if max_range < range_1:
            max_x_center = x_center
            max_range = range_1

        y_actual = current_file["yActual"]
        range_1, y_center = self.get_range_median(y_actual)

        if max_range < range_1:
            max_y_center = y_center
            max_range = range_1

        max_range += (max(self.robot_height, self.robot_width) * 2.5)

        paths.set_xlim(max_x_center - (max_range / 2), max_x_center + (max_range / 2))
        paths.set_ylim(max_y_center - (max_range / 2), max_y_center + (max_range / 2))

        paths.plot(x_actual, y_actual, c="red", label="Actual path")
        paths.plot(x_target, y_target, c="blue", label="Target path")

    def distinguish_paths(self, current_file, *args):
        max_mins = [0]

        min_time = current_file["Time"].min()
        colors = ["red", "blue", "green"]

        for i in range(current_file["pathNumber"].min(), current_file["pathNumber"].max() + 1):
            path = current_file[current_file["pathNumber"] == i]
            max_path_time = path["Time"].max()

            max_mins.append(max_path_time - min_time)

        for plot in args:
            for i in range(1, len(max_mins)):
                plot.axvspan(max_mins[i - 1], max_mins[i], facecolor=colors[i % len(colors)], alpha=0.1)


class RobotMovement(object):
    def __init__(self, ax, data, start_index=0, robot_width=.78, robot_height=.8):
        self.start_index = start_index
        self.robot_height = robot_height
        self.robot_width = robot_width
        self.ax = ax
        ax.figure.canvas.callbacks.connect('button_press_event', self)

        self.time = data["Time"]
        self.delta_times = self.time
        self.delta_times = np.roll(self.delta_times, -1, 0) - self.delta_times
        self.delta_times[-1] = 0

        self.data = {"actual": (data["xActual"], data["yActual"], data["angleActual"]),
                     "target": (data["xTarget"], data["yTarget"], data["angleTarget"])}
        self.patches = []

    def set_interval(self, time):
        """
        :param time: Time in seconds
        :return:
        """
        self.animation.event_source.interval = time * 1000

    def animate(self, i):
        # FIXME minimal issue of knowing what time delay to use. Time difference is unnoticeable only minor issue
        i += self.start_index

        self.set_interval(self.delta_times[i])
        self.set_patch_location(i)

        if i == self.delta_times.shape[0] - 1:
            self.set_patch_visibility(False)
        return self.patches

    def set_patch_location(self, i):
        for patch, key in zip(self.patches, ("actual", "target")):
            x, y, angle = self.data[key]

            patch.set_xy(
                get_coordinates_at_center(x[i], y[i], patch.get_height(),
                                          patch.get_width(), patch.angle))
            patch.angle = np.rad2deg(angle[i])

    def create_patches(self):
        patch_actual = patches.Rectangle((0, 0), width=self.robot_width, height=self.robot_height, angle=0,
                                         fc='y', color="red")

        patch_target = patches.Rectangle((0, 0), width=.78, height=.8, angle=0,
                                         fc='y', color="blue")
        self.patches = [self.ax.add_patch(patch_actual), self.ax.add_patch(patch_target)]

        self.set_patch_location(0)

        return self.patches

    def set_patch_visibility(self, is_visible):
        for patch in self.patches:
            patch.set_visible(is_visible)

    def stop_animation(self):
        if len(self.patches) != 0:
            self.animation.event_source.stop()
            self.set_patch_visibility(False)

    def __call__(self, event):
        ax = event.inaxes

        if ax is not None and ax == self.ax:

            if event.dblclick:
                self.stop_animation()
                self.animation = animation.FuncAnimation(self.ax.figure, self.animate,
                                                         init_func=self.create_patches,
                                                         frames=self.delta_times.shape[0] - self.start_index,
                                                         interval=0,
                                                         blit=True,
                                                         repeat=False)
            else:
                self.stop_animation()


def main(open_path):
    while True:
        files = easygui.fileopenbox('Please locate csv files', 'Specify File', default=open_path, filetypes='*.csv',
                                    multiple=True)

        if files:
            open_path = "{0:s}\*.csv".format(os.path.dirname(files[0]))

            csv_files = {}

            for file in files:
                file_data = get_data(file)

                if is_valid_log(file_data):
                    try:
                        name = datetime.strptime(os.path.basename(file), "%Y-%m-%d %H-%M-%S.csv")
                    except ValueError:
                        name = os.path.basename(file).split(".")[0]
                    csv_files[name] = file_data

                else:
                    easygui.msgbox(
                        "The file {0:s} is not a valid file it will be removed.".format(os.path.basename(file)))

            plots = Plot(csv_files)
            plots.show()
            plots.close_all()
        else:
            return open_path


if __name__ == '__main__':
    main(visualize.open_path)
