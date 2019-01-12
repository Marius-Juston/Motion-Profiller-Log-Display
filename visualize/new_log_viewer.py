# coding=utf-8
import os
from datetime import datetime

import easygui
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches
from matplotlib.animation import FuncAnimation
from matplotlib.axes import Axes
from matplotlib.backend_bases import KeyEvent, MouseEvent
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyArrow
from matplotlib.widgets import Button

import visualize
from visualize.helper import get_data, is_valid_log, sort_files, get_velocity, get_coordinates_at_center, \
    contains_key, get_range_middle, view_subplot_legends, rotate_points_around_point

NEEDED_KEYS = (
    "Time", "lagE", "xActual", "xTarget", "pRight", "pLeft", "yTarget", "yActual", "XTE", "angleE",
    "angleActual"
)


# TODO add a slider so that you can interactively zoom through time.
# TODO start button or some other mechanism to continue animation from current slider position
# TODO make system more efficient by creating the axis only once and setting the data on the graphs only instead
# TODO show a bar in the smaller graphs to show where it is in time right now
# TODO make 'h' button popup key shortcuts

def distinguish_paths(current_file: np.ndarray, *args: Axes) -> None:
    """
Draws colored lines to separate each path in the log file data on each plot.
    :param current_file: the log data to use find where to separate the lines
    :param args: the list of axes to use
    """
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


class DirectionalArrow(FancyArrow):
    """
Custom class for drawing an arrow in matplotlib
    """

    def __init__(self, xy, angle, total_length, width, in_degrees=False, **kwargs):
        """
        :param xy: the starting location of the arrow
        :param angle: the starting angle  of the arrow
        :param total_length: the total length of the arrow including the head
        :param width: the width of the arrow
        :param in_degrees: True if the angle given is in degrees if False then in radians
        :param kwargs: the additional arguments of FancyArrow
        """
        if in_degrees:
            angle = np.radians(angle)
        self.angle = angle
        self.length = total_length
        dx = np.cos(angle) * total_length
        dy = np.sin(angle) * total_length
        x, y = xy
        super().__init__(x, y, dx, dy, width=width, length_includes_head=True,
                         head_length=total_length / 2, **kwargs)

        self.center_xy = [x + dx / 2, y + dy / 2]

        self.set_center_xy((x, y))

    def set_angle(self, angle, in_degrees=False):
        """
    Sets the angle of the arrow
        :param angle: the new angle of the arrow
        :param in_degrees: True if the angle given is in degrees if False then in radians
        """
        if in_degrees:
            angle = np.radians(angle)
        rotate_angle = self.angle - angle

        rotated_points = rotate_points_around_point(self.get_xy(), rotate_angle, self.center_xy)
        self.set_xy(rotated_points)

        self.angle = angle

    def set_center_xy(self, xy):
        """
    Sets the center of the arrow location
        :param xy: the new x,y center coordinate of the arrow
        """
        self.set_xy(self.get_xy() + (xy[0] - self.center_xy[0], xy[1] - self.center_xy[1]))
        self.center_xy = xy


class Plot(object):
    """
Class meant to plot log file data for the Motion Profiler of Walton Robotics
    """

    def __init__(self, files: dict, show_buttons: bool = True, robot_width: float = .78,
                 robot_height: float = .8) -> None:
        """
        :param files: the log data file name, numpy data dictionary
        :param show_buttons: true if the buttons to move between plot are shown. If False then the key events are
        only used
        :param robot_width: the actual robot width. Changes the width for the animation
        :param robot_height: the actual robot height. Changes the height of the robot in the animation
        """
        super().__init__()
        self.robot_width = robot_width
        self.robot_height = robot_height
        self.show_buttons = show_buttons
        self.files = files
        self.sorted_names = sort_files(files)
        self.trim_paths()

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

        self.paths = self.fig.add_subplot(self.grid[:3, :3])
        self.powers = self.fig.add_subplot(self.grid[2, 3:])
        self.errors = self.fig.add_subplot(self.grid[1, 3:])
        self.velocities = self.fig.add_subplot(self.grid[0, 3:])

        self.format_axis()

        self.buttons_axes = []
        self.next_button = None
        self.previous_button = None

        self.animations = {}
        self.current_plot_index = 0

        self.plot_index(self.current_plot_index)

    def trim_paths(self):
        for file in self.files:
            data = self.files[file]

            if is_valid_log(data, ["pathNumber", "motionState"]):
                min_path_number = data["pathNumber"].min()

                min_remover = np.logical_and(data["pathNumber"] != min_path_number,
                                             data["motionState"] != "WAITING")

                data = data[min_remover]

                self.files[file] = data

    def clear_buttons(self):
        """
    Removes the buttons from the figure
        """
        while len(self.buttons_axes) != 0:
            self.buttons_axes.pop().remove()

    def handle_key_event(self, event: KeyEvent) -> None:
        """
    Handles the key events. If the right arrow key is pressed then then it plot the next plot if it was the left arrow
    key then it plots the previous plot.
        :param event: the key event
        """
        if event.key == "right":
            self.next_figure(1)
        if event.key == "left":
            self.next_figure(-1)

    def format_axis(self):
        """
    Sets the graphs axis labels
        :return: the graphs
        """

        self.velocities.set_xlabel("Time (sec)")
        self.velocities.set_ylabel("Velocity m/s")

        self.errors.set_xlabel("Time (sec)")

        self.powers.set_xlabel("Time (sec)")

        return self.paths, self.velocities, self.errors, self.powers

    def next_figure(self, increment: int) -> None:
        """
    Clears the axis' plots, clears the buttons and then recreates the buttons and the graphs for the next graph
        :param increment: -1 for the previous plot, 1 for the next plot
        """
        new_plot_index = max(min((self.current_plot_index + increment), (len(self.sorted_names) - 1)), 0)

        if new_plot_index != self.current_plot_index:
            self.clear_axis()
            self.clear_buttons()
            self.current_plot_index = new_plot_index
            self.plot_index(self.current_plot_index)

    def create_buttons(self, plot_index: int) -> None:
        """
    Creates the buttons on the figure if the plot index is either the first of the last index then there will only be a
    single button only allowing you to move to either the previous or the next plot.If the plot index is in between the
    first and last index then there will be two buttons created the left one goes to the previous plot and the right one
    goes to the next plot.
        :param plot_index: which log file's data to use for the graphs
        """
        if len(self.sorted_names) > 1 and self.show_buttons:
            if plot_index == 0:
                next_button_axis = self.fig.add_subplot(self.grid[self.grid_rows - 1, :])
                self.next_button = Button(next_button_axis, self.sorted_names[plot_index + 1])
                self.next_button.on_clicked(self.next_plot)

                self.buttons_axes.append(next_button_axis)
            elif plot_index == len(self.sorted_names) - 1:
                previous_button_axis = self.fig.add_subplot(self.grid[self.grid_rows - 1, :])
                self.previous_button = Button(previous_button_axis, self.sorted_names[plot_index - 1])
                self.previous_button.on_clicked(self.previous_plot)
                self.buttons_axes.append(previous_button_axis)
            else:
                next_button_axis = self.fig.add_subplot(self.grid[self.grid_rows - 1, :int(self.grid_column / 2)])
                self.next_button = Button(next_button_axis, self.sorted_names[plot_index + 1])
                self.next_button.on_clicked(self.next_plot)
                self.buttons_axes.append(next_button_axis)

                previous_button_axis = self.fig.add_subplot(self.grid[self.grid_rows - 1, int(self.grid_column / 2):])
                self.previous_button = Button(previous_button_axis, self.sorted_names[plot_index - 1])
                self.previous_button.on_clicked(self.previous_plot)
                self.buttons_axes.append(previous_button_axis)

    def next_plot(self, event: MouseEvent) -> None:
        """
    Plots the next log file's data
        :param event: the mouse event
        """
        self.next_figure(1)

    def previous_plot(self, event: MouseEvent) -> None:
        """
    Plots the previous log file's data
        :param event: the mouse event
        """
        self.next_figure(-1)

    def plot_index(self, plot_index: int) -> None:
        """
    Sets up the graphs, the buttons and plots the dat on graphs for the current plot index which indicates which log
    file's data to use
        :param plot_index: which log file's data to use for the graphs
        """
        self.fig.canvas.set_window_title(str(self.sorted_names[plot_index]))

        self.show_grid()

        self.create_buttons(plot_index)

        current_file = self.files[self.sorted_names[plot_index]]

        self.place_path(current_file)
        self.place_velocities(current_file)
        self.place_errors(current_file)
        self.place_powers(current_file)

        view_subplot_legends(self.paths, self.velocities, self.errors, self.powers)

        if contains_key(current_file, "pathNumber"):
            distinguish_paths(current_file, self.velocities, self.errors, self.powers)

        self.stop_all_animations()
        if plot_index not in self.animations:
            self.animations[plot_index] = RobotMovement(self.paths, self.velocities, self.errors, self.powers,
                                                        current_file, robot_width=self.robot_width,
                                                        robot_height=self.robot_height)
        else:
            self.animations[plot_index].enable_animation()

        plt.draw()

    def show(self):
        """
    Displays the figure in tight_layout.
        :return:
        """
        self.grid.tight_layout(self.fig)
        plt.show()

    def close_all(self):
        """
    Closes the figure.
        :return:
        """
        plt.close(self.fig)

    def place_powers(self, current_file: np.ndarray) -> None:
        """
    Places the motor powers in the powers Axes. Places the right power and the left power. The x axis is time.
        :param current_file:
        """
        time = current_file["Time"]
        time -= time.min()

        right_power = current_file["pRight"]
        left_power = current_file["pLeft"]

        self.powers.plot(time, right_power, c="blue", label="Right power")
        self.powers.plot(time, left_power, c="red", label="Left power")
        self.powers.set_xlim(0, time.max())

    def place_errors(self, current_file: np.ndarray) -> None:
        """
    Places the errors (lag error:lagE, cross track error:XTE and the angle error:angleE) in the errors Axes.
    Time is in the x axis.
        :param current_file: the data to use to plot the errors
        """
        time = current_file["Time"]
        time -= time.min()

        lag_error = current_file["lagE"]
        cross_track_error = current_file["XTE"]
        angle_error = current_file["angleE"]

        self.errors.plot(time, lag_error, c="red", label="Lag error")
        self.errors.plot(time, cross_track_error, c="blue", label="Cross track error")
        self.errors.plot(time, angle_error, c="green", label="Angle error")

        self.errors.set_xlim(0, time.max())

    def place_velocities(self, current_file: np.ndarray) -> None:
        """
    Places the velocities in the velocities Axes, plots the actual and target velocities with time as the x axis.
        :param current_file: the data to use to plot the velocities
        """
        time = current_file["Time"]
        time -= time.min()

        actual_velocity = get_velocity(time, current_file)
        target_velocity = get_velocity(time, current_file, actual=False)

        self.velocities.plot(time, actual_velocity, c="red", label="Actual max: {0:.4f}".format(actual_velocity.max()))
        self.velocities.plot(time, target_velocity, c="blue",
                             label="Target max: {0:.4f}".format(target_velocity.max()))
        self.velocities.set_xlim(0, time.max())

    def place_path(self, current_file: np.ndarray) -> None:
        """
    Places the path in the path Axes, plots the actual and target path.
        :param current_file: the data to use to plot the path
        """
        x_target = current_file["xTarget"]
        max_range, max_x_center = get_range_middle(x_target)

        y_target = current_file["yTarget"]
        range_1, max_y_center = get_range_middle(y_target)
        max_range = max(max_range, range_1)

        x_actual = current_file["xActual"]

        range_1, x_center = get_range_middle(x_actual)

        if max_range < range_1:
            max_x_center = x_center
            max_range = range_1

        y_actual = current_file["yActual"]
        range_1, y_center = get_range_middle(y_actual)

        if max_range < range_1:
            max_y_center = y_center
            max_range = range_1

        max_range += (max(self.robot_height, self.robot_width) * 2.5)

        self.paths.set_xlim(max_x_center - (max_range / 2), max_x_center + (max_range / 2))
        self.paths.set_ylim(max_y_center - (max_range / 2), max_y_center + (max_range / 2))

        self.paths.plot(x_actual, y_actual, c="red", label="Actual path")
        self.paths.plot(x_target, y_target, c="blue", label="Target path")

    def clear_axis(self):
        """
    Clear the axis that need to be cleared when changing data.
        """
        for ax in (self.paths, self.powers, self.velocities, self.errors):
            ax.cla()

    def show_grid(self):
        """
    Shows the grids for the major ticks in the plot.
        """
        for ax in (self.paths, self.powers, self.velocities, self.errors):
            ax.grid(True)

    def stop_all_animations(self):
        """
    Stops all the animations.
        :return:
        """
        if len(self.animations) > 0:
            for animation in self.animations:
                self.animations[animation].stop_animation()
                self.animations[animation].disconnect()


class RobotMovement(object):
    """
The class that handles the animation of a robot given its path data
    """

    def __init__(self, ax: Axes, velocity_ax: Axes, errors_ax: Axes, powers_ax: Axes, data: np.ndarray,
                 start_index: int = 0,
                 robot_width: float = .78,
                 robot_height: float = .8) -> None:
        """
        :param ax: the ax to draw the animation on
        :param data: the data to retrieve the position and time interval rom
        :param start_index: the index to start the animation at
        :param robot_width: the robot width
        :param robot_height: the robot height
        """
        self.powers_ax = powers_ax
        self.errors_ax = errors_ax
        self.velocity_ax = velocity_ax
        self.start_index = start_index
        self.robot_height = robot_height
        self.robot_width = robot_width
        self.ax = ax
        self.cid = ax.figure.canvas.callbacks.connect('button_press_event', self)
        self.playing = False
        self.time = data["Time"]
        self.delta_times = self.time
        self.delta_times = np.roll(self.delta_times, -1, 0) - self.delta_times
        self.delta_times[-1] = 0

        self.data = {"actual": (data["xActual"], data["yActual"], data["angleActual"]),
                     "target": (data["xTarget"], data["yTarget"], data["angleTarget"])}
        self.rectangles = []
        self.arrows = []
        self.patches = []
        self.lines = []

    def set_interval(self, time: float) -> None:
        """
    Sets the timer interval.
        :param time: interval in seconds
        """
        self.animation.event_source.interval = time * 1000

    def enable_animation(self):
        """
    Recreates a mouse listener.
        """
        self.cid = self.ax.figure.canvas.callbacks.connect('button_press_event', self)

    def restart_animation(self):
        """
    Restarts the animation. Stops the animation, disconnects its current mouse listener then creates a new listener.
        """
        self.stop_animation()
        self.disconnect()
        self.enable_animation()

    def animate(self, i: int) -> iter:
        """
    Method that is run during the animation, this method updates the patch locations and sets the timer interval to be
    correct.
        :param i: the index in the animation the animation is currently at
        :return: the patches in the animation to be drawn
        """
        # FIXME minimal issue of knowing what time delay to use. Time difference is unnoticeable only minor issue
        i += self.start_index

        self.set_interval(self.delta_times[i])
        self.set_patch_location(i)
        self.set_line_x_coordinate(self.time[i])

        if i == self.delta_times.shape[0] - 1:
            self.set_patch_visibility(False)
        return self.patches

    def set_patch_location(self, i: int) -> None:
        """
    Sets the location of the patches given its current index in the animation. Sets the x,y and angle coordinates.
        :param i: index of the animation is at
        """
        for robots, angle_indicator, key in zip(self.rectangles, self.arrows, ("actual", "target")):
            x, y, angle = self.data[key]

            robots.set_xy(
                get_coordinates_at_center(x[i], y[i], robots.get_height(),
                                          robots.get_width(), robots.angle))
            robots.angle = np.rad2deg(angle[i])

            angle_indicator.set_center_xy((x[i], y[i]))
            angle_indicator.set_angle(angle[i])

    def create_patches(self):
        """
    Creates the patches that will used during the animation.
        :return: the patches that will used during the animation
        """
        starting_xy = (0, 0)

        patch_actual = patches.Rectangle(starting_xy, width=self.robot_width, height=self.robot_height, angle=0,
                                         fc='y', color="red")

        patch_target = patches.Rectangle(starting_xy, width=.78, height=.8, angle=0,
                                         fc='y', color="blue")

        arrow_actual = DirectionalArrow(starting_xy, 0, self.robot_height, self.robot_width / 4, color="red")
        arrow_target = DirectionalArrow(starting_xy, 0, self.robot_height, self.robot_width / 4, color="blue")

        self.rectangles = [self.ax.add_patch(patch_actual), self.ax.add_patch(patch_target)]
        self.arrows = [self.ax.add_patch(arrow_actual), self.ax.add_patch(arrow_target)]

        velocities_line = self.velocity_ax.axvline(x=starting_xy[0])
        errors_line = self.errors_ax.axvline(x=starting_xy[0])
        powers_line = self.powers_ax.axvline(x=starting_xy[0])

        self.lines = [velocities_line, errors_line, powers_line]

        self.set_patch_location(0)

        self.patches.extend(self.rectangles)
        self.patches.extend(self.arrows)
        self.patches.extend(self.lines)

        return self.patches

    def set_line_x_coordinate(self, x):
        """
        Sets the coordinate of the Vertical lines on the Velocity, Powers and Error graphs
        :param x: the x coordinate to place the vertical line on
        """
        for line in self.lines:
            line.set_xdata(x)

    def set_patch_visibility(self, is_visible: bool) -> None:
        """
    Sets the patch visibility.
        :param is_visible: True means visible, False means not visible
        """
        for patch in self.patches:
            patch.set_visible(is_visible)

    def stop_animation(self):
        """
    Stops the animation running by stopping the event_source, setting the patch visibility to False,
    deleting the animation.
        """
        if len(self.rectangles) != 0 and self.playing:
            self.set_patch_visibility(False)
            self.animation.event_source.stop()
            del self.animation
            self.playing = False

    def disconnect(self):
        """
    Disconnects the mouse click listener that was created by this class.
        """
        self.ax.figure.canvas.mpl_disconnect(self.cid)

    def __call__(self, event: MouseEvent) -> None:
        """
    This is called when you click with your mouse on the current figure.
        :param event: the mouse event containing information about the mouse when calling this method
        """
        ax = event.inaxes

        if ax is not None and ax == self.ax:

            if event.dblclick:
                self.stop_animation()
                self.animation = FuncAnimation(self.ax.figure, self.animate,
                                               init_func=self.create_patches,
                                               frames=self.delta_times.shape[0] - self.start_index,
                                               interval=0,
                                               blit=True,
                                               repeat=False)
                self.playing = True


def has_motion_data(file_data):
    if file_data is None:
        return False

    if contains_key(file_data, "motionState"):
        motionless = np.alltrue(
            np.logical_or(file_data["motionState"] == "WAITING", file_data["motionState"] == "FINISHING"))

        return not motionless

    return True


def main(open_path):
    """
This is the main loop which runs until the user no selects any file.
    :param open_path: the default location to start your search
    :return: the ending location the folder search was looking at
    """
    while True:
        files = easygui.fileopenbox('Please locate csv files', 'Specify File', default=open_path, filetypes='*.csv',
                                    multiple=True)

        if files:
            open_path = "{0:s}\*.csv".format(os.path.dirname(files[0]))

            csv_files = {}

            for file in files:
                file_data = get_data(file)

                legacy_log = is_valid_log(file_data, visualize.LEGACY_COLUMNS)
                current_log = is_valid_log(file_data)

                if legacy_log or current_log:
                    show_log = True

                    if legacy_log and not current_log:
                        easygui.msgbox("Because this log ({}) is missing information that makes it optimal "
                                       "for manipulating the data efficiently results may be inaccurate"
                                       .format(os.path.basename(file)))
                    else:
                        robot_moved = has_motion_data(file_data)

                        if not robot_moved:
                            easygui.msgbox("Because the robot did not move this motion will not be shown")
                            show_log = False
                    if show_log:
                        try:
                            name = datetime.strptime(os.path.basename(file), "%Y-%m-%d %H-%M-%S.csv")
                        except ValueError:
                            name = os.path.basename(file).split(".")[0]
                        csv_files[name] = file_data
                else:
                    easygui.msgbox(
                        "The file {0:s} is not a valid file, it will not be plotted.".format(os.path.basename(file)))

            if len(csv_files) > 0:
                plots = Plot(csv_files)
                plots.show()
                plots.close_all()
        else:
            return open_path


if __name__ == '__main__':
    main(visualize.open_path)
