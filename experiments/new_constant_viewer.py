import os

import easygui
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.externals import joblib

import visualize
from visualize import MODEL_FILE
from visualize.helper import is_empty_model


class ConstantViewer(object):

    def __init__(self) -> None:
        super().__init__()

        self.fig = plt.figure("Scaled 3d data")
        gs = GridSpec(3, 4, self.fig)

        # Axes3D()
        master_plot = self.fig.add_subplot(gs[:3, :3], projection='3d')
        time_velocity = self.fig.add_subplot(gs[0, -1])
        time_power = self.fig.add_subplot(gs[1, -1])
        power_velocity = self.fig.add_subplot(gs[2, -1])

        gs.tight_layout(self.fig)

        fig_manager = plt.get_current_fig_manager()
        fig_manager.window.showMaximized()

    def show(self):
        self.fig.show()


def find_constants(open_path):
    if not os.path.exists(MODEL_FILE):
        easygui.msgbox("There are no models to use to classify the data. Please train algorithm first.")
        return

    clf = joblib.load(MODEL_FILE)

    if is_empty_model(clf):
        easygui.msgbox("The model has not been fitted yet. Please add training data to the model.")
        return

    plot = ConstantViewer()

    while True:
        file = easygui.fileopenbox('Please locate csv file', 'Specify File', default=open_path, filetypes='*.csv')

        if file:
            plot.show()
            open_path = "{0:s}\*.csv".format(os.path.dirname(file))

            file_data = get_data(file)

            if is_valid_log(file_data):
                ax2d.cla()
                ax3d.cla()

                plot_hyperplane(clf, ax3d)

                k_v, k_k, k_acc = find_gain(clf, file_data, is_data=True, ax3d=ax3d, ax2d=ax2d)

                # TODO ask user to give the max acceleration of the current spline
                # TODO scale k_acc / ()
                plt.show()

                easygui.msgbox("""
                The kV of this log is {0:f}.
                The kK of this log is {1:f}.
                The kAcc of this log is {2:f}.""".format(k_v, k_k, k_acc))
            else:
                easygui.msgbox(
                    "The file {0:s} is not a valid file.".format(os.path.basename(file)))

        else:
            break

    plt.ioff()
    plt.close("all")
    return open_path


if __name__ == '__main__':
    find_constants(open_path=visualize.open_path)
