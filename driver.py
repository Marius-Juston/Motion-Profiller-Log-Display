import os
from datetime import datetime

import easygui
import matplotlib.pyplot as plt
import numpy as np


def is_valid_log(file):
    fields = file.dtype.fields.keys()

    return all(key in fields for key in (
        "Time", "xActual", "yActual", "angleActual", "xTarget", "yTarget", "angleTarget", "XTE", "lagE", "angleE",
        "pLeft",
        "pRight"))


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


def plot_graphs(csv_files):
    fig, axs = plt.subplots(ncols=len(csv_files), nrows=3)

    if len(csv_files) == 1:
        get_axis = lambda x, index: x
    else:
        get_axis = lambda x, index: x[:, index]

    for i, date in enumerate(sort_files(csv_files)):
        path, errors, powers = get_axis(axs, i)
        current_file = csv_files[date]

        # TODO make the path plot bigger than the other two plots because it is more important
        path.set_title(date)
        path.plot(current_file["xActual"], current_file["yActual"], c="r", label="Actual")
        path.scatter(current_file["xActual"][[0, -1]], current_file["yActual"][[0, -1]], c="r")

        path.plot(current_file["xTarget"], current_file["yTarget"], c="b", label="Target")
        path.scatter(current_file["xTarget"][[0, -1]], current_file["yTarget"][[0, -1]], c="b")

        path.grid(True)
        handles, labels = path.get_legend_handles_labels()
        path.legend(handles, labels)

        errors.plot(current_file["XTE"], c="r", label="???")
        errors.plot(current_file["lagE"], c="g", label="Lag Error")
        errors.plot(current_file["angleE"], c="b", label="Angle Error")
        errors.grid(True)
        handles, labels = errors.get_legend_handles_labels()
        errors.legend(handles, labels)
        errors.set_xlim(0)

        powers.plot(current_file["pLeft"], c="r", label="Left Power")
        powers.plot(current_file["pRight"], c="b", label="Left Power")
        powers.grid(True)
        handles, labels = powers.get_legend_handles_labels()
        powers.legend(handles, labels)
        powers.set_xlim(0)

    plt.show()


def main():
    openPath = "{0:s}\*.csv".format(os.path.expanduser("~"))

    while True:
        # print(openPath)

        files = easygui.fileopenbox('Please locate csv files', 'Specify File', default=openPath, filetypes='*.csv',
                                    multiple=True)

        if files:
            openPath = "{0:s}\*.csv".format(os.path.dirname(files[0]))

            csv_files = {}

            for file in files:
                file_data = np.genfromtxt(file, delimiter=',', dtype=np.float32, names=True)

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
            break


if __name__ == '__main__':
    main()
