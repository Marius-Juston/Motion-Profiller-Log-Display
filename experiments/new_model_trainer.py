import os

import easygui
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler

import visualize
from visualize import LEGACY_COLUMNS
from visualize.helper import is_valid_log, get_data, get_features
from visualize.new_selector import Graphs, LassoManager


def train_model(open_path):
    """

       :param open_path:
       :return:
       """
    # TODO x = motor power, y velocity, z time

    while True:
        files = easygui.fileopenbox('Please locate csv file', 'Specify File', default=open_path, filetypes='*.csv',
                                    multiple=True)

        if files:
            open_path = "{0:s}\*.csv".format(os.path.dirname(files[0]))

            all_features = None

            for file in files:
                file_data = get_data(file)

                if is_valid_log(file_data, LEGACY_COLUMNS):
                    features, _ = get_features(file_data)

                    time = MinMaxScaler().fit_transform(features[:, 2].reshape(-1, 1))

                    features[:, 2] = time.reshape(1, -1)

                    if all_features is None:
                        all_features = features
                    else:
                        all_features = np.concatenate((all_features, features))
                else:
                    easygui.msgbox(
                        "The file {0:s} is not a valid file, it will not be plotted.".format(os.path.basename(file)))

            # time = MinMaxScaler().fit_transform(all_features[:, 2].reshape(-1, 1))
            #
            # all_features[:, 2] = time.reshape(1, -1)

            grpahs = Graphs(all_features)
            selector = LassoManager(grpahs)
            plt.show()
        else:
            break


if __name__ == '__main__':
    train_model(open_path=visualize.open_path)
