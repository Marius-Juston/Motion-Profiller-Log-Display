import os

import easygui
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

import visualize
from visualize import LEGACY_COLUMNS
from visualize.helper import is_valid_log, get_data, get_features, plot_hyperplane
from visualize.new_selector import *


def go_through_process(all_features: np.ndarray, all_data: np.ndarray):
    selector = remove_outliers(all_features)

    all_features = all_features[selector.indexes]
    all_data = all_data[selector.indexes]

    all_features, features, selector = manipulate_features_and_remove_outliers(all_features, all_data)

    all_features = all_features[selector.indexes]
    all_data = all_data[selector.indexes]

    selector = select_accelerating_vs_decelerating(all_features)

    clf = SVC()
    clf.fit(all_features, selector.get_labels())

    graphs = Graphs(all_features, title="Hyperplane")
    plot_hyperplane(clf, graphs.all_features_axes)

    plt.show()


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
            all_data = None

            for file in files:
                file_data = get_data(file)

                if is_valid_log(file_data, LEGACY_COLUMNS):
                    features, _ = get_features(file_data)

                    time = MinMaxScaler().fit_transform(features[:, 2].reshape(-1, 1))

                    features[:, 2] = time.reshape(1, -1)

                    if all_features is None:
                        all_features = features
                        all_data = file_data
                    else:
                        all_features = np.concatenate((all_features, features))

                        file_data['pathNumber'] += (all_data["pathNumber"].max() + 1)

                        all_data = np.concatenate((all_data, file_data))
                else:
                    easygui.msgbox(
                        "The file {0:s} is not a valid file, it will not be plotted.".format(os.path.basename(file)))

            go_through_process(all_features, all_data)
        else:
            break


if __name__ == '__main__':
    train_model(open_path=visualize.open_path)
