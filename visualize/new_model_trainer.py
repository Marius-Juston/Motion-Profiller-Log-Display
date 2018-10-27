import os

import easygui

import visualize
from visualize.helper import get_features, is_valid_log, get_data
from visualize.processing import OutlierAndScalingSelection


def train_model(open_path):
    """

       :param open_path:
       :return:
       """
    # TODO x = motor power, y velocity, z time

    while True:
        file = easygui.fileopenbox('Please locate csv file', 'Specify File', default=open_path, filetypes='*.csv')

        if file:
            open_path = "{0:s}\*.csv".format(os.path.dirname(file))

            file_data = get_data(file)

            # TODO make this loop thought the steps as many times as they are number of paths
            if is_valid_log(file_data, visualize.LEGACY_COLUMNS):
                x, _ = get_features(file_data)

                outlier = OutlierAndScalingSelection(file_data, x)
                outlier.show()

                del outlier

        else:
            break

    # total_data = {}
    # already_used_files = set()
    # changed_anything = False
    # hyperplane = None
    #
    # if os.path.exists(MODEL_FILE):
    #     answer = easygui.boolbox("A model already exists do you wish to use it?")
    #
    #     if answer is None:
    #         return
    #

    # get_features()  # elif answer:
    #         clf = joblib.load(MODEL_FILE)
    #         # hyperplane = plot_hyperplane(clf, ax3d)
    #         data = np.load(MODEL_DATA_FILE)
    #         total_data["features"] = data["features"]
    #         total_data["labels"] = data["labels"]
    #
    #         accelerating = total_data["features"][total_data["labels"] == 0]
    #         decelerating = total_data["features"][total_data["labels"] == 1]
    #
    #         ax3d.scatter(accelerating[:, 0], accelerating[:, 1], accelerating[:, 2], c="red",
    #                      label="acceleration")
    #         ax3d.scatter(decelerating[:, 0], decelerating[:, 1], decelerating[:, 2], c="blue",
    #                      label="deceleration")
    #
    #         already_used_files.add(*data["files"])
    #
    #         plt.show()
    #     else:
    #         clf = create_blank_classifier()
    #         changed_anything = True
    # else:
    #     clf = create_blank_classifier()
    #
    # if changed_anything and not is_empty_model(clf):
    #     joblib.dump(clf, MODEL_FILE)
    #     np.savez(MODEL_DATA_FILE, features=total_data["features"], labels=total_data["labels"],
    #              files=total_data["files"])
    #     easygui.msgbox("Model saved.")
    #
    # plt.close("all")
    # return open_path


if __name__ == '__main__':
    train_model("C:")
