# coding=utf-8
import os

import easygui
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from sklearn.ensemble import IsolationForest
from sklearn.externals import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

import visualize
from visualize import MODEL_FILE, MODEL_DATA_FILE, DTYPE, OUTLIER, ACCELERATING, DECELERATING
from visualize.helper import is_empty_model, is_valid_log, get_data, plot_hyperplane, get_features, \
    find_linear_best_fit_line


def get_labels(file_data):
    """

    :param file_data:
    :return:
    """
    return file_data["classification"]


def has_classification(file_data):
    """

    :param file_data:
    :return:
    """
    return "classification" in file_data.dtype.fields


def find_constants(open_path):
    """

    :param open_path:
    :return:
    """
    if not os.path.exists(MODEL_FILE):
        easygui.msgbox("There are no models to use to classify the data. Please train algorithm first.")
        return

    clf = joblib.load(MODEL_FILE)

    if is_empty_model(clf):
        easygui.msgbox("The model has not been fitted yet. Please add training data to the model.")
        return

    fig = plt.figure("Scaled 3d data")
    ax3d = Axes3D(fig)
    fig, ax2d = plt.subplots(1, 1, num="Fitted data")

    # plt.ion()

    while True:
        file = easygui.fileopenbox('Please locate csv file', 'Specify File', default=open_path, filetypes='*.csv')

        if file:
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


def find_gain(clf, file_data, is_data=False, ax3d=None, ax2d=None):
    """

    :param clf:
    :param file_data:
    :param is_data:
    :param ax3d:
    :param ax2d:
    :return:
    """
    if not is_data:
        file_data = np.genfromtxt(file_data, delimiter=',', dtype=DTYPE, names=True)

    x, _ = get_features(file_data)
    x = x[file_data["motionState"] == 'MOVING']

    out = IsolationForest(n_jobs=-1, random_state=0)
    out.fit(x)
    predicted = out.predict(x)
    x = x[predicted == 1]
    x_scaled = MinMaxScaler().fit_transform(x)
    predicted = clf.predict(x_scaled)

    acceleration = x[predicted == 0]
    average_power_accelerating = acceleration[:, 0]
    velocity_accelerating = acceleration[:, 1]

    deceleration = x[predicted == 1]
    average_power_decelerating = deceleration[:, 0]
    velocity_decelerating = deceleration[:, 1]

    accelerating_coefficient, accelerating_intercept = find_linear_best_fit_line(velocity_accelerating,
                                                                                 average_power_accelerating)
    decelerating_coefficient, decelerating_intercept = find_linear_best_fit_line(velocity_decelerating,
                                                                                 average_power_decelerating)
    k_v = (accelerating_coefficient + decelerating_coefficient) / 2
    k_k = (accelerating_intercept + decelerating_intercept) / 2

    acceleration_coefficient = (accelerating_coefficient - k_v)
    acceleration_intercept = (accelerating_intercept - k_k)
    k_acc = ((x[:, 1].max() - x[:, 1].min()) / 2) * acceleration_coefficient + acceleration_intercept

    if ax3d or ax2d:
        colors = ["red" if i == 0 else "blue" for i in predicted]

        if ax3d:
            ax3d.set_xlabel('Velocity')
            ax3d.set_ylabel('Average motor power')
            ax3d.set_zlabel('Scaled Time')

            scaled_average_power = np.hstack(x_scaled[:, 1])
            scaled_velocity = np.hstack(x_scaled[:, 0])
            time = np.hstack(x_scaled[:, 2])
            ax3d.scatter(scaled_velocity, scaled_average_power, time, c=colors)

        if ax2d:
            ax2d.set_xlabel('Velocity')
            ax2d.set_ylabel('Average motor power')
            velocity = x[:, 1]
            average_power = x[:, 0]
            ax2d.scatter(velocity, average_power, c=colors)

            y_lim = np.array(ax2d.get_ylim())
            # TODO make the lines not exceed the x limit as well

            for c, i in zip([k_v, accelerating_coefficient, decelerating_coefficient],
                            [k_k, accelerating_intercept, decelerating_intercept]):
                ax2d.plot((y_lim - i) / c, y_lim)

    return k_v, k_k, k_acc


def create_blank_classifier():
    """

    :return:
    """
    return SVC(kernel="rbf", random_state=0)


def train_model(open_path):
    """

    :param open_path:
    :return:
    """
    # TODO add lasso selection of points for data that was not classified manually.
    # TODO Should be able to select outliers and what side is positive or not

    # TODO create 2d plots for every dimension and use lasso selection from there
    fig = plt.figure("Complete classifier")
    ax3d = Axes3D(fig)
    ax3d.set_xlabel('Average motor power')
    ax3d.set_ylabel('Velocity')
    ax3d.set_zlabel('Time')

    total_data = {}
    already_used_files = []
    changed_anything = False
    hyperplane = None

    plt.ion()
    if os.path.exists(MODEL_FILE):
        answer = easygui.boolbox("A model already exists do you wish to use it?")

        if answer is None:
            return

        elif answer:
            clf = joblib.load(MODEL_FILE)
            hyperplane = plot_hyperplane(clf, ax3d)
            data = np.load(MODEL_DATA_FILE)
            total_data["features"] = data["features"]
            total_data["labels"] = data["labels"]

            accelerating = total_data["features"][total_data["labels"] == 0]
            decelerating = total_data["features"][total_data["labels"] == 1]

            ax3d.scatter(accelerating[:, 0], accelerating[:, 1], accelerating[:, 2], c="red",
                         label="acceleration")
            ax3d.scatter(decelerating[:, 0], decelerating[:, 1], decelerating[:, 2], c="blue",
                         label="deceleration")

            plt.show()
        else:
            clf = create_blank_classifier()
            changed_anything = True
    else:
        clf = create_blank_classifier()

    while True:
        file = easygui.fileopenbox('Please locate csv file', 'Specify File', default=open_path, filetypes='*.csv')

        if file:
            open_path = "{0:s}\*.csv".format(os.path.dirname(file))

            file_data = get_data(file)

            # TODO make this loop thought the steps as many times as they are number of paths
            if is_valid_log(file_data):
                x, _ = get_features(file_data)
                y = get_labels(file_data)

                x = x[file_data["motionState"] == 'MOVING']
                y = y[file_data["motionState"] == 'MOVING']

                outlier = IsolationForest(n_jobs=-1, random_state=0)

                temp_y = y[y != OUTLIER] = 1
                outlier.fit(x, temp_y)
                prediction = outlier.predict(x)
                # outlier = LocalOutlierFactor(n_jobs=-1, )
                # outlier = EllipticEnvelope(random_state=0)
                # prediction = outlier.fit_predict(x)

                y[prediction == OUTLIER] = OUTLIER

                outliers = x[y == OUTLIER]
                accelerating = x[y == ACCELERATING]
                decelerating = x[y == DECELERATING]
                outlier_power, outlier_velocity, outlier_time = separate_feature(outliers)
                accelerating_power, accelerating_velocity, accelerating_time = separate_feature(accelerating)
                decelerating_power, decelerating_velocity, decelerating_time = separate_feature(decelerating)

                temp_fig = plt.figure(os.path.basename(file).split(".")[0])
                temp_ax = Axes3D(temp_fig)
                temp_ax.set_xlabel('Average motor power')
                temp_ax.set_ylabel('Velocity')
                temp_ax.set_zlabel('Time')

                outlier_line = temp_ax.scatter(outlier_power, outlier_velocity, outlier_time, c="black",
                                               label="outliers")
                acceleration_line = temp_ax.scatter(accelerating_power, accelerating_velocity, accelerating_time,
                                                    c="red",
                                                    label="accelerating")
                deceleration_line = temp_ax.scatter(decelerating_power, decelerating_velocity, decelerating_time,
                                                    c="blue",
                                                    label="decelerating")
                plt.show()

                easygui.msgbox("Next without outliers and rescaled")

                x = x[prediction != OUTLIER]
                y = y[prediction != OUTLIER]
                x = MinMaxScaler().fit_transform(x)

                outlier_line.remove()
                acceleration_line.remove()
                deceleration_line.remove()

                accelerating = x[y == ACCELERATING]
                decelerating = x[y == DECELERATING]
                accelerating_power, accelerating_velocity, accelerating_time = separate_feature(accelerating)
                decelerating_power, decelerating_velocity, decelerating_time = separate_feature(decelerating)

                acceleration_line = temp_ax.scatter(accelerating_power, accelerating_velocity, accelerating_time,
                                                    c="red",
                                                    label="accelerating")
                deceleration_line = temp_ax.scatter(decelerating_power, decelerating_velocity, decelerating_time,
                                                    c="blue",
                                                    label="decelerating")

                # train, test, train_L, test_L = train_test_split(x, y, train_size=.8, test_size=.2, random_state=0,
                #                                                 shuffle=True)
                # clf.fit(train, train_L)

                clf.fit(x, y)
                plot_hyperplane(clf, temp_ax)

                if len(total_data) == 0:
                    total_data = {"features": x, "labels": y}
                    changed_anything = True
                elif file not in already_used_files:
                    new_x = np.concatenate((total_data["features"], x))
                    new_y = np.concatenate((total_data["labels"], y))
                    temp_x = np.hstack((new_x, new_y.reshape((-1, 1))))
                    temp_x = np.unique(temp_x, axis=0)
                    new_x = temp_x[:, :-1]
                    new_y = temp_x[:, -1]

                    total_data["features"] = new_x
                    total_data["labels"] = new_y.ravel()

                    clf.fit(total_data["features"], total_data["labels"])
                    changed_anything = True

                if file not in already_used_files:  # FIXME can this just be in a single if statement?
                    ax3d.scatter(accelerating[:, 0], accelerating[:, 1], accelerating[:, 2], c="red",
                                 label="positive")
                    ax3d.scatter(decelerating[:, 0], decelerating[:, 1], decelerating[:, 2], c="blue",
                                 label="negative")

                    if hyperplane is not None:
                        hyperplane.remove()

                    hyperplane = plot_hyperplane(clf, ax3d)

                already_used_files.append(file)
            else:
                easygui.msgbox(
                    "The file {0:s} is not a valid file.".format(os.path.basename(file)))

        else:
            break

    if changed_anything and not is_empty_model(clf):
        joblib.dump(clf, MODEL_FILE)
        np.savez(MODEL_DATA_FILE, features=total_data["features"], labels=total_data["labels"])
        easygui.msgbox("Model saved.")

    plt.close("all")
    return open_path


def separate_feature(x):
    """

    :param x:
    :return:
    """
    return x[:, 0], x[:, 1], x[:, 2]


def main(open_path):
    """

    :param open_path:
    :return:
    """
    while True:
        answer = easygui.boolbox("Do you wish to train your model or find constants?",
                                 choices=["[T]rain", "[V]iew Constants"])
        if answer is None:
            return open_path
        elif answer:
            open_path = train_model(open_path)
        else:
            open_path = find_constants(open_path)


if __name__ == '__main__':
    main(visualize.open_path)
