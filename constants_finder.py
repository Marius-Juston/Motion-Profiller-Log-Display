import os

import easygui
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from skimage import measure
from sklearn.ensemble import IsolationForest
from sklearn.exceptions import NotFittedError
from sklearn.externals import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

from log_viewer import is_valid_log

MODEL_FILE_NAME = "model.pkl"
MODEL_DATA_FILE_NAME = "data.npz"
open_path = "{0:s}\*.csv".format(os.path.expanduser("~"))
ACCELERATING = 0
DECELERATING = 1
OUTLIER = -1


def get_features(file_data):
    average_power = (file_data["pLeft"] + file_data["pRight"]) / 2.0

    time = file_data["Time"]
    previous_data = np.roll(file_data, 1)
    velocity = np.sqrt(
        (file_data["xActual"] - previous_data["xActual"]) ** 2 + (
                file_data["yActual"] - previous_data["yActual"]) ** 2) / (time - previous_data["Time"])

    velocity[0] = 0

    x = np.concatenate((np.vstack(average_power), np.vstack(velocity), np.vstack(time)), 1)
    return x


def get_labels(file_data):
    return file_data["classification"]


def has_classification(file_data):
    return "classification" in file_data.dtype.fields


def plot_hyperplane(clf, ax):
    # get the separating hyperplane
    interval = .1
    interval = int(1 / interval)

    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    z_min, z_max = ax.get_zlim()

    # create grid to evaluate model
    xx = np.linspace(x_min, x_max, interval)
    yy = np.linspace(y_min, y_max, interval)
    zz = np.linspace(z_min, z_max, interval)
    yy, xx, zz = np.meshgrid(yy, xx, zz)

    z = clf.decision_function(np.c_[xx.ravel(), yy.ravel(), zz.ravel()])
    z = z.reshape(xx.shape)

    verteces, faces, _, _ = measure.marching_cubes(z, 0)
    # Scale and transform to actual size of the interesting volume
    verteces = verteces * [x_max - x_min, y_max - y_min, z_max - z_min] / interval
    verteces = verteces + [x_min, y_min, z_min]
    # and create a mesh to display
    # mesh = Poly3DCollection(verteces[faces],
    #                         facecolor='orange', alpha=0.3)
    mesh = Line3DCollection(verteces[faces],
                            facecolor='orange', alpha=0.3)
    ax.add_collection3d(mesh)

    return mesh


def find_constants():
    global open_path

    if not os.path.exists(MODEL_FILE_NAME):
        easygui.msgbox("There are no models to use to classify the data. Please train algorithm first.")
        return

    clf = joblib.load(MODEL_FILE_NAME)

    if is_empty_model(clf):
        easygui.msgbox("The model has not been fitted yet. Please add training data to the model.")
        return

    fig = plt.figure("Scaled 3d data")
    ax3d = Axes3D(fig)
    fig, ax2d = plt.subplots(1, 1, num="Fitted data")

    plt.ion()

    while True:
        file = easygui.fileopenbox('Please locate csv file', 'Specify File', default=open_path, filetypes='*.csv')

        if file:
            open_path = "{0:s}\*.csv".format(os.path.dirname(file))

            file_data = np.genfromtxt(file, delimiter=',', dtype=np.float32, names=True)

            if is_valid_log(file_data):
                ax2d.cla()
                ax3d.cla()

                plot_hyperplane(clf, ax3d)

                k_v, k_k, k_acc = find_gain(clf, file_data, is_data=True, ax3d=ax3d, ax2d=ax2d)
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


def find_gain(clf, file_data, is_data=False, ax3d=None, ax2d=None):
    if not is_data:
        file_data = np.genfromtxt(file_data, delimiter=',', dtype=np.float32, names=True)

    x = get_features(file_data)

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


def find_linear_best_fit_line(x, y):
    m, b = np.polyfit(x, y, 1)

    return m, b


def create_blank_classifier():
    return SVC(kernel="rbf", random_state=0)


def train_model():
    # TODO add lasso selection of points for data that was not classified manually.
    # TODO Should be able to select outliers and what side is positive or not

    # TODO create 2d plots for every dimension and use lasso selection from there
    global open_path
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
    if os.path.exists(MODEL_FILE_NAME):
        answer = easygui.boolbox("A model already exists do you wish to use it?")

        if answer is None:
            return

        elif answer:
            clf = joblib.load(MODEL_FILE_NAME)
            hyperplane = plot_hyperplane(clf, ax3d)
            data = np.load(MODEL_DATA_FILE_NAME)
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

            file_data = np.genfromtxt(file, delimiter=',', dtype=np.float32, names=True)

            if is_valid_log(file_data):
                x = get_features(file_data)
                y = get_labels(file_data)

                outlier = IsolationForest(n_jobs=-1, random_state=0)
                outlier.fit(x, y)
                prediction = outlier.predict(x)
                # outlier = LocalOutlierFactor(n_jobs=-1, )
                # outlier = EllipticEnvelope(random_state=0)
                # prediction = outlier.fit_predict(x)

                y[prediction == -1] = OUTLIER

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
        joblib.dump(clf, MODEL_FILE_NAME)
        np.savez(MODEL_DATA_FILE_NAME, features=total_data["features"], labels=total_data["labels"])
        easygui.msgbox("Model saved.")

    plt.close("all")


def separate_feature(x):
    return x[:, 0], x[:, 1], x[:, 2]


def is_empty_model(clf):
    try:
        clf.predict([[0, 0, 0]])
        return False
    except NotFittedError:
        return True


def main():
    while True:
        answer = easygui.boolbox("Do you wish to train your model or find constants?",
                                 choices=["[T]rain", "[V]iew Constants"])
        if answer is None:
            break
        elif answer:
            train_model()
        else:
            find_constants()


if __name__ == '__main__':
    main()
