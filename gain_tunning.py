import os

import easygui
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from skimage import measure
from sklearn.ensemble import IsolationForest
from sklearn.externals import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

from driver import is_valid_log

MODEL_FILE_NAME = "model.pkl"
MODEL_DATA_FILE_NAME = "data.npz"
open_path = "{0:s}\*.csv".format(os.path.expanduser("~"))


def get_features(file_data):
    average_power = (file_data["pLeft"] + file_data["pRight"]) / 2

    previous_data = np.roll(file_data, 1)
    velocity = np.sqrt(
        (file_data["xActual"] - previous_data["xActual"]) ** 2 + (
                file_data["yActual"] - previous_data["yActual"]) ** 2) / (file_data["Time"] - previous_data["Time"])

    X = np.concatenate((np.vstack(average_power), np.vstack(velocity), np.vstack(file_data["Time"])), 1)

    return X


def get_labels(file_data):
    return file_data["classification"]


def has_classification(file_data):
    return "classification" in file_data.dtype.fields


def plot_hyperplane(clf, ax):
    # get the separating hyperplane
    interval = .1
    interval = int(1 / interval)

    X_MIN, X_MAX = ax.get_xlim()
    Y_MIN, Y_MAX = ax.get_ylim()
    Z_MIN, Z_MAX = ax.get_zlim()

    # create grid to evaluate model
    XX = np.linspace(X_MIN, X_MAX, interval)
    YY = np.linspace(Y_MIN, Y_MAX, interval)
    ZZ = np.linspace(Z_MIN, Z_MAX, interval)
    YY, XX, ZZ = np.meshgrid(YY, XX, ZZ)

    # xy = np.vstack([XX.ravel(), YY.ravel(), Z.ravel()]).T

    Z = clf.decision_function(np.c_[XX.ravel(), YY.ravel(), ZZ.ravel()])
    # Z = MinMaxScaler().fit_transform(Z.reshape(-1, 1))

    # w_norm = np.linalg.norm(clf.coef_)
    # Z = Z / w_norm
    Z = Z.reshape(XX.shape)

    verts, faces, _, _ = measure.marching_cubes(Z, 0)
    # Scale and transform to actual size of the interesting volume
    verts = verts * [X_MAX - X_MIN, Y_MAX - Y_MIN, Z_MAX - Z_MIN] / interval
    verts = verts + [X_MIN, Y_MIN, Z_MIN]
    # and create a mesh to display
    # mesh = Poly3DCollection(verts[faces],
    #                         facecolor='orange', alpha=0.3)
    mesh = Line3DCollection(verts[faces],
                            facecolor='orange', alpha=0.3)
    ax.add_collection3d(mesh)

    return mesh


def find_constants():
    global open_path

    fig = plt.figure()
    ax = Axes3D(fig)
    clf = joblib.load(MODEL_FILE_NAME)
    plt.ion()
    plot_hyperplane(clf, ax)

    while True:
        file = easygui.fileopenbox('Please locate csv file', 'Specify File', default=open_path, filetypes='*.csv')

        if file:
            open_path = "{0:s}\*.csv".format(os.path.dirname(file))

            file_data = np.genfromtxt(file, delimiter=',', dtype=np.float32, names=True)

            if is_valid_log(file_data):
                coef, intercept = find_gain(clf,file_data, is_data=True, plot=ax)
                plt.show()

                easygui.msgbox("The kV of this log is {0:f}.\nThe kC of this log is {1:f}".format(coef, intercept))
            else:
                easygui.msgbox(
                    "The file {0:s} is not a valid file.".format(os.path.basename(file)))

        else:
            break
    plt.ioff()
    # plt.show()


def find_gain(clf, file_data, is_data=False, plot=None):
    if not is_data:
        file_data = np.genfromtxt(file_data, delimiter=',', dtype=np.float32, names=True)

    X = get_features(file_data)

    out = IsolationForest(n_jobs=-1, random_state=0)
    out.fit(X)
    predicted = out.predict(X)
    X = X[predicted == 1]

    X_scaled = MinMaxScaler().fit_transform(X)

    predicted = clf.predict(X_scaled)

    going_up = X[predicted == 0]
    going_down = X[predicted == 1]

    coef1, intercept1 = find_best_fit_line(going_up[:, 0], going_up[:, 1])
    coef2, intercept2 = find_best_fit_line(going_down[:, 0], going_down[:, 1])

    if plot:
        average = np.hstack(X_scaled[:, 0])
        velocity = np.hstack(X_scaled[:, 1])
        plot.set_xlabel('Scaled average motor power')
        plot.set_ylabel('Scaled velocity')

        if isinstance(plot, Axes3D):
            plot.set_zlabel('Scaled Time')
            time = np.hstack(X_scaled[:, 2])
            plot.scatter(average, velocity, time, c=predicted)
        else:
            plot.scatter(average, velocity, c=predicted)

    return (coef1 + coef2) / 2, (intercept1 + intercept2) / 2


def find_best_fit_line(x, y):
    m, b = np.polyfit(x, y, 1)

    return m, b


def train_model():
    # TODO add lasso selection of points for data that was not classified manually.
    # TODO Should be able to select outliers and what side is positive or not
    global open_path
    # clf = SVC(random_state=0, kernel="linear")

    fig = plt.figure("Complete classifier")
    ax = Axes3D(fig)
    plt.ion()
    ax.set_xlabel('Average motor power')
    ax.set_ylabel('Velocity')
    ax.set_zlabel('Time')

    total_data = {}
    already_used_files = []
    changed_anything = False
    hyperplane = None

    if os.path.exists(MODEL_FILE_NAME):
        answer = easygui.boolbox("A model already exists do you wish to use it?")

        if answer:
            clf = joblib.load(MODEL_FILE_NAME)
            hyperplane = plot_hyperplane(clf, ax)
            data = np.load(MODEL_DATA_FILE_NAME)
            total_data["features"] = data["features"]
            total_data["labels"] = data["labels"]
            positive = total_data["features"][total_data["labels"] == 0]
            negative = total_data["features"][total_data["labels"] == 1]

            positive_line = ax.scatter(positive[:, 0], positive[:, 1], positive[:, 2], c="red",
                                       label="positive")
            negative_line = ax.scatter(negative[:, 0], negative[:, 1], negative[:, 2], c="blue",
                                       label="negative")

            total_data["labels"] = total_data["labels"].reshape((-1, 1))
            plt.show()
        else:
            clf = SVC(random_state=0, kernel="rbf")
            changed_anything = True
    else:
        clf = SVC(random_state=0, kernel="rbf")

    while True:
        file = easygui.fileopenbox('Please locate csv file', 'Specify File', default=open_path, filetypes='*.csv')

        if file:
            open_path = "{0:s}\*.csv".format(os.path.dirname(file))

            file_data = np.genfromtxt(file, delimiter=',', dtype=np.float32, names=True)

            if is_valid_log(file_data):
                X = get_features(file_data)
                Y = get_labels(file_data)

                X = MinMaxScaler().fit_transform(X)

                outlier = IsolationForest(n_jobs=-1, random_state=0)
                # outlier = LocalOutlierFactor(n_jobs=-1, )
                # outlier = EllipticEnvelope(random_state=0)
                outlier.fit(X, Y)
                prediction = outlier.predict(X)
                # prediction = outlier.fit_predict(X)

                Y[prediction == -1] = -1

                outliers = X[Y == -1]
                positive = X[Y == 0]
                negative = X[Y == 1]

                temp_fig = plt.figure(os.path.basename(file).split(".")[0])
                temp_ax = Axes3D(temp_fig)
                temp_ax.set_xlabel('Average motor power')
                temp_ax.set_ylabel('Velocity')
                temp_ax.set_zlabel('Time')

                outlier_line = temp_ax.scatter(outliers[:, 0], outliers[:, 1], outliers[:, 2], c="black",
                                               label="outliers")
                positive_line = temp_ax.scatter(positive[:, 0], positive[:, 1], positive[:, 2], c="red",
                                                label="positive")
                negative_line = temp_ax.scatter(negative[:, 0], negative[:, 1], negative[:, 2], c="blue",
                                                label="negative")
                plt.show()

                easygui.msgbox("Next without outliers and rescaled")

                X = X[prediction != -1]
                Y = Y[prediction != -1]
                X = MinMaxScaler().fit_transform(X)

                outlier_line.remove()
                positive_line.remove()
                negative_line.remove()

                positive = X[Y == 0]
                negative = X[Y == 1]
                positive_line = temp_ax.scatter(positive[:, 0], positive[:, 1], positive[:, 2], c="red",
                                                label="positive")
                negative_line = temp_ax.scatter(negative[:, 0], negative[:, 1], negative[:, 2], c="blue",
                                                label="negative")

                # train, test, train_L, test_L = train_test_split(X, Y, train_size=.8, test_size=.2, random_state=0,
                #                                                 shuffle=True)
                # clf.fit(train, train_L)

                clf.fit(X, Y)
                plot_hyperplane(clf, temp_ax)

                if len(total_data) == 0:
                    total_data = {"features": X, "labels": Y.reshape((-1, 1))}
                    changed_anything = True
                elif file not in already_used_files:
                    new_x = np.concatenate((total_data["features"], X))
                    new_y = np.concatenate((total_data["labels"], Y.reshape(-1, 1)))
                    temp_x = np.hstack((new_x, new_y))
                    temp_x = np.unique(temp_x, axis=0)
                    new_x = temp_x[:, :-1]
                    new_y = temp_x[:, -1]

                    total_data["features"] = new_x
                    total_data["labels"] = new_y

                    clf.fit(total_data["features"], total_data["labels"])
                    changed_anything = True
                if file not in already_used_files:  # FIXME can this just be in a single if statement?
                    ax.scatter(positive[:, 0], positive[:, 1], positive[:, 2], c="red",
                               label="positive")
                    ax.scatter(negative[:, 0], negative[:, 1], negative[:, 2], c="blue",
                               label="negative")

                    if hyperplane is not None:
                        hyperplane.remove()

                    hyperplane = plot_hyperplane(clf, ax)

                already_used_files.append(file)
            else:
                easygui.msgbox(
                    "The file {0:s} is not a valid file.".format(os.path.basename(file)))

        else:
            break

    if changed_anything:
        print("saving model")
        joblib.dump(clf, MODEL_FILE_NAME)
        np.savez(MODEL_DATA_FILE_NAME, features=total_data["features"], labels=total_data["labels"])


def main():
    while True:
        answer = easygui.boolbox("You you wish to train or find constants?", choices=["[T]rain", "[V]iew Constants"])
        if answer is None:
            break
        elif answer:
            train_model()
        else:
            find_constants()


if __name__ == '__main__':
    main()
