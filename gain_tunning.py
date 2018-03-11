import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from sklearn.ensemble import IsolationForest
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC


def get_features(file_data):
    averagePower = (file_data["pLeft"] + file_data["pRight"]) / 2

    previous_data = np.roll(file_data, 1)
    velocity = np.sqrt(
        (file_data["xActual"] - previous_data["xActual"]) ** 2 + (
                file_data["yActual"] - previous_data["yActual"]) ** 2) / (file_data["Time"] - previous_data["Time"])

    X = np.concatenate((np.vstack(averagePower), np.vstack(velocity), np.vstack(file_data["Time"])), 1)

    return X


def get_labels(file_data):
    return file_data["classification"]


def has_classification(file_data):
    return "classification" in file_data.dtype.fields


def plot_hyperplane(clf, min_x, max_x, linestyle, label):
    # get the separating hyperplane
    w = clf.coef_[0]
    a = -w[0] / w[1]
    xx = np.linspace(min_x - .1, max_x + .1)  # make sure the line is long enough
    yy = a * xx - (clf.intercept_[0]) / w[1]
    plt.plot(xx, yy, linestyle, label=label)


MODEL_FILE_NAME = "model.pkl"


def main():
    fig = plt.figure()
    ax = Axes3D(fig)

    find_gain("./test_data/2018-03-04 03-22-39.csv", plot=ax)
    plt.show()
    find_gain("./test_data/2018-02-13 07-57-11.csv", plot=ax)
    plt.show()


def find_gain(file_name, learn=False, plot=None):
    file_data = np.genfromtxt(file_name, delimiter=',', dtype=np.float32, names=True)

    X = get_features(file_data)

    out = IsolationForest(n_jobs=-1, random_state=0)
    out.fit(X)
    predicted = out.predict(X)
    X = X[predicted == 1]

    X_scaled = MinMaxScaler().fit_transform(X)

    if learn and has_classification(file_data):
        Y = get_labels(file_data)[predicted == 1]

        train, test, train_L, test_L, = train_test_split(X_scaled, Y, train_size=.8, test_size=.2, random_state=0)
        clf = SVC(kernel="linear", random_state=0)
        clf.fit(train, y=train_L)
        print(clf.score(test, test_L))

        joblib.dump(clf, MODEL_FILE_NAME)
    else:
        clf = joblib.load(MODEL_FILE_NAME)

    predicted = clf.predict(X_scaled)

    going_up = X[predicted == 0]
    going_down = X[predicted == 1]

    coef1 = find_coefficient(going_up[:, 0], going_up[:, 1])
    coef2 = find_coefficient(going_down[:, 0], going_down[:, 1])

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
    return (coef1 + coef2) / 2


def find_coefficient(average, velocity):
    m, b = np.polyfit(average, velocity, 1)

    return m


if __name__ == '__main__':
    main()
