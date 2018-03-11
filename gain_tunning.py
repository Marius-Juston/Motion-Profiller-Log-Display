import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


def main():
    file_data = np.genfromtxt("./test_data/2018-03-04 03-22-39.csv", delimiter=',', dtype=np.float32, names=True)

    average = (file_data["pLeft"] + file_data["pRight"]) / 2

    previous_data = np.roll(file_data, 1)
    velocity = np.sqrt(
        (file_data["xActual"] - previous_data["xActual"]) ** 2 + (
                file_data["yActual"] - previous_data["yActual"]) ** 2) / (file_data["Time"] - previous_data["Time"])

    out = IsolationForest(n_jobs=-1, random_state=0)

    X = np.concatenate((np.vstack(average), np.vstack(velocity), np.vstack(file_data["Time"])), 1)
    out.fit(X)

    predicted = out.predict(X)

    Y = file_data["classification"][predicted == 1]
    X = X[predicted == 1]
    average = np.hstack(X[:, 0])
    velocity = np.hstack(X[:, 1])

    clf = SVC(kernel="linear", random_state=0)

    train, test, train_L, test_L, = train_test_split(X, Y, train_size=.8, test_size=.2, random_state=0)

    clf.fit(train, y=train_L)
    predicted = clf.predict(X)
    print(clf.score(test, test_L))

    going_up = X[predicted == 0]
    going_down = X[predicted == 1]

    coef1 = find_coefficient(going_up[:, 0], going_up[:, 1])
    coef2 = find_coefficient(going_down[:, 0], going_down[:, 1])

    print((coef1 + coef2) / 2)

    plt.scatter(average, velocity, c=predicted)
    plt.show()

    print()


def find_coefficient(average, velocity):
    m, b = np.polyfit(average, velocity, 1)

    return m


if __name__ == '__main__':
    main()
