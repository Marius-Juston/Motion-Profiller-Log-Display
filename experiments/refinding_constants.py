import math

import matplotlib.pyplot as plt
import numpy as np
import pandas
import seaborn as sns
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D
from sklearn.externals import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import OneClassSVM

from visualize import helper, MODEL_FILE
from visualize.helper import contains_key, find_linear_best_fit_line, get_features


def manipulate_features(features: np.ndarray, file_data: np.ndarray) -> (np.ndarray, np.ndarray):
    """
Return the features manipulated in a way as to make the algorithm for separating the data more accurate.
    :param features: the features to use
    :param file_data: the log file's data
    :return: the manipulated features array, the outliers of the data set and the data scaler
    """

    if contains_key(file_data, "motionState"):
        moving_mask = file_data["motionState"] == "MOVING"
        features = features[moving_mask]
        file_data = file_data[moving_mask]

    new_features = None
    scalers = {}
    if contains_key(file_data, "pathNumber"):

        for i in range(file_data["pathNumber"].min(), file_data["pathNumber"].max() + 1):
            min_max_scaler = MinMaxScaler()

            path_number = file_data["pathNumber"] == i
            scalers[min_max_scaler] = path_number

            features_at_path = features[path_number]

            half = features_at_path.shape[0] // 2
            coefficient, _ = find_linear_best_fit_line(features_at_path[:half, 2], features_at_path[:half, 0])

            if coefficient < 0:
                features_at_path[:, 0] *= - 1

            features_at_path = min_max_scaler.fit_transform(features_at_path)
            outliers_free_features = features_at_path

            if new_features is None:
                new_features = outliers_free_features
            else:
                new_features = np.concatenate((new_features, outliers_free_features), 0)
    else:
        min_max_scaler = MinMaxScaler()
        scalers[min_max_scaler] = np.full(features.shape[0], True)
        new_features = min_max_scaler.fit_transform(features)

    # outlier_detector = OneClassSVM(gamma=10)  # Seems to work best

    # outlier_detector.fit(new_features)
    # outlier_prediction = outlier_detector.predict(new_features)
    # outliers = new_features[outlier_prediction == -1]
    # new_features = new_features[outlier_prediction == 1]

    # features = reverse_scalling(new_features, scalers, outlier_prediction)

    return new_features, features


def reverse_scalling(features, scalers, outlier_prediction):
    features = np.copy(features)

    for scaler, index in zip(scalers.keys(), scalers.values()):
        index = index[outlier_prediction == 1]

        features[index] = scaler.inverse_transform(features[index])

    return features


data = helper.get_data(r"..\example_data\2018-03-21 08-29-04.csv")

features, col = get_features(data)
filters = data['motionState'] == 'MOVING'
data = data[filters]
features = features[filters]
# f = helper.get_features(data)

XTE = data['XTE']
lagE = data['lagE']
angleE = data['angleE']

targetX = data["xTarget"]
targetY = data["yTarget"]
targetAngle = data["angleTarget"]

actualX = data["xActual"]
actualY = data["yActual"]
actualAngle = data["angleActual"]

dx = targetX - actualX
dy = targetY - actualY

lagError = dx * np.cos(angleE) + dy * np.sin(angleE)

crossTrackError = -dx * np.sin(targetAngle) + dy * np.cos(targetAngle)

angleError = targetAngle - actualAngle

angleError[angleError > math.pi] -= 2 * math.pi
angleError[angleError < -math.pi] += 2 * math.pi

# print(lagError)
# print(lagE)

avg = lagE[np.nonzero(lagError)] / lagError[np.nonzero(lagError)]
# print(avg)

# print(crossTrackError)
# print(XTE)

# print(angleError)
# print(angleE)


fig = plt.gcf()
gs = GridSpec(3, 1, fig)

velocity = fig.add_subplot(gs[0])
errors = fig.add_subplot(gs[1])
powers = fig.add_subplot(gs[2])

time = data["Time"]

velocity.plot(time, lagE)
velocity.plot(time, lagError)
# velocity.plot(time, lagError * np.mean(avg))

errors.plot(time, XTE)
errors.plot(time, crossTrackError)

powers.plot(time, angleE)
powers.plot(time, angleError)

fig = Axes3D(plt.figure("Hello"))
fig.plot(lagError, crossTrackError, angleError)

print(col)


clf = joblib.load(MODEL_FILE)

new_scaled_features,  features = manipulate_features(features, data)
# features = scaler.inverse_transform(new_scaled_features)

labels = clf.predict(new_scaled_features)

color_labels = list(map(lambda x: 'r' if x == 0 else 'b', labels))

print(features.shape)
print(new_scaled_features.shape)
print(len(color_labels))
print(XTE.shape)

d = {
    "XTE": XTE, "lag": lagE, "angle": angleE,
     col[0]: new_scaled_features[:, 0], col[1]: new_scaled_features[:, 1], col[2]: new_scaled_features[:, 2], "labels":color_labels}
print(d)

# p = pandas.DataFrame()
p = pandas.DataFrame(d)

sns.set(style="ticks", color_codes=True)
g = sns.pairplot(p, hue="labels")

# g = sns.PairGrid(data,
#                  hue='labels', palette='RdBu_r')
# g.map(plt.scatter, alpha=0.8)
# g.add_legend()

plt.show()
