import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import OneClassSVM

from visualize.helper import contains_key, find_linear_best_fit_line


def manipulate_features(features: np.ndarray, file_data: np.ndarray, view_individual_paths=False) -> (
        np.ndarray, np.ndarray):
    """
Return the features manipulated in a way as to make the algorithm for separating the data more accurate.
    :param features: the features to use
    :param file_data: the log file's data
    :return: the manipulated features array, the outliers of the data set and the data scaler
    :param view_individual_paths:
    """

    if contains_key(file_data, "motionState"):
        moving_mask = file_data["motionState"] == "MOVING"
        features = features[moving_mask]
        file_data = file_data[moving_mask]

    new_features = None
    scalers = {}
    if contains_key(file_data, "pathNumber"):

        for pathNumber in np.unique(file_data["pathNumber"]):
            min_max_scaler = MinMaxScaler()

            path_number = file_data["pathNumber"] == pathNumber
            scalers[min_max_scaler] = path_number

            features_at_path = features[path_number]

            half = features_at_path.shape[0] // 2
            coefficient, _ = find_linear_best_fit_line(features_at_path[:half, 2], features_at_path[:half, 0])

            if coefficient < 0:
                features_at_path[:, 0] *= - 1

            features_at_path = min_max_scaler.fit_transform(features_at_path)
            outliers_free_features = features_at_path

            if view_individual_paths:
                fig = plt.figure()

                fig.gca().scatter(features_at_path[:, 0], features_at_path[:, 1])

            if new_features is None:
                new_features = outliers_free_features
            else:
                new_features = np.concatenate((new_features, outliers_free_features), 0)
    else:
        min_max_scaler = MinMaxScaler()
        scalers[min_max_scaler] = np.full(features.shape[0], True)
        new_features = min_max_scaler.fit_transform(features)

    features = reverse_scaling(new_features, scalers)
    return new_features, features


def find_and_remove_outliers(new_features: np.ndarray):
    outlier_detector = OneClassSVM(gamma=10)  # Seems to work best

    outlier_detector.fit(new_features)
    outlier_prediction = outlier_detector.predict(new_features)
    outliers = new_features[outlier_prediction == -1]
    new_features = new_features[outlier_prediction == 1]

    return new_features, outliers, outlier_detector


def reverse_scaling(features: np.ndarray, scalers: dict, outlier_prediction: bool = None):
    features = np.copy(features)

    for scaler, index in zip(scalers.keys(), scalers.values()):
        if outlier_prediction is not None:
            index = index[outlier_prediction == 1]

        features[index] = scaler.inverse_transform(features[index])

    return features
