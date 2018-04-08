import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D
from sklearn.ensemble import IsolationForest

from visualize.helper import plot_hyperplane, plot_subplots

if __name__ == '__main__':
    fig = plt.figure()
    gs = GridSpec(3, 8, fig)

    from sklearn.datasets import make_classification

    # 0 == velocity
    # 1 == time
    # 2 == power

    features, labels = make_classification(1000, random_state=8, n_features=3, n_informative=2, n_redundant=1,
                                           n_clusters_per_class=1)

    # Axes3D
    master_plot = fig.add_subplot(gs[:3, :3], projection='3d')
    master_plot.set_aspect("equal", "box")
    master_plot.set_xlabel("Velocity")
    master_plot.set_ylabel("Time")
    master_plot.set_zlabel("Power")
    time_velocity = fig.add_subplot(gs[0, 3:5])
    time_power = fig.add_subplot(gs[1, 3:5])
    power_velocity = fig.add_subplot(gs[2, 3:5])
    constant_plot = fig.add_subplot(gs[:, 5:])
    constant_plot.set_aspect("equal", "box")

    import numpy as np

    color_label = np.array(list(map(lambda x: 'r' if x == 1 else 'b', labels)))

    plot_subplots(features, {0: 'Average Power', 1: 'Velocity', 2: 'Time'}, (time_velocity, power_velocity, time_power),
                  labels=color_label)

    # clf = SVC(kernel="rbf", random_state=0)
    # clf = sklearn.neighbors.KNeighborsClassifier()

    clf = IsolationForest(n_jobs=-1, random_state=0, contamination=.20)
    # clf = OneClassSVM()
    clf.fit(features, labels)
    outlier_prediction = clf.predict(features)

    outliers = features[outlier_prediction == -1]
    outlier_free = features[outlier_prediction == 1]
    color_label = color_label[outlier_prediction == 1]

    master_plot.scatter(outlier_free[:, 0], outlier_free[:, 1], outlier_free[:, 2], c=color_label)
    master_plot.scatter(outliers[:, 0], outliers[:, 1], outliers[:, 2], c="black")

    plot_hyperplane(clf, master_plot, interval=.05)

    gs.tight_layout(fig)

    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()

    # plt.show()
    fig.show()
    plt.show()

    # fig.clear()
    Axes3D(fig)
    fig.show()
    plt.show(fig)
