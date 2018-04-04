import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D
from sklearn.svm import SVC

from visualize.helper import plot_hyperplane, plot_subplots

if __name__ == '__main__':
    fig = plt.figure()
    gs = GridSpec(3, 4, fig)

    from sklearn.datasets import make_classification

    # 0 == velocity
    # 1 == time
    # 2 == power

    features, labels = make_classification(n_features=3, n_informative=2, n_redundant=1, n_clusters_per_class=1)

    # Axes3D
    master_plot = fig.add_subplot(gs[:3, :3], projection='3d')
    master_plot.set_xlabel("Velocity")
    master_plot.set_ylabel("Time")
    master_plot.set_zlabel("Power")
    time_velocity = fig.add_subplot(gs[0, -1])
    time_power = fig.add_subplot(gs[1, -1])
    power_velocity = fig.add_subplot(gs[2, -1])

    color_label = list(map(lambda x: 'r' if x == 1 else 'b', labels))

    master_plot.scatter(features[:, 0], features[:, 1], features[:, 2], c=color_label)

    plot_subplots(features, {0: 'Average Power', 1: 'Velocity', 2: 'Time'}, (time_velocity, power_velocity, time_power),
                  labels=color_label)

    clf = SVC(kernel="linear", gamma=10)
    # clf = sklearn.ensemble.BaggingClassifier()
    clf.fit(features, labels)

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
