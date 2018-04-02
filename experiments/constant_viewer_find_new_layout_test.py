import sklearn
from itertools import combinations

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D
from sklearn import ensemble
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from visualize.helper import plot_hyperplane

fig = plt.figure()
gs = GridSpec(3, 4, fig)

from sklearn.datasets import make_classification

features, labels = make_classification(n_features=3, n_informative=2, n_redundant=1, n_clusters_per_class=1)


def plot(features, lables, subplots):
    velocty, time, power = features[:, 0], features[:, 1], features[:, 2]

    for combination, subplot in zip(combinations((velocty, time, power), 2), subplots):
        subplot.scatter(combination[0], combination[1], c=labels)

# Axes3D
master_plot = fig.add_subplot(gs[:3, :3], projection='3d')
time_velocity = fig.add_subplot(gs[0, -1])
time_power = fig.add_subplot(gs[1, -1])
power_velocity = fig.add_subplot(gs[2, -1])

master_plot.scatter(features[:, 0], features[:, 1], features[:, 2], c=labels)

plot(features, labels, (time_velocity, time_power, power_velocity))

clf= SVC(kernel="rbf", C=10)
# clf = sklearn.ensemble.BaggingClassifier()
clf.fit(features, labels)

plot_hyperplane(clf, master_plot, interval=.05)

gs.tight_layout(fig)

figManager = plt.get_current_fig_manager()
figManager.window.showMaximized()

plt.show()
