import matplotlib.pyplot as plt
from sklearn.svm import OneClassSVM

from visualize import helper
from visualize.helper import get_features, plot_hyperplane

data = helper.get_data(r"..\example_data\2018-03-21 08-29-04.csv")
features, col = get_features(data)

ax3d = plt.gca(projection='3d')
ax3d.scatter(features[:, 0], features[:, 1], features[:, 2])

clf = OneClassSVM(degree=10)

clf.fit(features)

plot_hyperplane(clf, ax3d, interval=.01)

plt.show()
