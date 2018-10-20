import matplotlib.pyplot as plt
import numpy as np

from visualize.helper import is_straight_line

x = np.array([-4, -2, 0, 4, 2])
y = np.array([1, 2, 3, 4, 5])

data = dict(xTarget=x, yTarget=y)

plt.scatter(x, y, c=['r', 'b', 'g', 'indigo', 'magenta'])
print(is_straight_line(data))
plt.show()
