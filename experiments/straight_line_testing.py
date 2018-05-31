import numpy as np

from visualize.helper import is_straight_line

x = np.array([2, 4, 0, -4, 6])
y = np.array([3, 4, 2, 0, 5])

data = dict(xTarget=x, yTarget=y)

print(is_straight_line(data))
