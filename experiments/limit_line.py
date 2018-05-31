import matplotlib.pyplot as plt
import numpy as np

from visualize.helper import get_xy_limited

coef = 1 / 4
intercept = .8

ax = plt.gca()

x_min = 0
x_max = 2
y_min = 0
y_max = 1
y_lim = np.array([y_min, y_max])
x_lim = np.array([x_min, x_max])

# x = (y_lim - intercept) / coef
#
#
# x[x > x_max] = x_max
# x[x < x_min] = x_min
#
# y = x * coef + intercept


x, y = get_xy_limited(intercept, coef, x_lim, y_lim)

ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
ax.plot(x, y)

# ax.plot(x_lim, x_lim * coef + intercept)
plt.show()
