import matplotlib.pyplot as plt
import numpy as np

coef = 2
intercept = -1

ax = plt.gca()

x_min = 0
x_max = 2
y_min = 0
y_max = 1
y_lim = np.array([y_min, y_max])
x_lim = np.array([x_min, x_max])

# (y_lim - i) / c

x = (y_lim - intercept) / coef
x[x > x_max] = x_max
x[x < x_min] = x_min
y = y_lim

# if coef > 1:
#     ax.plot((y_lim - intercept) / coef, y_lim)
# else:
ax.set_xlim(x_min, x_max)
ax.plot(x, y)

# ax.plot(x_lim, x_lim * coef + intercept)
plt.show()
