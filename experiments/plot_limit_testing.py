import matplotlib.pyplot as plt

ax = plt.gca()
scat = ax.scatter([0, 1], [0, 1])

print(ax.get_xlim())
print(ax.get_ylim())
# plt.show()

scat.remove()
ax.clear()
scat = ax.scatter([.25, .75], [.25, .75])

ax.autoscale_view(True, True, True)
ax.relim()

print(ax.get_xlim())
print(ax.get_ylim())
plt.show()
