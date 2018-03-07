import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

x = [0, 1, 2]
y = [0, 1, 2]
yaw = [0.0, 0.5, 1.3]
fig, ax = plt.subplots(1, 1)
plt.axis('equal')
plt.grid()
ax.set_xlim(-10, 10)
ax.set_ylim(-10, 10)

anim = None


# file_data = np.genfromtxt("Hello.csv", delimiter=',', dtype=np.float32, names=True)
# delta_times = file_data["Time"]
# delta_times = np.roll(delta_times, -1, 0) - delta_times
# delta_times[-1] = 0
#
# patch = patches.Rectangle((file_data["xActual"][0], file_data["yActual"][0]), 0, 0, angle=file_data["angleActual"][0],
#                           fc='y')


def set_interval(time):
    """
    :param time: Time in seconds
    :return:
    """
    anim.event_source.interval = time * 1000


def init():
    ax.add_patch(patch)
    patch.set_width(.78)
    patch.set_height(.80)
    return patch,


def animate(i):
    i += 1
    if i == 2:
        patch.set_visible(False)
        del patch
        return ()
    return patch,


# anim = animation.FuncAnimation(fig, animate,
#                                init_func=init,
#                                frames=len(delta_times) - 1,
#                                interval=delta_times[0],
#                                blit=True, repeat=False)

def get_delta(patch):
    angle = np.math.atan2((patch.get_height() / 2), (patch.get_width() / 2)) + np.deg2rad(patch.angle)

    print(np.rad2deg(angle))

    d = np.sqrt((patch.get_width() / 2) ** 2 + (patch.get_height() / 2) ** 2)
    sin = np.cos(angle)
    x = ((d * sin) if sin != 0 else 0)
    cos = np.sin(angle)
    y = ((d * cos) if cos != 0 else 0)

    return x, y


patch = patches.Rectangle((2, 2), .8, .8, 45)
# ax.add_patch(patch)
p = get_delta(patch)
print(p)
# plt.show()
