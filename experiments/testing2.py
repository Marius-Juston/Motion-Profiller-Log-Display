import time

import matplotlib.pyplot as plt

max_t = 0
current_time = 0


def on_press(event):
    if event.inaxes is not None:
        global current_time
        current_time = time.time()
    else:
        print('Clicked ouside axes bounds but inside plot window')


def on_release(event):
    if event.inaxes is not None:
        global max_t
        max_t = max(time.time() - current_time, max_t)
        print(max_t)
    else:
        print('Clicked ouside axes bounds but inside plot window')


fig, ax = plt.subplots()
fig.canvas.callbacks.connect('button_click_event', on_release)
# fig.canvas.callbacks.connect('button_release_event', on_release)
plt.show()
