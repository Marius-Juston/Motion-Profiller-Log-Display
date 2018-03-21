import matplotlib.pyplot as plt

fig = plt.figure("Path")
path = fig.gca()

fig = plt.figure("Numbers")
errors, powers, velocity = fig.subplots(3)
plt.show()
