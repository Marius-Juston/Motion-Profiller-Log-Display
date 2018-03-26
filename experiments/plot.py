import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

from visualize.helper import get_data, get_velocity, get_acceleration

if __name__ == '__main__':
    data = get_data(r"C:\Users\mariu\Documents\Logs\2018-03-21 08-29-04.csv")

    ax.

    time = data["Time"]

    velocity = get_velocity(data)
    acceleration = get_acceleration(time, velocity)

    i = IsolationForest(n_jobs=-1)

    i.fit(acceleration.reshape(-1, 1))
    p = i.predict(acceleration.reshape(-1, 1))
    # p = p.ravel()

    acceleration = acceleration[p == 1]
    velocity = velocity[p == 1]

    plt.subplot(211).plot(time, velocity)
    plt.subplot(212).scatter(time, acceleration)

    plt.show()
