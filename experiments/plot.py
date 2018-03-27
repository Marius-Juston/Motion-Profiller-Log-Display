import matplotlib.pyplot as plt

from visualize.helper import get_data, get_velocity, get_acceleration

if __name__ == '__main__':
    data = get_data(r"C:\Users\mariu\Documents\Logs\2018-03-21 08-29-04.csv")

    time = data["Time"]

    velocity = get_velocity(data)
    acceleration = get_acceleration(time, velocity)

    plt.subplot(211).plot(time, velocity)
    plt.subplot(212).plot(time, acceleration)

    plt.show()
