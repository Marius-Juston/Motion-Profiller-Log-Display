import matplotlib.pyplot as plt

from visualize.helper import get_data, get_velocity, get_acceleration

if __name__ == '__main__':
    data = get_data(r"C:\Users\mariu\Documents\Logs\2018-03-21 08-29-04.csv")

    time = data["Time"]

    velocity = get_velocity(time, data, actual=False)
    acceleration = get_acceleration(time, velocity)

    velocities = plt.subplot(211)
    accelerations = plt.subplot(212)
    velocities.plot(time, velocity)
    accelerations.scatter(time, acceleration)
    import numpy as np

    X = np.linspace(time.min(), time.max(), velocity.shape[0])

    coefficients = np.polyfit(time, np.log(velocity), 10)  # Use log(x) as the input to polyfit.

    velocities.plot(time, np.polyval(coefficients, time), "--", label="fit")

    velocity = get_velocity(time, data, actual=True)
    acceleration = get_acceleration(time, velocity)

    velocities.plot(time, velocity)
    accelerations.scatter(time, acceleration)

    plt.show()
