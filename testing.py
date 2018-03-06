import matplotlib.pyplot as plt
from matplotlib import gridspec


def main():
    gs = gridspec.GridSpec(3, 2)
    ax1 = plt.subplot(gs[0:2, :])
    ax4 = plt.subplot(gs[-1, 0])
    ax5 = plt.subplot(gs[-1, -1])

    plt.show()


if __name__ == '__main__':
    main()
