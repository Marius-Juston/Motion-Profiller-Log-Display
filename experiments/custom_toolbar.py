import matplotlib.pyplot as plt
from matplotlib.backend_bases import NavigationToolbar2


class NavigationToolbar(NavigationToolbar2):
    def _init_toolbar(self):
        pass

    def save_figure(self, *args):
        pass

    # only display the buttons we need
    toolitems = [t for t in NavigationToolbar2.toolitems]


plt.show()
