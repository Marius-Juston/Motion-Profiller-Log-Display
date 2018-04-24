# coding=utf-8
import easygui

from visualize import new_log_viewer, new_constant_viewer
from visualize import open_path

if __name__ == '__main__':
    # Main loop that runs which allows for the toggling between the different individual programs
    while True:
        answer = easygui.boolbox("Do you wish to visualize logs or view/manipulate constants?",
                                 choices=["[L]ogs", "[C]onstants"])
        if answer is None:
            break
        elif answer:
            open_path = new_log_viewer.main(open_path)  # Runs the log viewer
        else:
            open_path = new_constant_viewer.find_constants(open_path)  # Runs the constant finder
