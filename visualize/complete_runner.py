import easygui

from visualize import constants_finder, new_log_viewer
from visualize import open_path

if __name__ == '__main__':
    while True:
        answer = easygui.boolbox("Do you wish to visualize logs or view/manipulate constants?",
                                 choices=["[L]ogs", "[C]onstants"])
        if answer is None:
            break
        elif answer:
            open_path = new_log_viewer.main(open_path)
        else:
            open_path = constants_finder.main(open_path)
