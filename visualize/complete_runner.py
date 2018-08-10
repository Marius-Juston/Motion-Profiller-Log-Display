# coding=utf-8
import easygui

from visualize import new_log_viewer, new_constant_viewer
from visualize import open_path
from experiments import new_model_trainer

LOGS = "[L]ogs"
CONSTANTS = "[C]onstants"
TRAIN = "[T]rain"

if __name__ == '__main__':
    # Main loop that runs which allows for the toggling between the different individual programs
    while True:
        answer = easygui.buttonbox(
            "Do you wish to visualize logs, view/manipulate constants or train the data classifier?",
            choices=[LOGS, CONSTANTS, TRAIN])

        if answer is None:
            break
        elif answer == LOGS:
            open_path = new_log_viewer.main(open_path)  # Runs the log viewer
        elif answer == CONSTANTS:
            open_path = new_constant_viewer.find_constants(open_path)  # Runs the constant finder
        else:
            open_path = new_model_trainer.train_model(open_path)
