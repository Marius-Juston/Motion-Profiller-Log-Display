import os

COLUMNS = (
    "Time", "xActual", "yActual", "angleActual", "xTarget", "yTarget", "angleTarget", "XTE", "lagE", "angleE",
    "pLeft", "pRight", "pathNumber", "motionState"
)

USER_HOME_DIRECTORY = os.path.expanduser("~")

MASTER_FOLDER = "{0:s}/MotionViewer/".format(USER_HOME_DIRECTORY)

if not os.path.exists(MASTER_FOLDER):
    os.makedirs(MASTER_FOLDER)

MODEL_FILE = "{0:s}model.pkl".format(MASTER_FOLDER)
MODEL_DATA_FILE = "{0:s}data.npz".format(MASTER_FOLDER)
open_path = "{0:s}\*.csv".format(USER_HOME_DIRECTORY)
ACCELERATING = 0
DECELERATING = 1
OUTLIER = -1
# DTYPE = tuple("U10" if i == len(COLUMNS) - 1 else np.float32 for i in range(len(COLUMNS)))
DTYPE = None
ENCODING = None
DELIMITER = ", "
