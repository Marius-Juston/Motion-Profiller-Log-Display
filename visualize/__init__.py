import os

COLUMNS = (
    "Time", "xActual", "yActual", "angleActual", "xTarget", "yTarget", "angleTarget", "XTE", "lagE", "angleE",
    "pLeft", "pRight", "pathNumber", "motionState"
)

MODEL_FILE_NAME = "model.pkl"
MODEL_DATA_FILE_NAME = "data.npz"
open_path = "{0:s}\*.csv".format(os.path.expanduser("~"))
ACCELERATING = 0
DECELERATING = 1
OUTLIER = -1
# DTYPE = tuple("U10" if i == len(COLUMNS) - 1 else np.float32 for i in range(len(COLUMNS)))
DTYPE = None
ENCODING = None
DELIMITER = ", "
