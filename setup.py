import os

import sys
from cx_Freeze import setup, Executable

base = None

python_environment = os.path.dirname(sys.executable)

# print(r"{0:s}\tcl\{1:s}".format(python_environment,"tcl8.6"))

os.environ['TCL_LIBRARY'] = r"{0:s}\tcl\{1:s}".format(python_environment,"tcl8.6")
os.environ['TK_LIBRARY'] = r"{0:s}\tcl\{1:s}".format(python_environment,"tk8.6")

executables = [Executable("driver.py", base=base)]

packages = ["numpy", "easygui", "matplotlib", ]
options = {
    'build_exe': {
        'packages': packages,
    },
}

setup(
    name="Log file shower",
    options=options,
    version="v1",
    description='Draws the log file data in graphs',
    executables=executables
)
