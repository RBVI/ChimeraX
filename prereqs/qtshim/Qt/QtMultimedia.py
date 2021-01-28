from . import using_pyqt5, using_pyside2

if using_pyqt5:
    from PyQt5.QtMultimedia import *

elif using_pyside2:
    from PySide2.QtMultimedia import *
