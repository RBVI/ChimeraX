from . import using_pyqt6, using_pyqt5, using_pyside2

if using_pyqt6:
    from PyQt6.QtMultimedia import *

elif using_pyqt5:
    from PyQt5.QtMultimedia import *

elif using_pyside2:
    from PySide2.QtMultimedia import *
