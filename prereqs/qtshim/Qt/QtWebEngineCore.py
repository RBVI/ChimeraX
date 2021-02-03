from . import using_pyqt5, using_pyside2

if using_pyqt5:
    from PyQt5.QtWebEngineCore import *

elif using_pyside2:
    from PySide2.QtWebEngineCore import *
