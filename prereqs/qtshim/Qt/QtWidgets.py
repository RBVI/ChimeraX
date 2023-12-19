from . import using_pyqt6, using_pyqt5, using_pyside2

if using_pyqt6:
    from PyQt6.QtWidgets import *

    # Allow using enum values without enumeration name.
    from .promote_enums import promote_enums
    from PyQt6 import QtWidgets
    promote_enums(QtWidgets)
    del QtWidgets

    # Make relocated classes available in their PyQt5 location.
    from PyQt6.QtGui import QAction, QShortcut

elif using_pyqt5:
    from PyQt5.QtWidgets import *

elif using_pyside2:
    from PySide2.QtWidgets import *
