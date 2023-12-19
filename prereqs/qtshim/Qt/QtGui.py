from . import using_pyqt6, using_pyqt5, using_pyside2

if using_pyqt6:
    from PyQt6.QtGui import *

    # Allow using enum values without enumeration name.
    from .promote_enums import promote_enums
    from PyQt6 import QtGui
    promote_enums(QtGui)
    del QtGui

elif using_pyqt5:
    from PyQt5.QtGui import *
    # Add QAction, QShortcut to match the location in Qt 6.
    from PyQt5.QtWidgets import QAction, QShortcut

elif using_pyside2:
    from PySide2.QtGui import *
    # Add QAction, QShortcut to match the location in Qt 6.
    from PySide2.QtWidgets import QAction, QShortcut
