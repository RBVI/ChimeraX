from . import using_pyqt6, using_pyqt5, using_pyside2

if using_pyqt6:
    from PyQt6.QtGui import *

    # Put some commonly used enumeration values in class namespaces.
    enum_values = [
        (QFont, 'Weight', ['Normal', 'Bold']),
        (QFontDatabase, 'SystemFont', ['FixedFont']),
        (QImage, 'Format', ['Format_ARGB32']),
        (QPainter, 'RenderHint', ['Antialiasing']),
        (QPalette, 'ColorRole', ['Window']),
    ]
    for cls, enum_name, values in enum_values:
        enum = getattr(cls, enum_name)
        for value in values:
            setattr(cls, value, getattr(enum, value))

elif using_pyqt5:
    from PyQt5.QtGui import *
    # Add QAction, QShortcut to match the location in Qt 6.
    from PyQt5.QtWidgets import QAction, QShortcut

elif using_pyside2:
    from PySide2.QtGui import *
    # Add QAction, QShortcut to match the location in Qt 6.
    from PySide2.QtWidgets import QAction, QShortcut
