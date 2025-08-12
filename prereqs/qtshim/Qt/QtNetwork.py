from . import using_pyqt6, using_pyqt5, using_pyside2, using_pyside6

if using_pyqt6:
    from PyQt6.QtNetwork import *

    # Allow using enum values without enumeration name.
    from .promote_enums import promote_enums_pyqt as promote_enums
    from PyQt6 import QtNetwork
    promote_enums(QtNetwork)
    del QtNetwork

elif using_pyqt5:
    from PyQt5.QtNetwork import *
    
elif using_pyside2:
    from PySide2.QtNetwork import *
 
elif using_pyside6:
    from PySide6.QtNetwork import *

    from .promote_enums import promote_enums_pyside as promote_enums
    from PySide6 import QtNetwork
    promote_enums(QtNetwork)
    del QtNetwork

