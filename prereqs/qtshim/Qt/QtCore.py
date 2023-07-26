from . import using_pyqt6, using_pyqt5, using_pyside2

if using_pyqt6:
    from PyQt6.QtCore import *
    from PyQt6.QtCore import pyqtSignal as Signal
    from PyQt6.QtCore import pyqtSlot as Slot
    from PyQt6.QtCore import pyqtProperty as Property
    from PyQt6.QtCore import QT_VERSION_STR as __version__

    # Allow using enum values without enumeration name.
    from .promote_enums import promote_enums
    from PyQt6 import QtCore
    promote_enums(QtCore)
    del QtCore
    
    # Those are imported from `import *`
    del pyqtSignal, pyqtSlot, pyqtProperty, QT_VERSION_STR

elif using_pyqt5:
    from PyQt5.QtCore import *
    from PyQt5.QtCore import pyqtSignal as Signal
    from PyQt5.QtCore import pyqtSlot as Slot
    from PyQt5.QtCore import pyqtProperty as Property
    from PyQt5.QtCore import QT_VERSION_STR as __version__

    # Those are imported from `import *`
    del pyqtSignal, pyqtSlot, pyqtProperty, QT_VERSION_STR

elif using_pyside2:
    from PySide2.QtCore import *
    import PySide2.QtCore
    __version__ = PySide2.QtCore.__version__
