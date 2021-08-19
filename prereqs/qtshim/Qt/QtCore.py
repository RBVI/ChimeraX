from . import using_pyqt5, using_pyside2

if using_pyqt5:
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
