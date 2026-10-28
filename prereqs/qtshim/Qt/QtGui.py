from . import using_pyqt6, using_pyqt5, using_pyside2, using_pyside6

if using_pyqt6:
    from PyQt6.QtGui import *

    # Allow using enum values without enumeration name.
    from .promote_enums import promote_enums_pyqt as promote_enums
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

elif using_pyside6:
    from PySide6 import QtGui as _native_module
    from .promote_enums import native_enum_shortcuts as _native_enum_shortcuts
    _lazy_exports = _native_enum_shortcuts(_native_module)
    if not _lazy_exports:
        from PySide6.QtGui import *

    from .promote_enums import promote_enums_pyside as promote_enums
    from PySide6 import QtGui
    promote_enums(QtGui)
    del QtGui

if using_pyside6 and _lazy_exports:
    from ._lazy import export_module as _export_module
    from .promote_enums import prepare_pyside_class as _prepare_pyside_class
    _export_module(globals(), _native_module, prepare_class=_prepare_pyside_class)
