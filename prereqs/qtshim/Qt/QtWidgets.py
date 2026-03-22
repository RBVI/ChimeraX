from . import using_pyqt6, using_pyqt5, using_pyside2, using_pyside6

if using_pyqt6:
    from PyQt6.QtWidgets import *

    # Allow using enum values without enumeration name.
    from .promote_enums import promote_enums_pyqt as promote_enums
    from PyQt6 import QtWidgets
    promote_enums(QtWidgets)
    del QtWidgets

    # Make relocated classes available in their PyQt5 location.
    from PyQt6.QtGui import QAction, QShortcut

elif using_pyqt5:
    from PyQt5.QtWidgets import *

elif using_pyside2:
    from PySide2.QtWidgets import *

elif using_pyside6:
    from PySide6.QtWidgets import *

    from .promote_enums import promote_enums_pyside as promote_enums
    from PySide6 import QtWidgets
    promote_enums(QtWidgets)
    del QtWidgets

    from PySide6.QtGui import QAction, QShortcut


def static_method_kwargs_wrapper(func, from_kwarg_name, to_kwarg_name):
    "Makes static methods accept the `from_kwarg_name` kwarg as `to_kwarg_name`."
    from functools import wraps
    @staticmethod
    @wraps(func)
    def _from_kwarg_name_to_kwarg_name_(*args, **kwargs):
        if from_kwarg_name in kwargs:
            kwargs[to_kwarg_name] = kwargs.pop(from_kwarg_name)
        return func(*args, **kwargs)

    return _from_kwarg_name_to_kwarg_name_
    
if using_pyside6:
    # Make QFileDialog static methods accept the directory kwarg as dir
    for func_name in ('getExistingDirectory', 'getOpenFileName', 'getOpenFileNames', 'getSaveFileName'):
        f = static_method_kwargs_wrapper(getattr(QFileDialog, func_name), "directory", "dir")
        setattr(QFileDialog, func_name, f)
elif using_pyqt6:
    # Make QFileDialog static methods accept the dir kwarg as directory
    for func_name in ('getExistingDirectory', 'getOpenFileName', 'getOpenFileNames', 'getSaveFileName'):
        f = static_method_kwargs_wrapper(getattr(QFileDialog, func_name), "dir", "directory")
        setattr(QFileDialog, func_name, f)
