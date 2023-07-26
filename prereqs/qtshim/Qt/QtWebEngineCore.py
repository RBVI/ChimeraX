from . import using_pyqt6, using_pyqt5, using_pyside2

if using_pyqt6:
    from PyQt6.QtWebEngineCore import *
    from .promote_enums import promote_enums
    from PyQt6 import QtWebEngineCore
    promote_enums(QtWebEngineCore)
    del QtWebEngineCore

elif using_pyqt5:
    from PyQt5.QtWebEngineCore import *
    # Match the location of QWebEnginePage, QWebEngineProfile in Qt 6.
    from PyQt5.QtWebEngineWidgets import QWebEnginePage, QWebEngineProfile

elif using_pyside2:
    from PySide2.QtWebEngineCore import *
    # Match the location of QWebEnginePage, QWebEngineProfile in Qt 6.
    from PySide2.QtWebEngineWidgets import QWebEnginePage, QWebEngineProfile
