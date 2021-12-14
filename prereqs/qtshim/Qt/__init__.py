"""
This Qt shim module allows using PyQt5 or PySide2 Python bindings to the Qt C++ library.
It allows easily switching between these two almost equivalent bindings without changing
all the imports in application code.

In the future we will try to also allow using PyQt6 and PySide6 if Qt 6 is sufficiently
compatible with Qt 5.

PyQt5 is used if it can be imported, otherwise PySide2 is used.

Example use:

>>> from Qt.QtWidgets import QPushButton

This shim was derived from qtpy version 1.9.0.  For a variety reasons it seemed best not
to try to patch all the deficiencies in qtpy.  I also tried Qt.py 1.3.3 -- it did not have
support for QtWebEngine.

Problems with qtpy included: It did not have QWebEngineCore and QWebEngineProfile.
The missing classes were reported years ago and ignored by the developers suggesting a lack of
support.  Most of the code was to handle the ancient PyQt4 and PySide.  Also it tried to
handle Python 2 and 3.  We don't need support for those ancient versions.
It did not wrap a method to determine object deletion (sip.isdeleted(), shiboken2.isValid()).
It did not have PyQt6 or PySide6 support and a ticket asking about that got no response.
The wrapping is very simple, and especially for experimenting with Qt 6 support it seemed
more productive to use our own shim code.

"""

# Choose between PyQt5 and PySide2
using_pyqt6 = using_pyqt5 = using_pyside2 = False
using_qt5 = using_qt6 = False
try:
    import PyQt6
    using_pyqt6 = True
    using_qt6 = True
except ImportError:
    try:
        import PyQt5
        using_pyqt5 = True
        using_qt5 = True
    except ImportError:
        import PySide2
        using_pyside2 = True
        using_qt5 = True

if using_pyqt6:
    from PyQt6.QtCore import PYQT_VERSION_STR as PYQT6_VERSION
    from PyQt6.QtCore import QT_VERSION_STR as QT_VERSION
    version = 'PyQt6 %s, Qt %s' % (PYQT6_VERSION, QT_VERSION)

    def qt_object_is_deleted(object):
        '''Return whether a C++ Qt QObject has been deleted.'''
        from PyQt6 import sip
        return sip.isdeleted(object)

    def qt_image_bytes(qimage):
        return qimage.bits().asstring(qimage.sizeInBytes())

if using_pyqt5:
    from PyQt5.QtCore import PYQT_VERSION_STR as PYQT5_VERSION
    from PyQt5.QtCore import QT_VERSION_STR as QT_VERSION
    version = 'PyQt5 %s, Qt %s' % (PYQT5_VERSION, QT_VERSION)

    def qt_object_is_deleted(object):
        '''Return whether a C++ Qt QObject has been deleted.'''
        import sip
        return sip.isdeleted(object)

    def qt_image_bytes(qimage):
        return qimage.bits().asstring(qimage.byteCount())

if using_pyside2:
    from PySide2 import __version__ as PYSIDE2_VERSION
    from PySide2.QtCore import __version__ as QT_VERSION
    version = 'PySide2 %s, Qt %s' % (PYSIDE2_VERSION, QT_VERSION)
    
    def qt_object_is_deleted(object):
        '''Return whether a C++ Qt QObject has been deleted.'''
        import shiboken2
        return not shiboken2.isValid(object)

    def qt_image_bytes(qimage):
        return qimage.bits().tobytes()

def qt_have_web_engine():
    try:
        from . import QtWebEngineWidgets
    except Exception:
        return False
    return True
