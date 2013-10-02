from PyQt5 import QtCore, QtGui, QtOpenGL, QtWidgets

# TODO: Monkey business for Qt to find cocoa platform plugin.
#       Can a relative path be built into Qt during compile?
from os.path import join, dirname
import PyQt5
plugins_path = join(dirname(PyQt5.__file__), 'plugins')
QtCore.QCoreApplication.addLibraryPath(plugins_path)

