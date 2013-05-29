"""
qtutils: helper code for Qt
===========================

These are convenience functions to make using Qt easier.

"""

from PyQt5 import QtCore, QtWidgets, QtGui, QtOpenGL
#from contextlib import closing

def create_form(ui_file, parent=None, opengl = {}, connections = {}):
	"""create GUI from .ui file

	The parent argument is needed to nest the GUI inside another widget.

	The opengl argument contains (object_name, widget_factory) pairs,
	and uses the widget_factory to create a new (presumably OpenGL) widget
	inside the GUI widget with the given object_name.  The new widget's
	object name has a 'GL' appended to it so it can be referred to in the
	connections table.

	The connections argument contains (signal, slot) pairs.
	The signal can be given in a dotted pair representation, i.e.,
	"object_name.signal_name" and the slots can either be a dotted pair,
	i.e., "object_name.slot_name", or an explicit reference to the slot.
	"""

	#from PySide import QUiTools
	#loader = QtUiTools.QUiLoader()
	# TODO: add ability to add plugins
	#with closing(QtCore.QFile(ui_file)) as uif:
	#	uif.open(QtCore.QFile.ReadOnly)
	#	form = loader.load(uif, parent)

	from PyQt5 import uic
	form = uic.loadUi(ui_file, parent)

	for object_name, widget_factory in opengl.items():
		obj = form.findChild(QtCore.QObject, object_name)
		if obj is None:
			raise ValueError("missing graphics placeholder")
		# TODO: use layout to replace obj with graphics in layout
		#layout = obj.layout()
		graphics = widget_factory(obj)
		graphics.setObjectName(object_name + "GL")
		grid = QtWidgets.QGridLayout()
		grid.addWidget(graphics, 0, 0)
		obj.setLayout(grid)

	from collections import Callable
	obj_cache = {}
	for src, dest in connections.items():
		object_name, signal_name = src.split('.')
		obj = obj_cache.get(object_name, None)
		if obj is None:
			obj = form.findChild(QtCore.QObject, object_name)
			if obj is None:
				raise ValueError("no child named '%s'" % object_name)
			obj_cache[object_name] = obj
		signal = getattr(obj, signal_name)
		if isinstance(dest, Callable):
			slot = dest
		else:
			obj_name, slot_name = dest.split('.')
			if obj is None:
				obj = form.findChild(QtCore.QObject, object_name)
				if obj is None:
					raise ValueError("no child named '%s'" % object_name)
				obj_cache[object_name] = obj
			slot = getattr(obj, slot_name)
		signal.connect(slot)

	return form

class OpenGLWidget(QtOpenGL.QGLWidget):
	"""Create a Qt Item for 3D OpenGL drawing.

	Actual 3D drawing should be done in a overridden _paintGL function
	in a subclass.	The OpenGL viewport used for drawing is in the
	viewport attribute.  Use the updateGL() member function to indicate
	that the 3D contents need to be updated. The framebuffer format
	is assumed to have color, depth, and stencil buffers.
	"""

	def __init__(self, format, parent=None, share=None, flags=0):
		if share == 0:
			share = None
		if isinstance(flags, int):
			from PyQt5.QtCore import Qt
			flags = Qt.WindowFlags(flags)
		#super().__init__(format, parent, share, flags)
		super().__init__(parent, share, flags)
		#self.setFlag(QtGui.QGraphicsItem.ItemHasNoContents, False)
		self.viewport = None	# (left, bottom, width, height)

		# check format assumptions
		fmt = self.format()
		msg = ""
		sep = ':'
		if not fmt.rgba():
			msg += sep + " missing RGBA buffer"
			sep = ','
		# TODO: Qt on Mac OS X returns -1 for redBufferSize
		#elif (fmt.redBufferSize() < 8 or fmt.greenBufferSize() < 8
		#or fmt.blueBufferSize() < 8):
		#	msg += sep + " less than 24-bit RGBA buffer"
		#	sep = ','
		if not fmt.depth():
			msg += sep + " missing depth buffer"
			sep = ','
		if not fmt.stencil():
			msg += sep + " missing stencil buffer"
			sep = ','
		if msg:
			raise RuntimeError("error: insufficient graphics%s" % msg)

	def resizeGL(self, width, height):
		self.viewport = (0, 0, width, height)

	mousePress = QtCore.pyqtSignal(QtGui.QMouseEvent)
	def mousePressEvent(self, event):
		self.mousePress.emit(event)

	mouseRelease = QtCore.pyqtSignal(QtGui.QMouseEvent)
	def mouseReleaseEvent(self, event):
		self.mouseRelease.emit(event)

	mouseDoubleClick = QtCore.pyqtSignal(QtGui.QMouseEvent)
	def mouseDoubleClickEvent(self, event):
		self.mouseDoubleClick.emit(event)

	mouseMove = QtCore.pyqtSignal(QtGui.QMouseEvent)
	def mouseMoveEvent(self, event):
		self.mouseMove.emit(event)

	wheel = QtCore.pyqtSignal(QtGui.QWheelEvent)
	def wheelEvent(self, event):
		self.wheel.emit(event)

	keyPress = QtCore.pyqtSignal(QtGui.QKeyEvent)
	def keyPressEvent(self, event):
		self.keyPress.emit(event)

	keyRelease = QtCore.pyqtSignal(QtGui.QKeyEvent)
	def keyReleaseEvent(self, event):
		self.keyRelease.emit(event)
