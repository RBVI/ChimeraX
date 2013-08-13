#!/usr/bin/env python
# -*- coding: utf-8 -*-
# vi:set noet sw=8:

import sys

from PyQt5.QtCore import (pyqtProperty, pyqtSlot, Qt, QObject,
		QCoreApplication, QStringListModel, QTimer, QEvent)
from PyQt5.QtGui import QCursor
from PyQt5.QtWidgets import QApplication, QCompleter, QFileDialog
from PyQt5.QtOpenGL import QGLFormat
from chimera2 import math3d, qtutils

app = None	# QApplication or QCoreApplication
dump_format = None

class ChimeraGraphics(qtutils.OpenGLWidget):
	"""An OpenGL widget that does LLGR rendering

	Multisampling can be turned on by setting the samples attribute
	to 2 or greater.
	"""
	# TODO: stereo support

	def __init__(self, parent=None, share=None, flags=0):
		self._samples = 4	# 0 turns off multisampling
		# TODO: format = QGLFormat()
		# TODO: format.setSampleBuffers(True)
		# TODO: add format to below
		super().__init__(parent, share, flags)
		self.vsphere_id = 1
		self.globalXform = math3d.Identity()

	def _getSamples(self):
		return self._samples

	def _setSamples(self, value):
		if not isinstance(value, int) or value < 0:
			raise ValueError('samples must be non-negative')
		if self._samples == value:
			return
		self._samples = value
		self.updateGL()

	samples = pyqtProperty(int, _getSamples, _setSamples)

	def paintGL(self):
		if app is None:
			# not initialized yet
			return
		from chimera2 import scene
		# assume 18 inches from screen
		dist_in = 18
		height_in = self.height() / app.physicalDotsPerInch()
		import math
		vertical_fov = 2 * math.atan2(height_in, dist_in)
		scene.render(self.viewport, vertical_fov, self.globalXform)

	def vsphere_press(self, x, y):
		import llgr
		width = self.width()
		height = self.height()
		llgr.vsphere_setup(self.vsphere_id, 0.4 * min(width, height),
						(width / 2.0, height / 2.0))
		cursor = llgr.vsphere_press(self.vsphere_id, x, y)
		return cursor

	def vsphere_drag(self, x, y, throttle):
		import llgr
		cursor, axis, angle = llgr.vsphere_drag(self.vsphere_id, x, y, throttle)
		if angle == 0:
			return cursor
		# update global transformation matrix
		rot = math3d.Rotation(axis, angle)
		if rot.isIdentity:
			return
		from chimera2 import scene
		try:
			center = scene.bbox.center()
		except ValueError:
			return
		trans = math3d.Translation(center)
		inv_trans = math3d.Translation(-center)
		self.globalXform = trans * rot * inv_trans * self.globalXform
		self.updateGL()
		return cursor

	def vsphere_release(self):
		pass
		#import llgr
		#llgr.vsphere_release(self.vsphere_id)

	def translate_xy(self, delta):
		from chimera2 import scene
		try:
			width, height, _ = scene.bbox.size()
		except ValueError:
			return
		dx = delta.x() * width / self.width()
		dy = -delta.y() * height / self.height()
		trans = math3d.Translation((dx, dy, 0))
		self.globalXform = trans * self.globalXform
		self.updateGL()

	def pick(self, x, y):
		self.makeCurrent()
		import llgr
		y = int(self.height()) - y
		print(llgr.pick(x, y))

class BaseApplication:

	def __init__(self, *args, **kw):
		self.setApplicationName("Chimera2")
		self.setApplicationVersion("0.9")	# TODO
		self.setOrganizationDomain("cgl.ucsf.edu")
		self.setOrganizationName("UCSF RBVI")

		from chimera2 import cmds
		self.command = cmds.Command()
		cmds.register('exit', self.cmd_exit)
		cmds.register('open', self.cmd_open)
		cmds.register('stop', self.cmd_stop)
		cmds.register('stereo', self.cmd_noop)

		# potentially changed in subclass:
		self.graphics = None
		self.statusbar = None

	def cmd_noop(self):
		pass

	def status(self, message, timeout=2000):
		# 2000 == 2 seconds
		if self.statusbar:
			self.statusbar.showMessage(message, timeout)
		else:
			print(message)

	def process_command(self, text=""):
		from chimera2 import cmds
		try:
			if text:
				self.command.parse_text(text, autocomplete=True)
			self.command.execute()
		except cmds.UserError as e:
			self.status(str(e))
		except Exception:
			# TODO: report error
			import traceback
			traceback.print_exc()

	def cmd_exit(self):
		# TODO: if nogui starts using event loop, then just self.quit
		if self.graphics:
			self.quit()
		else:
			raise SystemExit(0)

	def cmd_stop(self):
		self.status('use "exit"')

	def cmd_open(self, filename):
		if self.graphics:
			self.graphics.makeCurrent()
		from chimera2 import scene
		scene.reset()

		try:
			from chimera2 import data
			data.open(filename)
		except:
			raise
		finally:
			if self.graphics:
				self.graphics.globalXform = math3d.Identity()
				self.graphics.updateGL()

class ConsoleApplication(QCoreApplication, BaseApplication):

	def __init__(self, *args, **kw):
		QCoreApplication.__init__(self, *args, **kw)
		BaseApplication.__init__(self)

		self.window_size = (200, 200)

		from chimera2 import cmds
		cmds.register('render', self.cmd_render)
		cmds.register('windowsize', self.cmd_window_size)

	def physicalDotsPerInch(self):
		# assume 100 dpi
		return 100

	def cmd_render(self):
		from chimera2 import scene
		# assume 18 inches from screen
		dist_in = 18
		height_in = self.window_size[1] / self.physicalDotsPerInch()
		import math
		vertical_fov = 2 * math.atan2(height_in, dist_in)
		viewport = (0, 0) + self.window_size
		scene.render(viewport, vertical_fov, math3d.Identity())

	def cmd_window_size(self, width: int, height: int):
		self.window_size = (width, height)


class GuiApplication(QApplication, BaseApplication):
	# TODO: figure out how to catch close/delete window from window frame

	def __init__(self, *args, **kw):
		QApplication.__init__(self, *args, **kw)
		BaseApplication.__init__(self)

		self.view = qtutils.create_form("main.ui", opengl={
			"graphicsView": ChimeraGraphics
		    }, connections={
			"actionOpen.triggered": self.open,
			"actionQuit.triggered": self.quit,
			"lineEdit.textChanged": self.save_command,
			"lineEdit.returnPressed": self.process_command,
			"graphicsViewGL.mousePress": self.mouse_press,
			"graphicsViewGL.mouseRelease": self.mouse_release,
			"graphicsViewGL.mouseMove": self.mouse_drag,
			# TODO: why are't these needed?
			#"graphicsViewGL.keyPress": "lineEdit.event",
			#"graphicsViewGL.keyRelease": "lineEdit.event",
		})
		self.view.setWindowTitle(self.applicationName())
		self.statusbar = self.find_object("statusbar")
		assert self.statusbar is not None
		self.graphics = self.find_object("graphicsViewGL")
		assert self.graphics is not None
		self.graphics.setFocusPolicy(Qt.WheelFocus)
		self.line_edit = self.find_object("lineEdit")
		assert self.line_edit is not None
		self.completer = QCompleter(self.view)
		self.completer.setModel(QStringListModel(self.completer))
		#self.completer.setCompletionMode(QCompleter.PopupCompletion)
		self.line_edit.setCompleter(self.completer)
		self._mouse_mode = None
		self.view.show()
		self.cursors = {
			# TODO: custom cursors
			"pick": Qt.PointingHandCursor,
			"vsphere_z": Qt.IBeamCursor,
			"vsphere_rot": Qt.ClosedHandCursor,
			"translate": Qt.SizeAllCursor,
		}
		self.timer = QTimer(self.view)
		self.active_timer = False

	def physicalDotsPerInch(self):
		screen = self.primaryScreen()
		return screen.physicalDotsPerInch()

	def find_object(self, name):
		return self.view.findChild(QObject, name)

	@pyqtSlot()
	def open(self):
		# QFileDialog.getOpenFileName(QWidget parent=None, str caption='', str directory='', str filter='', str initialFilter='', QFileDialog.Options options=0) -> (str, str)
		from chimera2 import data
		filename, filter = QFileDialog.getOpenFileName(
				self.view, caption="Open File",
				filter=data.qt_open_file_filter())
		if filename:
			self.cmd_open(filename)

	@property
	def mouse_mode(self):
		return self._mouse_mode

	@mouse_mode.setter
	def mouse_mode(self, mode):
		if mode == self._mouse_mode:
			return
		self._mouse_mode = mode
		cursor = self.cursors.get(mode, None)
		if cursor:
			self.graphics.setCursor(cursor)
		else:
			self.graphics.setCursor(QCursor())

	@pyqtSlot(QEvent)
	def mouse_press(self, event):
		buttons = event.buttons()
		x = event.x()
		y = event.y()
		if buttons & Qt.RightButton:
			self.graphics.pick(x, y)
			self.mouse_mode = "pick"
		elif buttons & Qt.MiddleButton:
			self.mouse_mode = "translate"
			self.xy = event.globalPos()
		elif buttons & Qt.LeftButton:
			zrot = self.graphics.vsphere_press(x, y)
			if zrot:
				self.mouse_mode = "vsphere_z"
			else:
				self.mouse_mode = "vsphere_rot"

	@pyqtSlot(QEvent)
	def mouse_release(self, event):
		if self.mouse_mode in ("vsphere_z", "vsphere_rot"):
			self.graphics.vsphere_release()
		self.mouse_mode = None

	@pyqtSlot(QEvent)
	def mouse_drag(self, event):
		if self.mouse_mode in ("vsphere_z", "vsphere_rot"):
			x = event.x()
			y = event.y()
			throttle = event.modifiers() & Qt.ShiftModifier
			zrot = self.graphics.vsphere_drag(x, y, throttle)
			if zrot:
				self.mouse_mode = "vsphere_z"
			else:
				self.mouse_mode = "vsphere_rot"
		elif self.mouse_mode == "translate":
			xy = event.globalPos()
			delta = xy - self.xy
			self.xy = xy
			self.graphics.translate_xy(delta)

	@pyqtSlot(str)
	def save_command(self, text):
		self.command.parse_text(text)
		self.completer.setCompletionPrefix(self.command.completion_prefix)
		self.completer.model().setStringList(self.command.completions)
		self.completer.complete()

	@pyqtSlot()
	def process_command(self, cmd=None):
		self.status("")
		BaseApplication.process_command(self)

def set_default_context(major_version, minor_version, profile):
	f = QGLFormat()
	f.setVersion(major_version, minor_version)
	f.setProfile(profile)
	QGLFormat.setDefaultFormat(f)

def main():
	# typical Qt application startup
	global app
	if '--nogui' in sys.argv:
		app = ConsoleApplication(sys.argv)
	else:
		set_default_context(3, 2, QGLFormat.CoreProfile)
		app = GuiApplication(sys.argv)
	argv = sys.argv
	argv[0] = app.applicationName().casefold()
	import getopt
	try:
		opts, args = getopt.getopt(argv[1:], 'd:', ['dump=', 'nogui'])
	except getopt.error:
		print("usage: %s [--nogui] [-d|--dump format]" % argv[0],
				file=sys.stderr)
		raise SystemExit(2)
	global dump_format
	for option, value in opts:
		if option in ("-d", "--dump"):
			dump_format = value
		elif option == '--nogui':
			pass

	sys.path.insert(0, '../../build/lib')

	from llgr.dump import FORMATS
	if not app.graphics and not dump_format:
		print("%s: need non-opengl dump format in nogui mode"
				% sys.argv[0], file=sys.stderr)
		print("    available formats: %s" % ' '.join(FORMATS),
				file=sys.stderr)
		raise SystemExit(1)
	if dump_format and dump_format not in FORMATS:
		print("%s: bad format: %s" % (argv[0], dump_format),
				file=sys.stderr)
		print("    available formats: %s" % ' '.join(FORMATS),
				file=sys.stderr)
		raise SystemExit(1)

	import llgr
	if dump_format:
		llgr.set_output(dump_format)
	else:
		llgr.set_output('pyopengl')
	if len(args) > 0:
		print("%s: ignoring extra arguments: %s"
				% (argv[0], ' '.join(args)), file=sys.stderr)
	from chimera2 import data, bild, stl
	data.register_format("BILD",
		bild.open, None, None, (".bild",), (),
		category=data.GENERIC3D,
		reference="http://www.cgl.ucsf.edu/chimera/docs/UsersGuide/bild.html")
	data.register_format("STL",
		stl.open, None, None, (".stl",), (),
		category=data.GENERIC3D,
		reference="http://en.wikipedia.org/wiki/STL_%28file_format%29")
	if app.graphics:
		sys.exit(app.exec_())
	else:
		while 1:
			try:
				cmd_text = input('chimera2: ')
			except EOFError:
				raise SystemExit(0)
			app.process_command(cmd_text)

if __name__ in ( "__main__", "chimeraOpenSandbox"):
	main()
