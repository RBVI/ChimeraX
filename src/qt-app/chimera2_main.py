#!/usr/bin/env python
# -*- coding: utf-8 -*-
# vi:set noet sw=8:

import sys

from PySide import QtCore, QtGui, QtOpenGL
from chimera2 import math3d, qtutils

app = None	# QApplication
dump_format = None

class ChimeraGraphics(qtutils.OpenGLWidget):
	"""An OpenGL widget that does LLGR rendering

	Multisampling can be turned on by setting the samples attribute
	to 2 or greater.
	"""
	# TODO: stereo support

	def __init__(self, parent=None, share=None, flags=0):
		self._samples = 4	# 0 turns off multisampling
		format = QtOpenGL.QGLFormat()
		format.setSampleBuffers(True)
		super(ChimeraGraphics, self).__init__(parent, share, flags)
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

	samples = QtCore.Property(int, _getSamples, _setSamples)

	def paintGL(self):
		if app is None:
			# not initialized yet
			return
		from chimera2 import scene
		if dump_format and 'llgr' not in sys.modules:
			import llgr_dump
			sys.modules['llgr'] = llgr_dump
			llgr_dump.set_dump_format(dump_format)
			scene.Position = "position"
		# assume 18 inches from screen
		dist_mm = 18 * 25.4
		height_mm = self.height() / app.DPmm
		import math
		vertical_fov = 2 * math.atan2(height_mm, dist_mm)
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
		print llgr.pick(x, y)

class TextStatus:

	def showMessage(self, text, timeout=0):
		print text

class Application(QtGui.QApplication):
	# TODO: figure out how to catch close/delete window from window frame

	def __init__(self, *args, **kw):
		super(Application, self).__init__(*args, **kw)
		self.setApplicationName("Chimera2")

		self._cmd = ""
		from chimera2 import cmds
		cmds.register('exit', self.cmd_exit)
		cmds.register('open', self.cmd_open)
		# calculate DPmm -- dots (pixels) per mm
		desktop = self.desktop()
		if desktop.widthMM() == 0:
			self.graphics = None
			self.statusbar = TextStatus()
			return
		self.DPmm = min(desktop.width() / desktop.widthMM(),
					desktop.height() / desktop.heightMM())
		self.view = qtutils.create_form("main.ui", opengl={
			"graphicsView": ChimeraGraphics
		    }, connections={
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
		self.view.setWindowTitle("Chimera2")
		self.statusbar = self.find_object("statusbar")
		assert self.statusbar is not None
		self.graphics = self.find_object("graphicsViewGL")
		assert self.graphics is not None
		self._mouse_mode = None
		self.view.show()
		self.cursors = {
			# TODO: custom cursors
			"pick": QtCore.Qt.PointingHandCursor,
			"vsphere_z": QtCore.Qt.IBeamCursor,
			"vsphere_rot": QtCore.Qt.ClosedHandCursor,
			"translate": QtCore.Qt.SizeAllCursor,
		}

	def find_object(self, name):
		return self.view.findChild(QtCore.QObject, name)

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
			self.graphics.setCursor(QtGui.QCursor())

	@QtCore.Slot(QtCore.QEvent)
	def mouse_press(self, event):
		buttons = event.buttons()
		x = event.x()
		y = event.y()
		if buttons & QtCore.Qt.RightButton:
			self.graphics.pick(x, y)
			self.mouse_mode = "pick"
		elif buttons & QtCore.Qt.MiddleButton:
			self.mouse_mode = "translate"
			self.xy = event.globalPos()
		elif buttons & QtCore.Qt.LeftButton:
			zrot = self.graphics.vsphere_press(x, y)
			if zrot:
				self.mouse_mode = "vsphere_z"
			else:
				self.mouse_mode = "vsphere_rot"

	@QtCore.Slot(QtCore.QEvent)
	def mouse_release(self, event):
		if self.mouse_mode in ("vsphere_z", "vsphere_rot"):
			self.graphics.vsphere_release()
		self.mouse_mode = None

	@QtCore.Slot(QtCore.QEvent)
	def mouse_drag(self, event):
		if self.mouse_mode in ("vsphere_z", "vsphere_rot"):
			x = event.x()
			y = event.y()
			throttle = event.modifiers() & QtCore.Qt.ShiftModifier
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

	def status(self, message, timeout=2000):
		# 2000 == 2 seconds
		self.statusbar.showMessage(message, timeout)

	@QtCore.Slot(str)
	def save_command(self, cmd):
		self._cmd = cmd

	@QtCore.Slot()
	def process_command(self):
		from chimera2 import cmds
		try:
			cmds.process_command(self._cmd)
			self.status("")
		except cmds.UserError as e:
			self.status(str(e))

	def cmd_exit(self):
		if self.graphics:
			self.quit()
		else:
			raise SystemExit(0)

	def cmd_open(self, filename):
		self.graphics.makeCurrent()
		from chimera2 import scene
		scene.reset()

		try:
			from chimera2 import data
			data.open(filename)
		except:
			raise
		finally:
			self.graphics.globalXform = math3d.Identity()
			self.graphics.updateGL()

def main():
	# typical Qt application startup
	global app
	app = Application(sys.argv)
	#argv = app.arguments() # TODO -- has python.exe at front, but shouldn't
	argv = sys.argv
	import getopt
	try:
		opts, args = getopt.getopt(argv[1:], 'd:', ['dump='])
	except getopt.error:
		print >> sys.stderr, "usage: %s [-d|--dump format]" % argv[0]
		raise SystemExit, 2
	global dump_format
	for o in opts:
		if o[0] in ("-d", "--dump"):
			dump_format = o[1]
	import llgr_dump
	if dump_format and dump_format not in llgr_dump.FORMATS:
		print >> sys.stderr, "%s: bad format: %s" % (argv[0],
								dump_format)
		print >> sys.stderr, "    available formats: %s" % (
						' '.join(llgr_dump.FORMATS))
		raise SystemExit, 1
	if len(args) > 0:
		print >> sys.stderr, "%s: ignoring extra arguments: %s" % (
						argv[0], ' '.join(args))
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
				app._cmd = raw_input('chimera2: ')
			except EOFError:
				raise SystemExit(0)
			app.process_command()

if __name__ in ( "__main__", "chimeraOpenSandbox"):
	main()
