import wx

class Chimera(wx.Frame):
	"""Chimera is the main application class.

	The Chimera class creates the main user interface, registers
	for command line input and sets up the graphics."""

	_FileTypes = (	"Python scripts|*.py|"
			"Chimera2 scripts|*.cmd|"
			"Protein Data Bank files|*.pdb|"
			"STereoLithography files|*.stl|"
			"BILD files|*.bild|"
			)

	def __init__(self, app):
		"""Create Chimera application instance.
		
		"app" is the main wx.App instance."""
		
		wx.Frame.__init__(self, None, title="wxChimera")
		self.app = app

		self._create_menu()
		self._create_ui()
		self._create_status_bar()
		from chimera2 import scene
		self.main_view = scene.View()
		self.Show()
		self.command_text.SetFocus()

		from chimera2 import cli, commands
		commands.register()
		self.command = cli.Command()
		cli.register("exit", (), self._cmd_quit)
		cli.register("window", (), self._cmd_window)

		from chimera2.trackchanges import track
		track.add_handler(scene.View, self._update_view)

	def status(self, message, timeout=3000):
		"""Display status message and erase after timeout seconds."""
		self.SetStatusText(message)

	def process_command(self, text):
		from chimera2 import cli
		from chimera2.trackchanges import track
		try:
			track.block()
			self.command.parse_text(text, final=True)
			info = self.command.execute()
			if isinstance(info, str):
				self.status(info)
		except cli.UserError as e:
			self.status(str(e))
		except Exception:
			import traceback
			traceback.print_exc()

	def _create_menu(self):
		self.menubar = wx.MenuBar()
		self.SetMenuBar(self.menubar)
		self.Bind(wx.EVT_MENU, self._quit_cb, id=wx.ID_EXIT)

		file_menu = wx.Menu()
		file_menu.Append(100, "&Open", "Open file", wx.ITEM_NORMAL)
		self.open_dialog = None
		self.Bind(wx.EVT_MENU, self._open_cb, id=100)
		if 1:
			file_menu.AppendSeparator()
			file_menu.Append(101, "E&xit", "Quit wxChimera",
								wx.ITEM_NORMAL)
			self.Bind(wx.EVT_MENU, self._quit_cb, id=101)
		self.menubar.Append(file_menu, "&File")

	def _create_ui(self):
		self.ui_panel = wx.Panel(self)
		box = wx.BoxSizer(wx.VERTICAL)

		self.canvas = ChimeraGLCanvas(self, self.ui_panel)
		box.Add(self.canvas, 10, wx.EXPAND|wx.ALL, 0)

		cmdline = wx.BoxSizer(wx.HORIZONTAL)
		box.Add(cmdline, 0, wx.BOTTOM|wx.EXPAND, 0)
		t = wx.StaticText(self.ui_panel, label= "Command:")
		cmdline.Add(t, 0, wx.ALL, 2)
		self.command_text = wx.TextCtrl(self.ui_panel,
					style=wx.TE_PROCESS_ENTER)
		cmdline.Add(self.command_text, 10, wx.EXPAND|wx.ALL, 1)
		self.Bind(wx.EVT_TEXT_ENTER, self._command_cb,
					self.command_text)

		self.ui_panel.SetSizer(box)

	def _create_status_bar(self):
		self.CreateStatusBar()

	#
	# Callbacks registered for UI
	#
	def _quit_cb(self, evt):
		if self.open_dialog is not None:
			self.open_dialog.Destroy()
			self.open_dialog = None
		self.Destroy()

	def _open_cb(self, evt):
		if self.open_dialog is None:
			import os
			self.open_dialog = wx.FileDialog(self.ui_panel,
						defaultDir=os.getcwd(),
						defaultFile="",
						wildcard=self._FileTypes,
						style=wx.FD_OPEN |
							wx.FD_MULTIPLE |
							wx.FD_FILE_MUST_EXIST |
							wx.FD_PREVIEW)
		if self.open_dialog.ShowModal() != wx.ID_OK:
			print("cancelled open")
			return
		paths = self.open_dialog.GetPaths()
		from chimera2 import commands
		for p in paths:
			print("open:", p)
			commands.cmd_open(p)

	def _command_cb(self, evt):
		cmd = self.command_text.GetValue()
		print("Command:", cmd);
		self.process_command(cmd)
		self.command_text.SelectAll()

	#
	# Callbacks registered for commands
	#
	def _cmd_quit(self):
		self._quit_cb(None)

	def _cmd_window(self):
		self.main_view.reset_camera()

	#
	# Callbacks registered for data updates
	#
	def _update_view(self, *args, **kw):
		self.canvas.Refresh(True)

from wx import glcanvas
class ChimeraGLCanvas(glcanvas.GLCanvas):

	_DEFAULT_ATTRIB_LIST = [
		glcanvas.WX_GL_DOUBLEBUFFER,
		glcanvas.WX_GL_MIN_RED, 8,
		glcanvas.WX_GL_MIN_ALPHA, 0,
		glcanvas.WX_GL_DEPTH_SIZE, 8,
		glcanvas.WX_GL_OPENGL_PROFILE,
			glcanvas.WX_GL_OPENGL_PROFILE_3_2CORE,
		0
	]

	def __init__(self, chimera, parent, attr_list=None):
		if attr_list is None:
			attr_list = self._DEFAULT_ATTRIB_LIST
		glcanvas.GLCanvas.__init__(self, parent, attribList=attr_list)

		self.chimera = chimera
		self.context = glcanvas.GLContext(self)
		import llgr
		llgr.set_output("pyopengl")

		import OpenGL
		OpenGL.ERROR_LOGGING = True
		OpenGL.ERROR_ON_COPY = True
		OpenGL.FORWARD_COMPATIBILITY_ONLY = True
		import logging
		#logging.basicConfig(level=logging.ERROR)
		OpenGL.FULL_LOGGING = True
		logging.basicConfig(level=logging.DEBUG)

		self.cursor_rotate = wx.Cursor(wx.CURSOR_BULLSEYE)
		self.cursor_translate = wx.Cursor(wx.CURSOR_HAND)
		self.cursor_pick = wx.Cursor(wx.CURSOR_CROSS)

		self.Bind(wx.EVT_SIZE, self._size_cb)
		self.Bind(wx.EVT_PAINT, self._paint_cb)

		self.vsphere_id = 1
		self._mouse_mode = None	# one of None, 'rotate', 'translate', 'scale'
		self._motion_bound = False
		self._last_xy = None
		self.Bind(wx.EVT_LEFT_DOWN, self._button_down_cb)
		self.Bind(wx.EVT_MIDDLE_DOWN, self._button_down_cb)
		self.Bind(wx.EVT_LEFT_UP, self._button_up_cb)
		self.Bind(wx.EVT_MIDDLE_UP, self._button_up_cb)
		#self.Bind(wx.EVT_MOUSE_EVENTS, self._log_mouse_cb)

	def _initGL(self):
		self._initGL = None
		self._size_cb(None)
		from OpenGL import GL
		print("Version:", GL.glGetString(GL.GL_VERSION))
		print("Vendor:", GL.glGetString(GL.GL_VENDOR))
		print("Renderer:", GL.glGetString(GL.GL_RENDERER))

	def _size_cb(self, evt):
		if not self.IsShownOnScreen() or self._initGL:
			return
		self.SetCurrent(self.context)
		width, height = self.GetClientSize()
		from OpenGL import GL
		GL.glViewport(0, 0, width, height)
		# assume 18 inches from screen
		dist_in = 18
		res_x, res_y = wx.ScreenDC().GetPPI()
		height_in = height / res_y
		import math
		self.chimera.main_view.fov = 2 * math.atan2(height_in, dist_in)
		self.chimera.main_view.viewport = (width, height)

	def _paint_cb(self, evt):
		self.SetCurrent(self.context)
		if self._initGL:
			self._initGL()
		self.chimera.main_view.render()
		self.SwapBuffers()

	def _button_down_cb(self, evt):
		button = evt.GetButton()
		if button == wx.MOUSE_BTN_LEFT:
			if evt.AltDown():
				self._mouse_mode = "translate"
			else:
				self._mouse_mode = "rotate"
		elif button == wx.MOUSE_BTN_MIDDLE:
			self._mouse_mode = "translate"
		else:
			self._button_up_cb(None)
			return
		self._bind_motion()
		if self._mouse_mode == "rotate":
			zrot = self._vsphere_press(evt.GetX(), evt.GetY())
			self.SetCursor(self.cursor_rotate)
		elif self._mouse_mode == "translate":
			self._last_xy = (evt.GetX(), evt.GetY())
			self.SetCursor(self.cursor_translate)
		else:
			self.SetCursor(wx.NullCursor)

	def _button_up_cb(self, evt):
		if self._mouse_mode == "rotate":
			self._vsphere_release()
		self._mouse_mode = None
		self._bind_motion()
		self.SetCursor(wx.NullCursor)

	def _log_mouse_cb(self, evt):
		print("_mouse_cb type:", evt.GetEventType(),
				"button:", evt.GetButton(),
				"dragging:", evt.Dragging(),
				"moving:", evt.Moving(),
				"x:", evt.GetX(), "y:", evt.GetY())

	def _bind_motion(self):
		bind = self._mouse_mode is not None
		if bind:
			if not self._motion_bound:
				self.Bind(wx.EVT_MOTION, self._mouse_motion_cb)
				self._motion_bound = True
		else:
			if self._motion_bound:
				self.Unbind(wx.EVT_MOTION)
				self._motion_bound = False

	def _mouse_motion_cb(self, evt):
		x = evt.GetX()
		y = evt.GetY()
		if self._mouse_mode == "rotate":
			zrot = self._vsphere_drag(x, y, evt.ShiftDown())
		elif self._mouse_mode == "translate":
			deltaX = x - self._last_xy[0]
			deltaY = y - self._last_xy[1]
			self._last_xy = (x, y)
			self._translate_xy(deltaX, deltaY)

	#
	# Virtual sphere methods
	#
	def _vsphere_press(self, x, y):
		import llgr
		width, height = self.GetClientSize()
		llgr.vsphere_setup(self.vsphere_id, 0.4 * min(width, height),
						(width / 2.0, height / 2.0))
		cursor = llgr.vsphere_press(self.vsphere_id, x, y)
		return cursor

	def _vsphere_drag(self, x, y, throttle):
		import llgr
		cursor, axis, angle = llgr.vsphere_drag(self.vsphere_id, x, y, throttle)
		if angle == 0:
			return cursor
		# update global transformation matrix
		from chimera2 import math3d
		rot = math3d.Rotation(axis, angle)
		if rot.isIdentity:
			return
		main_view = self.chimera.main_view
		try:
			center = main_view.bbox.center()
		except ValueError:
			return
		if main_view.camera:
			main_view.camera.xform(rot)
		self.Refresh(True)
		return cursor

	def _vsphere_release(self):
		pass
		#import llgr
		#llgr.vsphere_release(self.vsphere_id)

	def _translate_xy(self, deltaX, deltaY):
		main_view = self.chimera.main_view
		try:
			width, height, _ = main_view.bbox.size()
		except ValueError:
			return
		w, h = self.GetClientSize()
		dx = deltaX * width / w
		dy = -deltaY * height / h
		from chimera2 import math3d
		trans = math3d.Translation((dx, dy, 0))
		if main_view.camera:
			main_view.camera.xform(trans)
		self.Refresh(True)

	def _pick(self, x, y):
		self.makeCurrent()
		import llgr
		w, h = self.GetClientSize()
		y = h - y
		print(llgr.pick(x, y))

def main():
	try:
		import sys
		sys.path.insert(0, "../../build/lib")
		app = wx.App(redirect=False)
		app.SetAppName("wxChimera")
		chimera = Chimera(app)
		from chimera2 import scene
		scene.reset()
		from chimera2 import io
		io.initialize_formats()
		app.MainLoop()
	except:
		import traceback
		traceback.print_exc()

if __name__ == "__main__":
	main()
