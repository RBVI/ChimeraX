# vim: set expandtab ts=4 sw=4:

import wx

class GraphicsWindow(wx.Panel):
    """
    The graphics window that displays the three-dimensional models.
    """

    def __init__(self, parent, ui):
        wx.Panel.__init__(self, parent,
            style=wx.TAB_TRAVERSAL|wx.NO_BORDER|wx.WANTS_CHARS)
        self.timer = None
        self.opengl_canvas = OpenGLCanvas(self, ui)
        from wx.glcanvas import GLContext
        oc = self.opengl_context = GLContext(self.opengl_canvas)
        oc.make_current = self.make_context_current
        oc.swap_buffers = self.swap_buffers
        sizer = wx.BoxSizer(wx.HORIZONTAL)
        sizer.Add(self.opengl_canvas, 1, wx.EXPAND)
        self.SetSizerAndFit(sizer)

        self.view = ui.session.main_view
        self.view.initialize_context(oc)

        self.redraw_interval = 16 # milliseconds
        # perhaps redraw interval should be 10 to reduce
        # frame drops at 60 frames/sec

        # temporary mouse bindings
        self.session = ui.session
        self.mouse_down_position = self.last_mouse_position = None
        self.mouse_perimeter = False
        self.opengl_canvas.Bind(wx.EVT_LEFT_DOWN, self.mouse_down)
        self.opengl_canvas.Bind(wx.EVT_MIDDLE_DOWN, self.mouse_down)
        self.opengl_canvas.Bind(wx.EVT_RIGHT_DOWN, self.mouse_down)
        self.opengl_canvas.Bind(wx.EVT_MOTION, self.mouse_drag)
        self.opengl_canvas.Bind(wx.EVT_LEFT_UP, self.mouse_up_select)
        self.opengl_canvas.Bind(wx.EVT_MIDDLE_UP, self.mouse_up)
        self.opengl_canvas.Bind(wx.EVT_RIGHT_UP, self.mouse_up)
        self.opengl_canvas.Bind(wx.EVT_MOUSEWHEEL, self.wheel_event)

    def make_context_current(self):
        # creates context if needed
        if self.timer is None:
            self.timer = wx.Timer(self)
            self.Bind(wx.EVT_TIMER, self._redraw_timer_callback, self.timer)
            self.timer.Start(self.redraw_interval)
        self.opengl_canvas.SetCurrent(self.opengl_context)

    def mouse_contour_level(self, event):
        return

    def mouse_down(self, event):
        self.opengl_canvas.CaptureMouse()
        w,h = self.view.window_size
        x,y = pos = event.GetPosition()
        cx, cy = x - 0.5*w, y - 0.5*h
        fperim = 0.9
        self.mouse_perimeter = (abs(cx) > fperim*0.5*w
            or abs(cy) > fperim*0.5*h)
        self.mouse_down_position = pos
        self.last_mouse_position = pos

    def mouse_drag(self, event):
        if not event.Dragging():
            return
        if not self.opengl_canvas.HasCapture():
            # a Windows thing; can lose mouse capture w/o mouse up
            return
        if event.LeftIsDown():
            self.mouse_rotate(event)
        elif event.MiddleIsDown():
            self.mouse_contour_level(event)
        elif event.RightIsDown():
            self.mouse_translate(event)

    def mouse_motion(self, event):
        lmp = self.last_mouse_position
        x, y = pos = event.GetPosition()
        if lmp is None:
            dx = dy = 0
        else:
            dx = x - lmp[0]
            dy = y - lmp[1]
            # dy > 0 is downward motion
        self.last_mouse_position = pos
        return dx, dy

    def mouse_rotate(self, event):
        axis, angle = self.mouse_rotation(event)
        self.rotate(axis, angle)

    def mouse_rotation(self, event):
        dx, dy = self.mouse_motion(event)
        import math
        angle = 0.5 * math.sqrt(dx*dx + dy*dy)
        if self.mouse_perimeter:
            # z-rotation
            axis = (0,0,1)
            w, h = self.view.window_size
            ex, ey = event.GetX() - 0.5*w, event.GetY() - 0.5*h
            if -dy*ex + dx*ey < 0:
                angle = -angle
        else:
            axis = (dy, dx, 0)
        return axis, angle

    def mouse_select(self, event):
        x,y = event.GetPosition()
        v = self.view
        p, pick = v.first_intercept(x,y)
        toggle = event.ShiftDown()
        # TODO: notify session of selection change
        return
        if pick is None:
            if not toggle:
                self.session.clear_selection()
        else:
            if not toggle:
                self.session.clear_selection()
            pick.select(toggle)
        self.session.clear_selection_hierarchy()

    def mouse_translate(self, event):
        dx, dy = self.mouse_motion(event)
        self.translate((dx, -dy, 0))

    def mouse_up(self, event):
        self.opengl_canvas.ReleaseMouse()
        self.mouse_down_position = None
        self.last_mouse_position = None

    def mouse_up_select(self, event):
        self.opengl_canvas.ReleaseMouse()
        if event.GetPosition() == self.mouse_down_position:
            self.mouse_select(event)
        self.mouse_down_position = None
        self.last_mouse_position = None

    def rotate(self, axis, angle):
        v = self.view
        # Convert axis from camera to scene coordinates
        saxis = v.camera.position.apply_without_translation(axis)
        v.rotate(saxis, angle)

    def swap_buffers(self):
        self.opengl_canvas.SwapBuffers()

    def translate(self, shift):
        v = self.view
        psize = v.pixel_size()
        s = tuple(delta*psize for delta in shift)
        step = v.camera.position.apply_without_translation(s) # scene coord sys
        v.translate(step)

    def wheel_event(self, event):
        # Usually one wheel click is delta of 120
        d = event.GetWheelRotation()/120.0
        v = self.view
        psize = v.pixel_size()
        shift = v.camera.position.apply_without_translation((0, 0, 100*d*psize))
        v.translate(shift)

    def _redraw_timer_callback(self, event):
        self.view.draw(only_if_changed=True)

from wx import glcanvas
class OpenGLCanvas(glcanvas.GLCanvas):

    def __init__(self, parent, ui=None):
        self.graphics_window = parent
        attribs = [ glcanvas.WX_GL_RGBA, glcanvas.WX_GL_DOUBLEBUFFER ]
        import sys
        if sys.platform.startswith('darwin'):
            attribs += [
                glcanvas.WX_GL_OPENGL_PROFILE,
                glcanvas.WX_GL_OPENGL_PROFILE_3_2CORE
            ]
        gl_supported = glcanvas.GLCanvas.IsDisplaySupported
        if not gl_supported(attribs + [0]):
            raise AssertionError("Required OpenGL capabilities, RGBA and/or"
                " double buffering and/or OpenGL 3, not supported")
        for depth in range(32, 0, -8):
            test_attribs = attribs + [glcanvas.WX_GL_DEPTH_SIZE, depth]
            if gl_supported(test_attribs + [0]):
                attribs = test_attribs
                # TODO: log this
                print("Using {}-bit OpenGL depth buffer".format(depth))
                break
        else:
            raise AssertionError("Required OpenGL depth buffer capability"
                " not supported")
        test_attribs = attribs + [glcanvas.WX_GL_STEREO]
        if gl_supported(test_attribs + [0]):
            # TODO: keep track of fact that 3D stereo is available, but
            # don't use it
            pass
        else:
            print("Stereo mode is not supported by OpenGL driver")
        glcanvas.GLCanvas.__init__(self, parent, -1, attribList=attribs + [0],
            style=wx.WANTS_CHARS)

        self.SetBackgroundStyle(wx.BG_STYLE_PAINT)

        if ui:
            self.Bind(wx.EVT_CHAR, ui.forward_keystroke)
        self.Bind(wx.EVT_PAINT, self.OnPaint)
        self.Bind(wx.EVT_SIZE, self.OnSize)

    def OnPaint(self, event):
        self.graphics_window.view.draw()

    def OnSize(self, event):
        wx.CallAfter(self.set_viewport)
        event.Skip()

    def set_viewport(self):
        self.graphics_window.view.resize(*self.GetClientSize())
