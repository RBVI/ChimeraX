# vi: set expandtab ts=4 sw=4:

import wx

class GraphicsWindow(wx.Panel):
    """
    The graphics window that displays the three-dimensional models.
    """

    def __init__(self, parent, ui):
        wx.Panel.__init__(self, parent,
            style=wx.TAB_TRAVERSAL | wx.NO_BORDER | wx.WANTS_CHARS)
        self.timer = None
        self.session = ui.session
        self.view = ui.session.main_view
        self.opengl_canvas = OpenGLCanvas(self, self.view, ui)
        if ui.have_stereo:
            from ..graphics import StereoCamera
            self.view.camera = StereoCamera()
        from wx.glcanvas import GLContext
        oc = self.opengl_context = GLContext(self.opengl_canvas)
        oc.make_current = self.make_context_current
        oc.swap_buffers = self.swap_buffers
        self.view.initialize_context(oc)
        sizer = wx.BoxSizer(wx.HORIZONTAL)
        sizer.Add(self.opengl_canvas, 1, wx.EXPAND)
        self.SetSizerAndFit(sizer)

        self.popup = Popup(parent)        # For display of atom spec balloons

        self.redraw_interval = 16  # milliseconds
        # perhaps redraw interval should be 10 to reduce
        # frame drops at 60 frames/sec

        from .mousemodes import MouseModes
        self.mouse_modes = MouseModes(self, ui.session)

    def set_redraw_interval(self, msec):
        self.redraw_interval = msec  # milliseconds
        t = self.timer
        if t is not None:
            t.Start(self.redraw_interval)

    def make_context_current(self):
        # creates context if needed
        if self.timer is None:
            self.timer = wx.Timer(self)
            self.Bind(wx.EVT_TIMER, self._redraw_timer_callback, self.timer)
            self.timer.Start(self.redraw_interval)
        self.opengl_canvas.SetCurrent(self.opengl_context)

    def swap_buffers(self):
        self.opengl_canvas.SwapBuffers()

    def _redraw_timer_callback(self, event):
        # apparently we're in a thread, so use CallAfter since the
        # routines being called query wx widgets (mainly the
        # main graphics window) for attributes

        # 'x or y' is the "lambda way" of saying 'if not x: y'
        wx.CallAfter(lambda s=self:
                     s.session.update_loop.draw_new_frame(s.session)
                     or s.mouse_modes.mouse_pause_tracking())


class Popup(wx.PopupWindow):

    def __init__(self, parent, style = wx.BORDER_SIMPLE):
        wx.PopupWindow.__init__(self, parent, style)
        self._panel = wx.Panel(self)
#        self._panel.SetBackgroundColour((220,220,220))  # RGB 0-255
        self._pad = p = 2
        self._text = wx.StaticText(self._panel, -1, '', pos=(p,p))

    def show_text(self, text, position):

        import sys
        mac = (sys.platform == 'darwin')
        if mac:
            # On Mac balloons rise above other apps that cover the Chimear app
            # when mousing over those apps even when Chimera does not have focus.
            fw = wx.Window.FindFocus()
            if fw is None:
                return

        t = self._text
        t.SetLabel(text)
        sz = t.GetBestSize()
        p = 2*self._pad
        self.SetSize( (sz.width+p, sz.height+p) )
        self._panel.SetSize( (sz.width+p, sz.height+p) )
        offset = (0,0)
        xy = self.GetParent().ClientToScreen(position)
        self.Position(xy, offset)

        self.Show(True)

        if mac and fw:
            # Popup takes focus on Mac, restore it after showing popup.
            # fw.SetFocus() does not work on Mac.
            # Apparently wx cannot change the focus between top level windows.
            # But we can raise a different top level, and luckily this does
            # not raise it above the popup.
            fw.GetTopLevelParent().Raise()

    def hide(self):
        self.Show(False)


from wx import glcanvas


class OpenGLCanvas(glcanvas.GLCanvas):

    def __init__(self, parent, view, ui=None, size=None):
        self.view = view
        attribs = [glcanvas.WX_GL_RGBA, glcanvas.WX_GL_DOUBLEBUFFER]
        from ..core_settings import settings
        ppi = max(wx.GetDisplayPPI())
        if ppi < settings.multisample_threshold:
            # TODO: how to pick number of samples
            attribs += [glcanvas.WX_GL_SAMPLE_BUFFERS, 1,
                        glcanvas.WX_GL_SAMPLES, 4]
        attribs += [
            glcanvas.WX_GL_CORE_PROFILE,
            glcanvas.WX_GL_MAJOR_VERSION, 3,
            glcanvas.WX_GL_MINOR_VERSION, 3,
        ]
        gl_supported = glcanvas.GLCanvas.IsDisplaySupported
        if not gl_supported(attribs + [0]):
            raise AssertionError("Required OpenGL capabilities, RGBA and/or"
                " double buffering and/or OpenGL 3, not supported")
        for depth in range(32, 0, -8):
            test_attribs = attribs + [glcanvas.WX_GL_DEPTH_SIZE, depth]
            if gl_supported(test_attribs + [0]):
                attribs = test_attribs
                break
        else:
            raise AssertionError("Required OpenGL depth buffer capability"
                " not supported")
        if ui:
            ui.have_stereo = False
            if hasattr(ui, 'stereo') and ui.stereo:
                test_attribs = attribs + [glcanvas.WX_GL_STEREO]
                if gl_supported(test_attribs + [0]):
                    attribs = test_attribs
                    ui.have_stereo = True
        if ui:
            ui.opengl_attribs = attribs + [0]

        ckw = {} if size is None else {'size': size}
        glcanvas.GLCanvas.__init__(self, parent, -1, attribList=attribs + [0],
                                   style=wx.WANTS_CHARS, **ckw)

        self.SetBackgroundStyle(wx.BG_STYLE_PAINT)

        if ui:
            self.Bind(wx.EVT_CHAR, ui.forward_keystroke)
        self.Bind(wx.EVT_PAINT, self.on_paint)
        self.Bind(wx.EVT_SIZE, self.on_size)

    def on_paint(self, event):
        # self.SetCurrent(view.opengl_context())
        self.view.draw()

    def on_size(self, event):
        wx.CallAfter(self.set_viewport)
        event.Skip()

    def set_viewport(self):
        self.view.resize(*self.GetClientSize())


class OculusGraphicsWindow(wx.Frame):
    """
    The graphics window for using Oculus Rift goggles.
    """

    def __init__(self, view, parent=None):

        wx.Frame.__init__(self, parent, title="Oculus Rift")

        class View:

            def draw(self):
                pass

            def resize(self, *args):
                pass
        self.opengl_canvas = OpenGLCanvas(self, View())

        from wx.glcanvas import GLContext
        oc = self.opengl_context = GLContext(self.opengl_canvas, view._opengl_context)
        oc.make_current = self.make_context_current
        oc.swap_buffers = self.swap_buffers
        self.opengl_context = oc
        self.primary_opengl_context = view._opengl_context

        sizer = wx.BoxSizer(wx.HORIZONTAL)
        sizer.Add(self.opengl_canvas, 1, wx.EXPAND)
        self.SetSizerAndFit(sizer)

        self.Show(True)

    def make_context_current(self):
        self.opengl_canvas.SetCurrent(self.opengl_context)

    def swap_buffers(self):
        self.opengl_canvas.SwapBuffers()

    def close(self):
        self.opengl_context = None
        self.opengl_canvas = None
        wx.Frame.Close(self)

    def full_screen(self, width, height):
        ndisp = wx.Display.GetCount()
        for i in range(ndisp):
            d = wx.Display(i)
            # TODO: Would like to use d.GetName() but it is empty string on Mac.
            if not d.IsPrimary():
                g = d.GetGeometry()
                s = g.GetSize()
                if s.GetWidth() == width and s.GetHeight() == height:
                    self.Move(g.GetX(), g.GetY())
                    self.SetSize(width, height)
                    break
        # self.EnableFullScreenView(True) # Not available in wxpython
        # TODO: full screen always shows on primary display.
#        self.ShowFullScreen(True)
