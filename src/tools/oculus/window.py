# vim: set expandtab ts=4 sw=4:

from chimerax.core import window_sys
if window_sys == "wx":
    import wx

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
            from chimerax.core.ui import graphics 
            self.opengl_canvas = graphics.OpenGLCanvas(self, View())

            from wx.glcanvas import GLContext
            oc = self.opengl_context = GLContext(self.opengl_canvas, view.render._opengl_context)
            oc.make_current = self.make_context_current
            oc.swap_buffers = self.swap_buffers
            self.opengl_context = oc
            self.primary_opengl_context = view.render._opengl_context

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

elif window_sys == "qt":
    

    from PyQt5.QtGui import QWindow
    class OculusGraphicsWindow(QWindow):
        """
        The graphics window for using Oculus Rift goggles.
        """

        def __init__(self, view, parent=None):

            QWindow.__init__(self)

            from PyQt5.QtGui import QSurface
            self.setSurfaceType(QSurface.OpenGLSurface)

            class OculusOpenGLContext:
                def __init__(self, window, opengl_context):
                    self.window = window
                    self.opengl_context = opengl_context
                def make_current(self):
                    if not self.opengl_context.makeCurrent(self.window):
                        raise RuntimeError("Could not make graphics context current on oculus window")
                def swap_buffers(self):
                    self.opengl_context.swapBuffers(self.window)

            self.primary_opengl_context = poc = view.render._opengl_context
            self.opengl_context = OculusOpenGLContext(self, poc)
            self.show()

        def make_context_current(self):
            self.opengl_context.make_current()

        def swap_buffers(self):
            self.opengl_context.swap_buffers()

        def close(self):
            self.opengl_context = None
            self.destroy()	# Destroy QWindow

        def full_screen(self, width, height):
            from PyQt5.QtGui import QGuiApplication
            from PyQt5.QtCore import Qt
            screens = QGuiApplication.screens()
            os = [s for s in screens if s.name() == 'Rift DK2']
            if len(os) == 0:
                raise RuntimeError('Could not find Oculus screen, found screends "%s"'
                                   % ', '.join(s.name() for s in screens))
            else:
                self.setGeometry(os[0].geometry())
                # self.showFullScreen()	# Not necessary in Qt5.6
