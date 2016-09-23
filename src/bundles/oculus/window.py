# vim: set expandtab ts=4 sw=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2016 Regents of the University of California.
# All rights reserved.  This software provided pursuant to a
# license agreement containing restrictions on its disclosure,
# duplication and use.  For details see:
# http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
# This notice must be embedded in or attached to all copies,
# including partial copies, of the software or any revisions
# or derivations thereof.
# === UCSF ChimeraX Copyright ===

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
