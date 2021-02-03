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

#
# Status bar implementation is complex because Qt only will draw a QStatusBar if the
# event loop is run.  This causes arbitrary Qt events to be processed when a status message
# is shown which causes bugs in code that expects status message to have no side-effects.
# Qt offers no way to redraw a widget without event processing.
#
# We tried two approaches to solve this.  First we tried restricting event processing to
# non-user events (_StatusBarQt class below).  This didn't limit event processing much and still
# lead to errors.  The second more drastic approach (_StatusBarOpenGL class) which is currently
# used is to render the status line with OpenGL.
#
class _StatusBarOpenGL:
    def __init__(self, session):
        self.session = session
        self._opengl_context = None
        self._initialization_failed = False
        self._renderer = None
        self._window = None
        self._drawing = None
        self._drawing2 = None	# Secondary status
        self.background_color = (0.85,0.85,0.85,1.0)
        self.text_color = (0,0,0,255)
        self.font = 'Arial'
        self.pad_vert = 0.2 		# Fraction of status bar height
        self.pad_horz = 0.3 		# Fraction of status bar height (not width)
        self.widget = self._make_widget()
        self._last_message = ''
        self._last_color = 'black'

    def destroy(self):
        self.widget.destroy()
        self.widget = None
        self._window = None

        for attr in ('_drawing', '_drawing2', '_opengl_context', '_renderer'):
            v = getattr(self, attr)
            if v is not None:
                v.delete()
                setattr(self, attr, None)

    def show(self, show):
        w = self.widget
        if w:
            w.setVisible(show)
            
    def _make_widget(self):
        from Qt.QtWidgets import QStatusBar, QSizePolicy
        sb = QStatusBar()
        sb.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        w = StatusOpenGLWindow(parent = sb,
                               expose_cb = self._expose_event,
                               resize_cb = self._resize_event,
                               key_press_cb = self.session.ui.forward_keystroke)
        self._window = w
        return sb

    def _resize_event(self, event):
        r = self._renderer
        win = self._window
        if r and win:
            s = win.size()
            w,h = s.width(), s.height()
            r.set_default_framebuffer_size(w, h)

    def _expose_event(self, event):
        r = self._renderer
        if r is None:
            self._create_opengl_context()
        self.status(self._last_message, self._last_color)

    def _create_opengl_context(self):

        if self._initialization_failed:
            return False
        
        # Create opengl context
        w = self._window
        if w is None or _qwindow_deleted(w):
            return False  # Window has been closed
        if not w.isExposed():
            return False  # Window has not yet created.
        
        from chimerax.graphics import OpenGLContext, OpenGLVersionError, OpenGLError
        self._opengl_context = c = OpenGLContext(w, self.session.ui.primaryScreen())
        
        # Create texture drawing to render status messages
        from chimerax.graphics import Drawing, Render
        self._drawing = Drawing('statusbar')
        self._drawing2 = Drawing('secondary statusbar')
        self._renderer = r = Render(c)
        try:
            if not r.make_current():
                raise RuntimeError('Failed to make status line opengl context current')
        except (OpenGLError, OpenGLVersionError):
            self._opengl_context = None
            self._renderer = None
            self._initialization_failed = True
            self.session.logger.warning('No statusbar messages will be shown due to inadequate OpenGL')
            return False
        lw, lh = w.width(), w.height()
        r.initialize_opengl(lw, lh)
        r.set_background_color(self.background_color)
        return True

    # TODO: Handle expose events on status bar windows so resizes show label.
    #  Should probably handle status bar as one QWindow created by this class.
    #  Can put primary and secondary areas in same window.
    def status(self, msg, color = 'black', secondary = False):
        if not secondary:
            self._last_message = msg
            self._last_color = color
            
        if self._window is None or not self._window.isExposed():
            return # TODO: Need to show the status message when window is mapped.

        # Need to preserve OpenGL context across processing events, otherwise
        # a status message during the graphics draw, causes an OpenGL error because
        # Qt changed the current context.
        from chimerax.graphics import remember_current_opengl_context, restore_current_opengl_context
        cc = remember_current_opengl_context()

        if self._opengl_context is None:
            if not self._create_opengl_context():
                self.session.logger.info(msg)
                return	# OpenGL version is not sufficient

        r = self._renderer
        if not r.make_current():
            raise RuntimeError('Failed to make status line opengl context current')

        if len(msg) > 256:
            msg = msg[:253] + '...'

        r.update_viewport()	# Need this when window resized.
        r.draw_background()
        self._draw_text(msg, color, secondary)
        r.swap_buffers()

        restore_current_opengl_context(cc)

    def _draw_text(self, msg, color, secondary):
        self._update_texture(msg, color, secondary)
        dlist = self._drawings()
        if dlist:
            from chimerax.graphics.drawing import draw_overlays
            draw_overlays(dlist, self._renderer)

    def _drawings(self):
        return [d for d in [self._drawing, self._drawing2] if not getattr(d, 'cleared', False)]
    
    def _update_texture(self, msg, color, secondary):
        d = self._drawing2 if secondary else self._drawing
        d.cleared = (msg == '')
        if d.cleared:
            return

        lw, lh = self._renderer.render_size()
        aspect = lh/lw
        xpad,ypad = self.pad_horz, self.pad_vert

        from chimerax.core.colors import BuiltinColors
        tcolor = BuiltinColors[color].uint8x4() if color in BuiltinColors else self.text_color
        image_height = lh
        ixpad, iypad = max(1, int(xpad*lh)), max(1, int(ypad*lh))
        from chimerax.graphics import text_image_rgba
        rgba = text_image_rgba(msg, tcolor, image_height, self.font,
                               xpad=ixpad, ypad=iypad, pixels=True)
        th, tw = rgba.shape[:2]

        # Make image pixel exactly match screen pixel size for best appearance.
        uw, uh = 2*(tw/th)*(lh/lw), 2
        # Right align secondary status
        x = (1-uw) if secondary else -1
        y = -1

        from chimerax.graphics.drawing import rgba_drawing, draw_overlays
        rgba_drawing(d, rgba, (x, y), (uw, uh), opaque = False)

def _qwindow_deleted(w):
    from Qt import qt_object_is_deleted
    return qt_object_is_deleted(w)

from Qt.QtGui import QWindow
class StatusOpenGLWindow(QWindow):
    def __init__(self, parent, expose_cb = None, resize_cb = None, key_press_cb = None):
        QWindow.__init__(self)
        if expose_cb:
            self.exposeEvent = expose_cb
        if resize_cb:
            self.resizeEvent = resize_cb
        if key_press_cb:
            self.keyPressEvent = key_press_cb
        from Qt.QtWidgets import QWidget
        self._widget = QWidget.createWindowContainer(self, parent)
        from Qt.QtGui import QSurface
        self.setSurfaceType(QSurface.OpenGLSurface)
        parent.addWidget(self._widget, stretch = 1)

    # Override QWindow size(), width() and height() to use widget values.
    # In Qt 5.12.9 QWindow reports values that are half the correct size
    # after main window is dragged from devicePixelRatio = 2 screen
    # to a devicePixelRatio = 1 screen on Windows 10.
    def size(self):
        return self._widget.size()
    def width(self):
        return self._widget.width()
    def height(self):
        return self._widget.height()

#
# Status bar drawing that partially restricts Qt event processing.  Allows event related
# callbacks to be invoked during status message display which leads to hard to reproduce
# errors.  No longer using this.
#
class _StatusBarQt:
    def __init__(self, session):
        self.session = session
        self.widget = self._make_widget()

    def destroy(self):
        self.widget.destroy()
        self.widget = None
        
    def _make_widget(self):
        from Qt.QtWidgets import QStatusBar, QSizePolicy, QLabel
        sb = QStatusBar()
        sb.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        sb._primary_status_label = QLabel()
        sb._secondary_status_label = QLabel()
        sb.addWidget(sb._primary_status_label)
        sb.addPermanentWidget(sb._secondary_status_label)
        return sb

    def status(self, msg, color = 'black', secondary = False):
        sb = self.widget
        sb.clearMessage()
        if secondary:
            label = sb._secondary_status_label
        else:
            label = sb._primary_status_label
        label.setText("<font color='" + color + "'>" + msg + "</font>")
        label.show()

        self._show_status_now()

    def _show_status_now(self):
        # In Qt 5.7.1 there is no way to for the status line to redraw without running the event loop.
        # But we don't want requesting a status message to have any side effects, such as dispatching
        # mouse events.  This could cause havoc in the code writing the status message which does not
        # expect any side effects.

        # The only viable solution seems to be to process Qt events but exclude mouse and key events.
        # The unprocessed mouse/key events are kept and supposed to be processed later but due to
        # Qt bugs (57718 and 53126), those events don't get processed and mouse up events are lost
        # during mouse drags, causing the mouse to still drag controls even after the button is released.
        # This is seen in volume viewer when dragging the level bar on the histogram making the tool
        # very annoying to use. Some work-around code suggested in Qt bug 57718 of calling processEvents()
        # to send those deferred events is used below.

        if getattr(self, '_processing_deferred_events', False):
            return

        # Need to preserve OpenGL context across processing events, otherwise
        # a status message during the graphics draw, causes an OpenGL error because
        # Qt changed the current context.
        from chimerax.graphics import remember_current_opengl_context, restore_current_opengl_context
        cc = remember_current_opengl_context()

        s = self.session
        ul = s.update_loop
        ul.block_redraw()	# Prevent graphics redraw. Qt timers can fire.
        self._in_status_event_processing = True
        from Qt.QtCore import QEventLoop
        s.ui.processEvents(QEventLoop.ExcludeUserInputEvents)
        self._in_status_event_processing = False
        ul.unblock_redraw()

        restore_current_opengl_context(cc)
            
        self._process_deferred_events()

    def _process_deferred_events(self):
        # Handle bug where deferred mouse/key events are never processed on Mac Qt 5.7.1.
        from sys import platform
        if platform != 'darwin':
            return
        if getattr(self, '_flush_timer_queued', False):
            return

        def flush_pending_user_events(self=self):
            self._flush_timer_queued = False
            if getattr(self, '_in_status_event_processing', False):
                # Avoid processing deferred events if timer goes off during status message.
                self._process_deferred_events()
            else:
                self._processing_deferred_events = True
                self.session.ui.processEvents()
                self._processing_deferred_events = False

        self._flush_timer_queued = True
        from Qt.QtCore import QTimer
        QTimer.singleShot(0, flush_pending_user_events)

#_StatusBar = _StatusBarQt
_StatusBar = _StatusBarOpenGL
