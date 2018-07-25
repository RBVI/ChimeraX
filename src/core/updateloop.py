# vim: set expandtab shiftwidth=4 softtabstop=4:

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

class UpdateLoop:

    def __init__(self, session):
        self.session = session
        self._block_redraw_count = 0
        self.last_draw_time = 0
        self.last_new_frame_time = 0
        self.last_atomic_check_for_changes_time = 0
        self.last_drawing_change_time = 0
        self.last_clip_time = 0

        # TODO: perhaps redraw interval should be 10 to reduce frame drops at 60 frames/sec
        self.redraw_interval = 16.667  # milliseconds, 60 frames per second

        self._minimum_event_processing_ratio = 0.1 # Event processing time as a fraction
                                                   # of time since start of last drawing
        self._last_redraw_start_time = self._last_redraw_finish_time = 0
        self._timer = None
        
    def draw_new_frame(self):
        '''
        Draw the scene if it has changed or camera or rendering options have changed.
        Before checking if the scene has changed fire the "new frame" trigger
        typically used by movie recording to animate such as rotating the view, fading
        models in and out, ....  If the scene is drawn fire the "frame drawn" trigger
        after drawing.  Return true if draw, otherwise false.
        '''
        if self._block_redraw_count > 0:
            # Avoid redrawing during callbacks of the current redraw.
            return False

        # TODO: Would be nice to somehow minimize all the ugly timing.
        # TODO: Maybe timing is slowing things down a little, if so make it optional.
        session = self.session
        view = session.main_view
        self.block_redraw()
        from time import time
        try:
            t0 = time()
            session.triggers.activate_trigger('new frame', self)
            self.last_new_frame_time = time() - t0
            from chimerax import atomic
            t0 = time()
            atomic.check_for_changes(session)
            self.last_atomic_check_for_changes_time = time() - t0
            from chimerax import surface
            t0 = time()
            surface.update_clip_caps(view)
            self.last_clip_time = time() - t0
            t0 = time()
            changed = view.check_for_drawing_change()
            self.last_drawing_change_time = time() - t0
            if changed:
                from .graphics import OpenGLError, OpenGLVersionError
                try:
                    if session.ui.is_gui and session.ui.main_window.graphics_window.is_drawable:
                        t0 = time()
                        view.draw(check_for_changes = False)
                        self.last_draw_time = time() - t0
                except OpenGLVersionError as e:
                    self.block_redraw()
                    session.logger.error(str(e))
                except OpenGLError as e:
                    self.block_redraw()
                    msg = 'An OpenGL graphics error occurred. Most often this is caused by a graphics driver bug. The only way to fix such bugs is to update your graphics driver. Redrawing graphics is now stopped to avoid a continuous stream of error messages. To restart graphics use the command "graphics restart" after changing the settings that caused the error.'
                    session.logger.error(msg + '\n\n' + str(e))
                except:
                    self.block_redraw()
                    msg = 'An error occurred in drawing the scene. Redrawing graphics is now stopped to avoid a continuous stream of error messages. To restart graphics use the command "graphics restart" after changing the settings that caused the error.'
                    import traceback
                    session.logger.error(msg + '\n\n' + traceback.format_exc())
                session.triggers.activate_trigger('frame drawn', self)
        finally:
            self.unblock_redraw()

        view.frame_number += 1

        return changed

    def block_redraw(self):
        # Avoid redrawing when we are already in the middle of drawing.
        self._block_redraw_count += 1

    def unblock_redraw(self):
        self._block_redraw_count -= 1

    def blocked(self):
        return self._block_redraw_count > 0
        
    def set_redraw_interval(self, msec):
        self.redraw_interval = msec  # milliseconds
        t = self._timer
        if t is not None:
            t.start(self.redraw_interval)

    def start_redraw_timer(self):
        if self._timer is not None:
            return
        from PyQt5.QtCore import QTimer, Qt
        self._timer = t = QTimer()
        t.timerType = Qt.PreciseTimer
        t.timeout.connect(self._redraw_timer_callback)
        t.start(self.redraw_interval)

    def _redraw_timer_callback(self):
        import time
        t = time.perf_counter()
        dur = t - self._last_redraw_start_time
        if t >= self._last_redraw_finish_time + self._minimum_event_processing_ratio * dur:
            # Redraw only if enough time has elapsed since last frame to process some events.
            # This keeps the user interface responsive even during slow rendering
            self._last_redraw_start_time = t
            s = self.session
            if not self.draw_new_frame():
                s.ui.mouse_modes.mouse_pause_tracking()
            self._last_redraw_finish_time = time.perf_counter()

    def update_graphics_now(self):
        '''
        Redraw graphics now if there are any changes.  This is typically only used by
        mouse drag code that wants to update the graphics as responsively as possible,
        particularly when a mouse step may take significant computation, such as contour
        surface level change.  After each mouse event this is called to force a redraw.
        '''
        self.draw_new_frame()
