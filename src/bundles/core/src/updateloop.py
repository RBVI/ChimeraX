# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2022 Regents of the University of California. All rights reserved.
# This software is provided pursuant to the ChimeraX license agreement, which
# covers academic and commercial uses. For more information, see
# <http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
#
# This file is part of the ChimeraX library. You can also redistribute and/or
# modify it under the GNU Lesser General Public License version 2.1 as
# published by the Free Software Foundation. For more details, see
# <https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html>
#
# This file is distributed WITHOUT ANY WARRANTY; without even the implied
# warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. This notice
# must be embedded in or attached to all copies, including partial copies, of
# the software or any revisions or derivations thereof.
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
        self.redraw_interval = 16.667  # milliseconds, 60 frames per second, can be 0, only integer part used.

        self._minimum_event_processing_ratio = 0.1 # Event processing time as a fraction
                                                   # of time since start of last drawing
        self._last_timer_start_time = self._last_timer_finish_time = 0
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
        drew = False
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
                from chimerax.graphics import OpenGLError, OpenGLVersionError
                try:
                    if ((session.ui.is_gui and session.ui.main_window.graphics_window.is_drawable)
                        or getattr(view.camera, 'always_draw', False)):
                        t0 = time()
                        view.draw(check_for_changes = False)
                        self.last_draw_time = time() - t0
                        drew = True
                except OpenGLVersionError as e:
                    self.block_redraw()
                    session.logger.error(str(e))
                except OpenGLError as e:
                    self.block_redraw()
                    msg = 'An OpenGL graphics error occurred. Most often this is caused by a graphics driver bug. The only way to fix such bugs is to update your graphics driver. Redrawing graphics is now stopped to avoid a continuous stream of error messages. To restart graphics use the command "graphics restart" after changing the settings that caused the error.'
                    import traceback
                    session.logger.bug(msg + '\n\n' + str(e) + '\n\n' + traceback.format_exc())
                except Exception as e:
                    self.block_redraw()
                    msg = 'An error occurred in drawing the scene. Redrawing graphics is now stopped to avoid a continuous stream of error messages. To restart graphics use the command "graphics restart" after changing the settings that caused the error.'
                    import traceback
                    session.logger.bug(msg + '\n\n' + str(e) + '\n\n' + traceback.format_exc())
                session.triggers.activate_trigger('frame drawn', self)
        finally:
            self.unblock_redraw()

        view.frame_number += 1

        return drew

    def block_redraw(self):
        # Avoid redrawing when we are already in the middle of drawing.
        self._block_redraw_count += 1

    def unblock_redraw(self):
        self._block_redraw_count -= 1

    def blocked(self):
        return self._block_redraw_count > 0

    def set_redraw_interval(self, msec):
        '''
        A redraw interval of 0 is allowed and means redraw will
        occur as soon as all Qt events have been processed.
        '''
        self.redraw_interval = msec  # milliseconds
        t = self._timer
        if t is not None:
            t.start(int(self.redraw_interval))

    def start_redraw_timer(self):
        if self._timer is not None or not self.session.ui.is_gui:
            return
        from Qt.QtCore import QTimer, Qt
        self._timer = t = QTimer()
        t.timerType = Qt.TimerType.PreciseTimer
        t.timeout.connect(self._redraw_timer_callback)
        t.start(int(self.redraw_interval))

        # Stop the redraw timer when the app quits
        self.session.triggers.add_handler('app quit', self._app_quit)

    def _redraw_timer_callback(self):
        import time
        t = time.perf_counter()
        time_since_last_timer = t - self._last_timer_start_time
        after_timer_interval = t - self._last_timer_finish_time
        self._last_timer_start_time = t
        if  (self.redraw_interval == 0 or
             after_timer_interval >= self._minimum_event_processing_ratio * time_since_last_timer):
            # Redraw only if enough time has elapsed since last frame to process some events.
            # This keeps the user interface responsive even during slow rendering
            drew = self.draw_new_frame()
            self.session.ui.mouse_modes.mouse_pause_tracking()
        self._last_timer_finish_time = time.perf_counter()

    def _app_quit(self, tname, tdata):
        # Avoid errors caused by attempting to redraw after quiting.
        t = self._timer
        if t:
            t.stop()
            self._timer = None
        self.block_redraw()

    def update_graphics_now(self):
        '''
        Redraw graphics now if there are any changes.  This is typically only used by
        mouse drag code that wants to update the graphics as responsively as possible,
        particularly when a mouse step may take significant computation, such as contour
        surface level change.  After each mouse event this is called to force a redraw.
        '''
        self.draw_new_frame()
