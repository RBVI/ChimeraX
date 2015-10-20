# vi: set expandtab shiftwidth=4 softtabstop=4:
class UpdateLoop:

    def __init__(self):
        self._block_redraw_count = 0

    def draw_new_frame(self, session):
        '''
        Draw the scene if it has changed or camera or rendering options have changed.
        Before checking if the scene has changed fire the "new frame" trigger
        typically used by movie recording to animate such as rotating the view, fading
        models in and out, ....  If the scene is drawn fire the "rendered frame" trigger
        after drawing.  Return true if draw, otherwise false.
        '''
        if self._block_redraw_count > 0:
            # Avoid redrawing during callbacks of the current redraw.
            return False

        view = session.main_view
        with self.block_redraw():
            session.triggers.activate_trigger('new frame', self)
            from . import atomic
            atomic.check_for_changes(session)
            changed = view.check_for_drawing_change()
            if changed:
                try:
                    view.draw(check_for_changes = False)
                except:
                    # Stop redraw if an error occurs to avoid continuous stream of errors.
                    self.block_redraw()
                    import traceback
                    session.logger.error('Error in drawing scene. Redraw is now stopped.\n\n' +
                                        traceback.format_exc())
                session.triggers.activate_trigger('rendered frame', self)

        view.frame_number += 1

        return changed

    from contextlib import contextmanager
    @contextmanager
    def block_redraw(self):
        # Avoid redrawing when we are already in the middle of drawing.
        self._block_redraw_count += 1
        yield
        self._block_redraw_count -= 1
