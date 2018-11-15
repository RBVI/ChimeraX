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

from chimerax.mouse_modes import MouseMode
class MoveLabelMouseMode(MouseMode):
    '''Move 2D labels with mouse.'''
    name = 'move label'
    icon_file = 'movelabel.png'

    def __init__(self, session):
        MouseMode.__init__(self, session)
        self._label = None

    def mouse_down(self, event):
        MouseMode.mouse_down(self, event)
        x,y = event.position()
        from .label2d import label_under_window_position
        self._label = label_under_window_position(self.session, x, y)

    def mouse_drag(self, event):
        dx, dy = self.mouse_motion(event)
        lbl = self._label
        if lbl:
            ses = self.session
            w,h = ses.main_view.window_size
            xpos = lbl.xpos + dx/w
            ypos = lbl.ypos - dy/h
            from .label2d import label_change
            label_change(ses, lbl.name, xpos = xpos, ypos = ypos)

    def mouse_up(self, event):
        self._log_label_move_command()
        self._label = None
        MouseMode.mouse_up(self, event)

    def _log_label_move_command(self):
        lbl = self._label
        if lbl:
            command = '2dlabel change %s xpos %.3f ypos %.3f' % (lbl.name, lbl.xpos, lbl.ypos)
            from chimerax.core.commands import log_equivalent_command
            log_equivalent_command(self.session, command)
        
def register_mousemode(session):
    mm = session.ui.mouse_modes
    mm.add_mode(MoveLabelMouseMode(session))
