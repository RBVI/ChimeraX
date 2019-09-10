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
        self._label = label = label_under_window_position(self.session, x, y)
        if label is None:
            from .label3d import picked_3d_label
            self._label = picked_3d_label(self.session, x, y)

    def mouse_drag(self, event):
        dx, dy = self.mouse_motion(event)
        lbl = self._label
        ses = self.session
        from .label3d import ObjectLabel
        from .label2d import Label
        if isinstance(lbl, ObjectLabel):
            lmodel = lbl._label_model
            ps = ses.main_view.pixel_size(lmodel.scene_position * lbl.location())
            ox,oy,oz = lbl.offset
            lbl.offset = (ox + ps*dx, oy - ps*dy, oz)
            lmodel._positions_need_update = True
        elif isinstance(lbl, Label):
            w,h = ses.main_view.window_size
            xpos = lbl.xpos + dx/w
            ypos = lbl.ypos - dy/h
            from .label2d import label2d
            label2d(ses, [lbl], xpos = xpos, ypos = ypos)

    def mouse_up(self, event):
        self._log_label_move_command()
        self._label = None
        MouseMode.mouse_up(self, event)

    def _log_label_move_command(self):
        lbl = self._label
        from .label2d import Label
        if isinstance(lbl, Label):
            command = '2dlabel #%s xpos %.3f ypos %.3f' % (lbl.drawing.id_string, lbl.xpos, lbl.ypos)
            from chimerax.core.commands import log_equivalent_command
            log_equivalent_command(self.session, command)
        
def register_mousemode(session):
    mm = session.ui.mouse_modes
    mm.add_mode(MoveLabelMouseMode(session))
