# vim: set expandtab ts=4 sw=4:

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

from chimerax.mouse_modes import MouseMode
class MoveLabelMouseMode(MouseMode):
    '''Move 2D labels or arrows with mouse.'''
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
        self._arr_part = None
        if label is None:
            from .arrows import arrow_under_window_position
            self._label, self._arr_part = arrow_under_window_position(self.session, x, y)
            if self._label is None:
                from .label3d import picked_3d_label
                self._label = picked_3d_label(self.session, x, y)

    def mouse_drag(self, event):
        dx, dy = self.mouse_motion(event)
        lbl = self._label
        ses = self.session
        from .label3d import ObjectLabel
        from .label2d import Label
        from .arrows import Arrow
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
        elif isinstance(lbl, Arrow):
            w,h = ses.main_view.window_size
            x,y = self._get_arr_xy()
            xpos = x + dx/w
            ypos = y - dy/h
            from .arrows import arrow
            arrow(ses, [lbl], **{self._arr_part: (xpos, ypos)})

    def mouse_up(self, event):
        self._log_label_move_command()
        self._label = None
        MouseMode.mouse_up(self, event)

    def _log_label_move_command(self):
        lbl = self._label
        from .label2d import Label
        from .arrows import Arrow
        if isinstance(lbl, Label):
            command = '2dlabel #%s xpos %.3f ypos %.3f' % (lbl.drawing.id_string, lbl.xpos, lbl.ypos)
            from chimerax.core.commands import log_equivalent_command
            log_equivalent_command(self.session, command)
        elif isinstance(lbl, Arrow):
            x,y = self._get_arr_xy()
            command = '2dlabel arrow #%s %s %.3f,%.3f' % (lbl.drawing.id_string, self._arr_part, x, y)
            from chimerax.core.commands import log_equivalent_command
            log_equivalent_command(self.session, command)

    def _get_arr_xy(self):
        if self._arr_part == "start":
            x, y = self._label.start
        else:
            x, y = self._label.end
        return x,y
        
def register_mousemode(session):
    mm = session.ui.mouse_modes
    mm.add_mode(MoveLabelMouseMode(session))
