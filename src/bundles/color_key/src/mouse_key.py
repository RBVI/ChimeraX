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

from chimerax.mouse_modes import MouseMode

class ColorKeyMouseMode(MouseMode):
    name = 'color key'

    def __init__(self, session):
        MouseMode.__init__(self, session)
        from chimerax.core.triggerset import TriggerSet
        self.triggers = TriggerSet()
        self.triggers.add_trigger('drag finished')

    def mouse_down(self, event):
        MouseMode.mouse_down(self, event)
        self.window_mouse_down = self._mouse_xy_to_window(self.mouse_down_position)
        self._start_or_grab_key()

    def mouse_drag(self, event):
        pos, size = self._key_position_size(event)
        from .model import get_model
        key = get_model(self.session)
        key.pos = pos
        if size is not None:
            key.size = size

    def mouse_up(self, event):
        pos, size = self._key_position_size(event)
        MouseMode.mouse_up(self, event)
        cmd = "key pos %g,%g" % pos
        if size is not None:
            cmd += " size %g,%g" % size
        from chimerax.core.commands import run
        run(self.session, cmd)
        self.triggers.activate_trigger('drag finished', None)

    def _key_position_size(self, event):
        from .model import get_model
        key = get_model(self.session)
        ex, ey = self._mouse_xy_to_window(event.position())
        dx, dy = [(epos - startpos) for epos, startpos in zip((ex, ey), self.window_mouse_down)]
        if self.grab:
            return (self.grab_key_pos[0] + dx, self.grab_key_pos[1] + dy), None
        if dx < 0:
            key_x = self.window_mouse_down[0] + dx
            size_x = -dx
        else:
            key_x = self.window_mouse_down[0]
            size_x = dx
        if dy < 0:
            key_y = self.window_mouse_down[1] + dy
            size_y = -dy
        else:
            key_y = self.window_mouse_down[1]
            size_y = dy
        return (key_x, key_y), (size_x, size_y)

    def _mouse_xy_to_window(self, xy):
        # Y axis inverted
        ws = self.session.main_view.window_size
        return (xy[0] / ws[0], 1 - (xy[1] / ws[1]))

    def _start_or_grab_key(self):
        from .model import get_model
        key_exists = get_model(self.session, create=False) is not None
        key = get_model(self.session)
        if key_exists:
            # check for grab; see if in middle thrid of long side...
            x1, y1 = key.pos
            x2, y2 = x1 + key.size[0], y1 + key.size[1]
            if abs(x2 - x1) < abs(y2 - y1):
                long_axis = 1
                ymin = min(y2, y1)
                ymax = max(y2, y1)
                b1 = (2*ymin + ymax) / 3
                b2 = (2*ymax + ymin) / 3
                o1 = min(x1, x2)
                o2 = max(x1, x2)
            else:
                long_axis = 0
                xmin = min(x2, x1)
                xmax = max(x2, x1)
                b1 = (2*xmin + xmax) / 3
                b2 = (2*xmax + xmin) / 3
                o1 = min(y1, y2)
                o2 = max(y1, y2)
            if b1 < self.window_mouse_down[long_axis] < b2 \
            and o1 < self.window_mouse_down[1-long_axis] < o2:
                # grab
                self.grab = True
                self.grab_key_pos = key.pos
                return
        self.grab = False
        key.pos = self.window_mouse_down
        key.size = [1 / ws for ws in self.session.main_view.window_size]
        self.session.logger.status("Grab middle of key to reposition", color="blue")

