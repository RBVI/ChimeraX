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

from chimerax.mouse_modes import MouseMode

class ColorKeyMouseMode(MouseMode):
    name = 'color key'
    #icon_file = 'bondrot.png'

    def __init__(self, session):
        MouseMode.__init__(self, session)

    def mouse_down(self, event):
        MouseMode.mouse_down(self, event)
        self.window_mouse_down = [e/w for e, w in zip(self.mouse_down_position,
            self.session.main_view.window_size)]
        self._start_or_grab_key()

    def mouse_drag(self, event):
        pos, size = self._key_position_size(event)
        from .model import get_model
        key = get_model()
        key.position = pos
        if size is not None:
            key.size = size

    def mouse_up(self, event):
        pos, size = self._key_position_size(event)
        MouseMode.mouse_up(self, event)
        cmd = "key pos %g,%g" % pos
        if size is not None:
            cmd += " size %g,%g" % size
        from chimera.core.commands import run
        run(self.session, cmd)

    def _key_position_size(self, event):
        dx, dy = [e/w for e, w in zip(self.mouse_motion(event), self.session.main_view.window_size)]
        if self.grab:
            return (self.window_mouse_down[0] + dx, self.window_mouse_down[1] + dy), None
        if dx == 0 or dy == 0:
            return
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

    def _start_or_grab_key(self):
        from .model import get_model
        key_exists = get_model(create=False) is not None
        key = get_model()
        if key_exists:
            # check for grab; see if in middle thrid of long side...
            p1, p2 = key.position
            x1, y1 = p1
            x2, x2 = p2
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
                return
        self.grab = False
        key.position = self.window_mouse_down
        key.size = (1,1)
        self.session.logger.status("Grab middle of key to reposition", color="blue")

