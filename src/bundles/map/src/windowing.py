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
# Adjust gray-levels in medical images by translating brightness/transparency curve
# or scaling the width of the curve about its center.
#
from chimerax.mouse_modes import MouseMode
class WindowingMouseMode(MouseMode):
    name = 'windowing'
    icon_file = 'windowing.png'

    def __init__(self, session):
        MouseMode.__init__(self, session)
        self._maps = []
        
    def mouse_down(self, event):
        MouseMode.mouse_down(self, event)
        self._maps = self._visible_maps()

    def _visible_maps(self):
        from . import Volume
        return [m for m in self.session.models.list(type = Volume)
                if m.visible and m.representation == 'solid']
        
    def mouse_drag(self, event):

        dx, dy = self.mouse_motion(event)
        if abs(dx) > abs(dy):
            translate_levels(self._maps, 0.001 * dx)
        else:
            scale_levels(self._maps, -0.002 * dy)

        # Make sure new level is shown before another mouse event causes another level change.
        self.session.update_loop.update_graphics_now()
    
    def wheel(self, event):
        d = event.wheel_value()
        f = d/30
        maps = self._visible_maps()
        scale_levels(maps, f)

        # Make sure new level is shown before another mouse event causes another level change.
        self.session.update_loop.update_graphics_now()

    def mouse_up(self, event):
        self.log_volume_command()
        self._maps = []
        MouseMode.mouse_up(self, event)
        
    def laser_click(self, xyz1, xyz2):
        self._maps = self._visible_maps()

    def drag_3d(self, position, move, delta_z):
        if delta_z is not None:
            xshift,yshift,zshift = move.translation()	# Hand controller axes
            if abs(xshift) > abs(yshift):
                translate_levels(self._maps, xshift)
            else:
                scale_levels(self._maps, yshift)
        else:
            self.log_volume_command()
            self._maps = []

    def log_volume_command(self):
        for v in self._maps:
            if isinstance(v, tuple):
                v = v[0]
            log_volume_levels_command(v)


def translate_levels(maps, f):
    for v in maps:
        ms = v.matrix_value_statistics()
        vrange = (ms.maximum - ms.minimum)
        shift = f*vrange
        levels = [(lev+shift,y) for lev,y in v.solid_levels]
        v.set_parameters(solid_levels = levels)

def scale_levels(maps, f):
    from numpy import mean
    for v in maps:
        center = mean([lev for lev,y in v.solid_levels])
        levels = [(lev+f*(lev-center),y) for lev,y in v.solid_levels]
        v.set_parameters(solid_levels = levels)

def log_volume_levels_command(v):
    if v.representation == 'solid':
        levels = ' '.join('level %.4g,%.4g' % sl for sl in v.solid_levels)
        command = 'volume #%s %s' % (v.id_string, levels)
        from chimerax.core.commands import log_equivalent_command
        log_equivalent_command(v.session, command)

def register_mousemode(session):
    mm = session.ui.mouse_modes
    mm.add_mode(WindowingMouseMode(session))
