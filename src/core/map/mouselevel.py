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

from ..ui import MouseMode
class ContourLevelMouseMode(MouseMode):
    name = 'contour level'
    icon_file = 'contour.png'

    def __init__(self, session):

        MouseMode.__init__(self, session)

    def mouse_drag(self, event):

        dx, dy = self.mouse_motion(event)
        f = -0.001*dy

        for m in mouse_maps(self.session.models):
            adjust_threshold_level(m, f)
            m.show()

        # Make sure new level is shown before another mouse event causes another level change.
        self.session.ui.update_graphics_now()
    
    def wheel(self, event):
        d = event.wheel_value()
        f = d/30
        for m in mouse_maps(self.session.models):
            adjust_threshold_level(m, f)
            m.show()

        # Make sure new level is shown before another mouse event causes another level change.
        self.session.ui.update_graphics_now()

def mouse_maps(models):    
    mall = models.list()
    from .volume import Volume
    mdisp = [m for m in mall if isinstance(m,Volume) and m.display]
    msel = [m for m in mdisp if m.any_part_selected()]
    maps = msel if msel else mdisp
    return maps

def adjust_threshold_level(m, f):
    ms = m.matrix_value_statistics()
    step = f * (ms.maximum - ms.minimum)
    if m.representation == 'solid':
        new_levels = [(l+step,b) for l,b in m.solid_levels]
        l,b = new_levels[-1]
        new_levels[-1] = (max(l,1.01*ms.maximum),b)
        m.set_parameters(solid_levels = new_levels)
    else:
        new_levels = tuple(l+step for l in m.surface_levels)
        m.set_parameters(surface_levels = new_levels)
