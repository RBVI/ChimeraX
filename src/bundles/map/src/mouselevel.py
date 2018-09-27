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

from chimerax.ui import MouseMode
class ContourLevelMouseMode(MouseMode):
    name = 'contour level'
    icon_file = 'contour.png'

    def __init__(self, session):
        MouseMode.__init__(self, session)
        self._maps = []
        
    def mouse_down(self, event):
        MouseMode.mouse_down(self, event)
        self._maps = self._picked_maps(event)

    def _picked_maps(self, event):
        x,y = event.position()
        v = self.session.main_view
        xyz1, xyz2 = v.clip_plane_points(x,y)    # scene coordinates
        return self._picked_maps_on_segment(xyz1, xyz2)

    def _picked_maps_on_segment(self, xyz1, xyz2):
        closest = None
        dist = None
        shown_maps = mouse_maps(self.session.models)
        for m in shown_maps:
            ppos = (m.scene_position * m.position.inverse()).inverse() # Map scene to parent coordinates
            mxyz1, mxyz2 =  ppos * xyz1, ppos * xyz2
            p = m.first_intercept(mxyz1, mxyz2)
            if p and (dist is None or p.distance < dist):
                if hasattr(p, 'triangle_pick'):
                    closest = (m, p.triangle_pick.drawing())	# Remember which surface
                else:
                    closest = m
        return shown_maps if closest is None else [closest]

    def mouse_drag(self, event):

        dx, dy = self.mouse_motion(event)
        f = -0.001*dy

        adjust_threshold_levels(self._maps, f)

        # Make sure new level is shown before another mouse event causes another level change.
        self.session.update_loop.update_graphics_now()
    
    def wheel(self, event):
        d = event.wheel_value()
        f = d/30
        maps = self._picked_maps(event)
        adjust_threshold_levels(maps, f)

        # Make sure new level is shown before another mouse event causes another level change.
        self.session.update_loop.update_graphics_now()

    def mouse_up(self, event):
        self.log_volume_command()
        self._maps = []
        MouseMode.mouse_up(self, event)
        
    def laser_click(self, xyz1, xyz2):
        self._maps = self._picked_maps_on_segment(xyz1, xyz2)

    def drag_3d(self, position, move, delta_z):
        if delta_z is not None:
            adjust_threshold_levels(self._maps, delta_z)
        else:
            self.log_volume_command()
            self._maps = []

    def log_volume_command(self):
        for v in self._maps:
            if isinstance(v, tuple):
                v = v[0]
            log_volume_level_command(v)

def mouse_maps(models):    
    mall = models.list()
    from .volume import Volume
    mdisp = [m for m in mall if isinstance(m,Volume) and m.display]
    msel = [m for m in mdisp if m.get_selected(include_children=True)]
    maps = msel if msel else mdisp
    return maps

def adjust_threshold_levels(maps, f):
    for m in maps:
        if isinstance(m, tuple):
            adjust_threshold_level(m[0], f, surf = m[1])
        else:
            adjust_threshold_level(m, f)
            
def adjust_threshold_level(m, f, surf=None):
    ms = m.matrix_value_statistics()
    step = f * (ms.maximum - ms.minimum)
    if m.representation == 'solid':
        new_levels = [(l+step,b) for l,b in m.solid_levels]
        l,b = new_levels[-1]
        new_levels[-1] = (max(l,1.01*ms.maximum),b)
        m.set_parameters(solid_levels = new_levels)
    else:
        if surf:
            new_levels = tuple((s.level+step if s is surf else s.level) for s in m.surfaces)
        else:
            new_levels = tuple(s.level+step for s in m.surfaces)
        m.set_parameters(surface_levels = new_levels, threaded_surface_calculation = True)

def log_volume_level_command(v):
    if v.representation == 'solid':
        levels = ' '.join('level %.4g,%.4g' % sl for sl in v.solid_levels)
    else:
        levels = ' '.join('level %.4g' % s.level for s in v.surfaces)
    command = 'volume #%s %s' % (v.id_string, levels)
    from chimerax.core.commands import log_equivalent_command
    log_equivalent_command(v.session, command)

            
