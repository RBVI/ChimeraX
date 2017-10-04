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

# Mouse mode to place markers on surfaces
from chimerax.core.ui import MouseMode
class MarkerMouseMode(MouseMode):
    name = 'place marker'
    icon_file = 'marker.png'

    def __init__(self, session):

        MouseMode.__init__(self, session)

        self.mode_name = 'place markers'
        self.bound_button = None

    def enable(self):
        from .markergui import marker_panel
        p = marker_panel(self.session, 'Markers')
        p.update_settings()
        p.show()

    @property
    def placement_mode(self):
        return marker_settings(self.session, 'placement_mode')

    @property
    def link_consecutive(self):
        return marker_settings(self.session, 'link_consecutive')

    def mouse_down(self, event):
        if self.link_consecutive:
            if link_consecutive(self.session, event):
                return

        if self.placement_mode != 'link only':
            self.place_marker(event)

    def place_marker(self, event):
        x,y = event.position()
        s = self.session
        v = s.main_view
        p = v.first_intercept(x,y)
        if p is None:
            c = None
        elif self.placement_mode == 'surface center' and hasattr(p, 'triangle_pick'):
            c, vol = connected_center(p.triangle_pick)
            self.session.logger.info('Enclosed volume for marked surface: %.3g' % vol)
        else:
            c = p.position
        log = s.logger
        if c is None:
            log.status('No marker placed')
            return
        place_marker(self.session, c)

    def mouse_drag(self, event):
        pass

    def mouse_up(self, event):
        pass

class ConnectMouseMode(MarkerMouseMode):
    name = 'connect markers'
    icon_file = 'bond.png'

    def enable(self):
        s = marker_settings(self.session)
        s['link_consecutive'] = True
        s['placement_mode'] = 'link only'
        MarkerMouseMode.enable(self)

def link_consecutive(session, event):
    s = session
    from chimerax.core.atomic import selected_atoms
    atoms1 = selected_atoms(s)

    x,y = event.position()
    from chimerax.core.ui.mousemodes import picked_object, select_pick
    pick = picked_object(x, y, session.main_view)
    from chimerax.core.atomic import PickedAtom
    if not isinstance(pick, PickedAtom):
        return False
    a2 = pick.atom
    select_pick(session, pick, 'replace')

    if len(atoms1) != 1:
        return False
    a1 = atoms1[0]

    if a1.structure != a2.structure:
        s.logger.status('Cannot connect atoms from different molecules')
        return False
    if a1.connects_to(a2):
        return False
    
    m = a1.structure
    b = m.new_bond(a1,a2)
    b.radius = 0.5*min(a1.radius, a2.radius)
    b.color = (101,156,239,255)	# cornflowerblue
    b.halfbond = False
    s.logger.status('Made connection, distance %.3g' % b.length)
    return True

def marker_settings(session, attr = None):
    if not hasattr(session, '_marker_settings'):
        session._marker_settings = {
            'molecule': None,
            'next_marker_num': 1,
            'marker_chain_id': 'M',
            'color': (255,255,0,255),
            'radius': 1.0,
            'link_consecutive': False,  # Link consecutively clicked markers
            'placement_mode': 'surface'	# 'surface', 'surface center', 'link only'
        }
    s = session._marker_settings
    return s if attr is None else s[attr]

def marker_molecule(session):
    ms = marker_settings(session)
    m = ms['molecule']
    if m is None or m.was_deleted:
        from chimerax.core.atomic import Structure
        ms['molecule'] = m = Structure(session, name = 'markers', auto_style = False)
        m.ball_scale = 1.0
        session.models.add([m])
    return m

def place_marker(session, center):
    m = marker_molecule(session)
    a = m.new_atom('', 'H')
    a.coord = center
    ms = marker_settings(session)
    a.radius = ms['radius']
    a.color = ms['color']
    a.draw_mode = a.BALL_STYLE	# Sphere style hides bonds between markers, so use ball style.
    r = m.new_residue('mark', ms['marker_chain_id'], ms['next_marker_num'])
    r.add_atom(a)
    ms['next_marker_num'] += 1
    m.new_atoms()
    session.logger.status('Placed marker')

def connected_center(triangle_pick):
    d = triangle_pick.drawing()
    t = triangle_pick.triangle_number
    va, ta = d.vertices, d.triangles
    from chimerax.core import surface
    ti = surface.connected_triangles(ta, t)
    tc = ta[ti,:]
    varea = surface.vertex_areas(va, tc)
    a = varea.sum()
    c = varea.dot(va)/a
    cscene = d.scene_position * c
    vol, holes = surface.enclosed_volume(va, tc)
    return cscene, vol

def mark_map_center(volume):
    for s in volume.surface_drawings:
        va, ta = s.vertices, s.triangles
        from chimerax.core import surface
        varea = surface.vertex_areas(va, ta)
        a = varea.sum()
        c = varea.dot(va)/a
        place_marker(volume.session, c)
        
