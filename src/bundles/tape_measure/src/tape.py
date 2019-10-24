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
class TapeMeasureMouseMode(MouseMode):
    name = 'tape measure'
    icon_file = 'tape.png'

    def __init__(self, session):
        MouseMode.__init__(self, session)
        self._start_point = None
        self._marker_set = None
        self._markers = []
        self._color = (255,255,0,255)
        self._radius = .2
        self._min_move = 5	# minimum pixels to draw tape
        self._start_time = 0
        self._clear_time = 0.3	# seconds. Fast click/release causes clear.
        
    def mouse_down(self, event):
        MouseMode.mouse_down(self, event)
        p, v = self._picked_point(event)
        self._start_point = p
        if v:
            self._radius = 0.25 * min(v.data.step)
    
    def mouse_drag(self, event):
        if self._start_point is None:
            return
        if not self._minimum_move(event):
            return
        p, v = self._picked_point(event)
        if p is None:
            p = self._closest_point(event)
        self._show_distance(p)

    def _clear(self):
        mset = self._marker_set
        if mset:
            self._marker_set = None
            self.session.models.close([mset])
            
    def _show_distance(self, end_point):
        if self._markers:
            self._markers[1].scene_coord = end_point
        else:
            self._create_line(end_point)
        self._show_label()

    def _create_line(self, end_point):
        mset = self._marker_set
        if mset is None or mset.deleted:
            from chimerax.markers import MarkerSet
            self._marker_set = mset = MarkerSet(self.session, 'tape measure')
            self.session.models.add([mset])
        rgba = self._color
        r = self._radius
        # Create end-point markers
        m1 = mset.create_marker(self._start_point, rgba, r)
        m2 = mset.create_marker(end_point, rgba, r)
        self._markers = (m1, m2)
        # Create line between markers
        from chimerax.markers import create_link
        self._link = create_link(m1, m2, rgba, r)

    def _show_label(self):
        # Add distance label
        from chimerax.label.label3d import label
        from chimerax.core.objects import Objects
        from chimerax.atomic import Bonds
        b = Objects(bonds = Bonds([self._link]))
        from chimerax.core.geometry import distance
        m1, m2 = self._markers
        d = distance(m1.coord, m2.coord)
        h = max(2*self._radius, 0.1*d)
        from chimerax.core.colors import Color
        label(self.session, objects = b, object_type = 'bonds',
              text = '%.4g' % d, height = h, color = Color(self._color))

    def mouse_up(self, event):
        MouseMode.mouse_up(self, event)
        if self._markers:
            self._markers = []
        else:
            self._clear()

    def _minimum_move(self, event):
        if self._markers:
            return True
        x0,y0 = self.mouse_down_position
        x1,y1 = event.position()
        dx, dy = x1-x0, y1-y0
        d = self._min_move
        return dx*dx + dy*dy >= d*d
        
    def _picked_point(self, event):
        xyz1, xyz2 = self._view_line(event)
        p = d = v = None
        from chimerax.core.geometry import distance
        for method in [self._surface_point,
                       self._volume_maximum_point,
                       self._volume_plane_point]:
            pm, vm = method(xyz1, xyz2)
            if pm is not None:
                dm = distance(xyz1,pm)
                if d is None or dm < d:
                    p = pm
                    d = dm
                    v = vm
        return p, v
                
    def _surface_point(self, xyz1, xyz2):
        from chimerax.mouse_modes import picked_object_on_segment
        p = picked_object_on_segment(xyz1, xyz2, self.session.main_view,
                                     exclude = self._exclude_markers_from_surface_pick)
        from chimerax.core.graphics import PickedTriangle
        from chimerax.map.volume import PickedMap
        if isinstance(p, PickedMap) and hasattr(p, 'triangle_pick'):
            sxyz = p.position
            v = p.map
        elif isinstance(p, PickedTriangle):
            sxyz = p.position
            v = None
        else:
            sxyz = v = None
        return sxyz, v

    def _exclude_markers_from_surface_pick(self, drawing):
        from chimerax.mouse_modes import unpickable
        return unpickable(drawing) or drawing is self._marker_set
            
    def _volume_maximum_point(self, xyz1, xyz2):
        from chimerax.map import Volume
        vlist = self.session.models.list(type = Volume)
        from chimerax.markers.mouse import first_volume_maxima
        sxyz, v = first_volume_maxima(xyz1, xyz2, vlist)
        return sxyz, v

    def _volume_plane_point(self, xyz1, xyz2):
        from chimerax.map import Volume
        vlist = [v for v in self.session.models.list(type = Volume) if v.showing_one_plane]
        from chimerax.markers.mouse import volume_plane_intercept
        sxyz, v = volume_plane_intercept(xyz1, xyz2, vlist)
        return sxyz, v
        
    def _closest_point(self, event):
        '''Project start point to view line through mouse position.'''
        xyz1, xyz2 = self._view_line(event)
        p = self._start_point
        dx = xyz2 - xyz1
        from chimerax.core.geometry import inner_product
        f = inner_product(p - xyz1, dx) / inner_product(dx, dx)
        cp = xyz1 + f * dx
        return cp

    def _view_line(self, event):
        x,y = event.position()
        xyz1, xyz2 = self.session.main_view.clip_plane_points(x, y)
        return xyz1, xyz2
            
    def vr_press(self, xyz1, xyz2):
        # Virtual reality hand controller button press.
        self._start_point = xyz1
        from time import time
        self._start_time = time()

    def vr_motion(self, position, move, delta_z):
        self._show_distance(position.origin())

    def vr_release(self):
        # Virtual reality hand controller button release.
        self._markers = []
        from time import time
        end_time = time()
        if end_time - self._start_time < self._clear_time:
            self._clear()
