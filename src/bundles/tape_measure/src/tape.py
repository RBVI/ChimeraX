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
class TapeMeasureMouseMode(MouseMode):
    name = 'tape measure'
    icon_file = 'tape.png'

    def __init__(self, session):
        MouseMode.__init__(self, session)
        self._start_point = None
        self._marker_set = None
        self._markers = []
        self._color = (255,255,0,255)
        self._radius = .1	# scene units (usually Angstroms)
        self._vr_radius = .002	# meters, for use in VR
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
        if p is None:
            return
        self._show_distance(p)

    def _clear(self):
        mset = self._marker_set
        if mset and not mset.deleted:
            self._log_clear_command()
            self.session.models.close([mset])
        self._marker_set = None
            
    def _show_distance(self, end_point):
        adjust = (len(self._markers) == 2)
        if adjust:
            self._markers[1].scene_coord = end_point
        else:
            self._create_line(end_point)
        self._show_label()
        self._motion_command(adjust = adjust)

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
        text, h = self._label_text_and_height()
        from chimerax.core.colors import Color
        label(self.session, objects = b, object_type = 'bonds',
              text = text, height = h, color = Color(self._color))

    def _label_text_and_height(self):
        from chimerax.geometry import distance
        m1, m2 = self._markers
        d = distance(m1.coord, m2.coord)
        text = '%.4g' % d
        h = max(2*self._radius, 0.1*d)
        return text, h

    def mouse_up(self, event):
        MouseMode.mouse_up(self, event)
        if self._markers:
            self._log_tape_command()
            self._markers = []
        else:
            self._clear()

    def _log_tape_command(self):
        cmd = self._tape_command(adjust = self._logging_motion())
        from chimerax.core.commands import log_equivalent_command
        log_equivalent_command(self.session, cmd)

    def _tape_command(self, adjust = False):
        m1, m2 = self._markers
        label,h = self._label_text_and_height()
        mset = m1.structure
        p1 = '%.4g,%.4g,%.4g' % tuple(m1.scene_coord)
        p2 = '%.4g,%.4g,%.4g' % tuple(m2.scene_coord)
        from chimerax.core.colors import color_name
        cname = color_name(self._color)
        cmd = ('marker segment %s position %s toPosition %s color %s radius %.4g label %s labelHeight %.4g labelColor %s'
               % (mset.atomspec, p1, p2, cname, self._radius, label, h, cname))
        if adjust:
            n1,n2 = m1.residue.number, m2.residue.number
            cmd += ' adjust %s:%d,%d' % (mset.atomspec, n1, n2)
        return cmd
            
    def _motion_command(self, adjust = True):
        if self._logging_motion():
            cmd = self._tape_command(adjust = adjust)
            from chimerax.core.commands import motion_command
            motion_command(self.session, cmd)

    def _logging_motion(self):
        from chimerax.core.commands import motion_commands_enabled
        return motion_commands_enabled(self.session)
        
    def _log_clear_command(self):
        mset = self._marker_set
        cmd = 'marker delete %s' % mset.atomspec
        from chimerax.core.commands import log_equivalent_command
        log_equivalent_command(mset.session, cmd)

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
        if xyz1 is None or xyz2 is None:
            return p, v
        from chimerax.geometry import distance
        for method in [self._surface_or_atom_point,
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
                
    def _surface_or_atom_point(self, xyz1, xyz2):
        view = self.session.main_view
        p = view.picked_object_on_segment(xyz1, xyz2, exclude = self._exclude_markers_from_pick)
        from chimerax.graphics import PickedTriangle
        from chimerax.map.volume import PickedMap
        from chimerax.atomic import PickedAtom, PickedResidue
        sxyz = v = None
        if isinstance(p, PickedMap) and hasattr(p, 'triangle_pick'):
            sxyz = p.position
            v = p.map
        elif isinstance(p, PickedTriangle):
            sxyz = p.position
        elif isinstance(p, PickedAtom):
            sxyz = p.atom.scene_coord
        elif isinstance(p, PickedResidue):
            a = p.residue.principal_atom
            if a:
                sxyz = a.scene_coord
        return sxyz, v

    def _exclude_markers_from_pick(self, drawing):
        if drawing is self._marker_set:
            return 'all'
        return not drawing.pickable
            
    def _volume_maximum_point(self, xyz1, xyz2):
        from chimerax.map import Volume
        vlist = self.session.models.list(type = Volume)
        from chimerax.markers.mouse import first_volume_maxima
        sxyz, v = first_volume_maxima(xyz1, xyz2, vlist)
        return sxyz, v

    def _volume_plane_point(self, xyz1, xyz2):
        from chimerax.map import Volume
        vlist = [v for v in self.session.models.list(type = Volume)
                 if v.showing_one_plane or v.showing_image('orthoplanes') or v.showing_image('box faces')]
        from chimerax.markers.mouse import volume_plane_intercept
        sxyz, v = volume_plane_intercept(xyz1, xyz2, vlist)
        return sxyz, v
        
    def _closest_point(self, event):
        '''Project start point to view line through mouse position.'''
        xyz1, xyz2 = self._view_line(event)
        if xyz1 is None or xyz2 is None:
            return None
        p = self._start_point
        dx = xyz2 - xyz1
        from chimerax.geometry import inner_product
        f = inner_product(p - xyz1, dx) / inner_product(dx, dx)
        cp = xyz1 + f * dx
        return cp

    def _view_line(self, event):
        x,y = event.position()
        xyz1, xyz2 = self.session.main_view.clip_plane_points(x, y)
        return xyz1, xyz2
            
    def vr_press(self, event):
        # Virtual reality hand controller button press.
        xyz1, xyz2 = event.picking_segment()
        self._start_point = xyz1
        from time import time
        self._start_time = time()
        # Set radius to self._vr_radius (meters) based on current vr scene scaling
        s = self.session.main_view.camera.scene_scale  # scale factor from scene to room (meters)
        self._radius = self._vr_radius / s

    def vr_motion(self, event):
        self._show_distance(event.tip_position)

    def vr_release(self, event):
        # Virtual reality hand controller button release.
        from time import time
        end_time = time()
        if end_time - self._start_time < self._clear_time:
            self._clear()
        elif self._markers:
            self._log_tape_command()
        self._markers = []
