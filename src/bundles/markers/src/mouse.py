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

# -----------------------------------------------------------------------------
# Mouse mode to place markers on surfaces
#
from chimerax.mouse_modes import MouseMode
class MarkerMouseMode(MouseMode):

    def __init__(self, session):

        MouseMode.__init__(self, session)

        self._moving_marker = None		# Atom
        self._resizing_marker_or_link = None	# Atom or Bond
        self._set_initial_sizes = True		# First marker on volume sets marker radius

    def enable(self):
        from .markergui import marker_panel
        p = marker_panel(self.session, 'Markers')
        p.update_settings()
        p.show()

    @property
    def link_new(self):
        return _mouse_marker_settings(self.session, 'link_new_markers')

    @property
    def marker_mode(self):
        return self.name	# Name of mouse mode
    
    def mouse_down(self, event):
        mode = self.marker_mode
        if mode == 'link markers':
            self.link_consecutive(event)
        elif mode == 'move markers':
            self.move_marker_begin(event)
        elif mode == 'resize markers':
            self.resize_begin(event)
        elif mode == 'delete markers':
            self.delete_marker_or_link(event)
        elif mode in ('mark surface', 'mark center'):
            self.place_on_surface(event)
        elif mode == 'mark maximum':
            self.place_on_maximum(event)
        elif mode == 'mark plane':
            self.place_on_plane(event)
        elif mode == 'mark point':
            self.place_on_point(event)

    def place_on_surface(self, event):
        xyz1, xyz2 = self._view_line(event)
        s = self.session
        log = s.logger
        p = s.main_view.picked_object_on_segment(xyz1, xyz2, max_transparent_layers = 0)
        if p is None:
            c = None
        elif self.marker_mode == 'mark center' and hasattr(p, 'triangle_pick'):
            c, vol = connected_center(p.triangle_pick)
            log.info('Enclosed volume for marked surface: %.3g' % vol)
        else:
            c = p.position

        if c is None:
            log.status('No marker placed')
            return
        d = p.drawing() if hasattr(p, 'drawing') else None
        _mouse_place_marker(s, c, link_to_selected = self.link_new, on_model = d)
            
    def place_on_maximum(self, event):
        from chimerax.map import Volume
        vlist = self.session.models.list(type = Volume)
        xyz1, xyz2 = self._view_line(event)
        sxyz, v = first_volume_maxima(xyz1, xyz2, vlist)
        if sxyz is not None:
            self._set_sizes(v)
            _mouse_place_marker(self.session, sxyz, link_to_selected = self.link_new, on_model = v)

    def _set_sizes(self, volume):
        if not self._set_initial_sizes:
            return
        self._set_initial_sizes = False
        ms = _mouse_marker_settings(self.session)
        r = max(volume.data.step)
        ms['marker radius'] = r
        ms['link radius'] = 0.5*r

    def place_on_plane(self, event):
        from chimerax.map import Volume
        vlist = self.session.models.list(type = Volume)
        xyz1, xyz2 = self._view_line(event)
        sxyz, v = volume_plane_intercept(xyz1, xyz2, vlist)
        if sxyz is not None:
            self._set_sizes(v)
            _mouse_place_marker(self.session, sxyz, link_to_selected = self.link_new, on_model = v)

    def place_on_point(self, event):
        if isinstance(event, LaserEvent):
            xyz = event.xyz1
        else:
            xyz1, xyz2 = self._view_line(event)
            xyz = .5 * (xyz1 + xyz2)
        _mouse_place_marker(self.session, xyz, link_to_selected = self.link_new)

    def link_consecutive(self, event):
        s = self.session
        from chimerax.atomic import selected_atoms
        atoms1 = selected_atoms(s)

        a2 = self.picked_marker(event, select = True)
        if a2 is None or len(atoms1) != 1:
            return False
        a1 = atoms1[0]

        if a1.structure != a2.structure:
            s.logger.status('Cannot connect markers from different marker sets')
            return False
        if a1 is a2 or a1.connects_to(a2):
            return False

        ms = _mouse_marker_settings(self.session)
        from .markers import create_link
        b = create_link(a1, a2, radius = ms['link radius'], rgba = ms['link color'], log = True)
        s.logger.status('Made connection, distance %.3g' % b.length)
        return True

    def picked_marker(self, event, select = False):
        m, l = self.picked_marker_or_link(event, select)
        return m
    
    def picked_marker_or_link(self, event, select = False):
        xyz1, xyz2 = self._view_line(event)
        view = self.session.main_view
        pick = view.picked_object_on_segment(xyz1, xyz2, exclude = _exclude_transparent_surfaces)
        m = l = None
        from chimerax.atomic import PickedAtom, PickedBond
        from .markers import MarkerSet
        if isinstance(pick, PickedAtom) and isinstance(pick.atom.structure, MarkerSet):
            m = pick.atom
        elif isinstance(pick, PickedBond) and isinstance(pick.bond.structure, MarkerSet):
            l = pick.bond
        if select and (m or l):
            self.session.selection.clear()
            if m:
                m.selected = True
            if l:
                l.selected = True
        return m, l

    def _view_line(self, event):
        if isinstance(event, LaserEvent):
            xyz1, xyz2 = event.xyz1, event.xyz2
        else:
            # Mouse event
            x,y = event.position()
            xyz1, xyz2 = self.session.main_view.clip_plane_points(x, y)
        return xyz1, xyz2

    def mouse_drag(self, event):
        self.move_marker(event)
        self.resize_marker_or_link(event)

    def move_marker_begin(self, event):
        self._moving_marker = self.picked_marker(event)
        if not isinstance(event, LaserEvent):
            MouseMode.mouse_down(self, event)

    def move_marker(self, event):        
        m = self._moving_marker
        if m is None:
            return
        sxyz = m.scene_coord
        dx, dy = self.mouse_motion(event)
        psize = self.pixel_size(sxyz)
        s = (dx*psize, -dy*psize, 0)	# Screen shift
        cpos = self.session.main_view.camera.position
        step = cpos.transform_vector(s)    # Scene coord system
        m.scene_coord = sxyz + step

    def resize_begin(self, event):
        m, l = self.picked_marker_or_link(event)
        self._resizing_marker_or_link = m or l
        if not isinstance(event, LaserEvent):
            MouseMode.mouse_down(self, event)

    def resize_marker_or_link(self, event):        
        m = self._resizing_marker_or_link
        if m is None:
            return
        dx, dy = self.mouse_motion(event)
        from math import exp
        self._resize_ml(m, exp(-0.01*dy))

    def _resize_ml(self, m, scale):
        r = m.radius * scale
        m.radius = r
        s = _mouse_marker_settings(self.session)
        from chimerax.atomic import Atom
        if isinstance(m, Atom):
            s['marker radius'] = r
        else:
            s['link radius'] = r

    def _update_marker_panel(self):
        from .markergui import marker_panel
        p = marker_panel(self.session, 'Markers')
        p.update_settings()
            
    def delete_marker_or_link(self, event):
        m, l = self.picked_marker_or_link(event)
        if m:
            _log_marker_delete(m)
            if m.structure.num_atoms == 1:
                # TODO: Leaving an empty structure causes errors
                self.session.models.close([m.structure])
            else:
                m.delete()
        elif l:
            _log_link_delete(l)
            l.delete()

    def mouse_up(self, event = None):
        mm = self._moving_marker
        if mm:
            _log_marker_move(mm)
            self._moving_marker = None

        rml = self._resizing_marker_or_link
        if rml:
            self._update_marker_panel()
            self._resizing_marker_or_link = None
            from chimerax.atomic import Atom
            if isinstance(rml, Atom):
                _log_marker_resize(rml)
            else:
                _log_link_resize(rml)

    def vr_press(self, event):
        # Virtual reality hand controller button press.
        xyz1, xyz2 = event.picking_segment()
        self.mouse_down(LaserEvent(xyz1,xyz2))
        
    def vr_motion(self, event):
        # Virtual reality hand controller motion.
        mm = self._moving_marker
        if mm:
            mm.scene_coord = event.motion * mm.scene_coord
        rm = self._resizing_marker_or_link
        if rm:
            from math import exp
            scale = exp(5*event.room_vertical_motion)
            self._resize_ml(rm, scale)

    def vr_release(self, event):
        # Virtual reality hand controller button release.
        self.mouse_up()

def _exclude_transparent_surfaces(drawing):
    if not drawing.pickable:
        return True
    from chimerax.core.models import Surface
    if isinstance(drawing, Surface):
        if drawing.display_style != drawing.Solid:
            return True
        any_opaque, any_transparent = drawing._transparency()
        if any_transparent:
            return True
    return False

# -----------------------------------------------------------------------------
#
class MarkMaximumMouseMode(MarkerMouseMode):
    name = 'mark maximum'
    icon_file = 'icons/maximum.png'
    description = 'Place marker at density maximum'
    
class MarkPlaneMouseMode(MarkerMouseMode):
    name = 'mark plane'
    icon_file = 'icons/plane.png'
    description = 'Place marker on volume plane'
    
class MarkSurfaceMouseMode(MarkerMouseMode):
    name = 'mark surface'
    icon_file = 'icons/surface.png'
    description = 'Place marker on surface'
    
class MarkCenterMouseMode(MarkerMouseMode):
    name = 'mark center'
    icon_file = 'icons/center.png'
    description = 'Place marker at center of connected surface'
    
class MarkPointMouseMode(MarkerMouseMode):
    name = 'mark point'
    icon_file = 'icons/point.png'
    description = 'Place marker at 3d pointer position'
    
class LinkMarkersPointMouseMode(MarkerMouseMode):
    name = 'link markers'
    icon_file = 'icons/link.png'
    description = 'Link consecutively clicked markers'
    
class MoveMarkersPointMouseMode(MarkerMouseMode):
    name = 'move markers'
    icon_file = 'icons/move.png'
    description = 'Move markers'
    
class ResizeMarkersPointMouseMode(MarkerMouseMode):
    name = 'resize markers'
    icon_file = 'icons/resize.png'
    description = 'Resize markers or links'
    
class DeleteMarkersPointMouseMode(MarkerMouseMode):
    name = 'delete markers'
    icon_file = 'icons/delete.png'
    description = 'Delete markers or links'
    
# -----------------------------------------------------------------------------
#
class LaserEvent:
    '''Handle 3d VR hand controller clicks.'''
    def __init__(self, xyz1, xyz2):
        self.xyz1 = xyz1
        self.xyz2 = xyz2

# -----------------------------------------------------------------------------
#
def connected_center(triangle_pick):
    d = triangle_pick.drawing()
    t = triangle_pick.triangle_number
    va, ta = d.vertices, d.triangles
    from chimerax import surface
    ti = surface.connected_triangles(ta, t)
    tc = ta[ti,:]
    varea = surface.vertex_areas(va, tc)
    a = varea.sum()
    c = varea.dot(va)/a
    cscene = d.scene_position * c
    vol, holes = surface.enclosed_volume(va, tc)
    return cscene, vol
        
# -----------------------------------------------------------------------------
#
def mark_map_center(volume):
    for s in volume.surfaces:
        va, ta = s.vertices, s.triangles
        from chimerax import surface
        varea = surface.vertex_areas(va, ta)
        a = varea.sum()
        c = varea.dot(va)/a
        cscene = v.scene_position * c
        _mouse_place_marker(volume.session, cscene, on_model = volume)
        
# -----------------------------------------------------------------------------
#
def first_volume_maxima(xyz_in, xyz_out, vlist):

    line = (xyz_in, xyz_out)	# Scene coords
    hits = []
    from chimerax.geometry import distance
    for v in vlist:
        if not v.shown():
            continue
        v_xyz_in, v_xyz_out = data_slice(v, line)
        if v_xyz_in is None:
            continue
        threshold = v.minimum_surface_level
        if threshold is None:
            if len(v.image_levels) == 0:
                return None, None
            threshold = min(lev for lev,h in v.image_levels)
        f = first_maximum_along_ray(v, v_xyz_in, v_xyz_out, threshold)
        if f is None:
            continue
        vxyz = (1-f)*v_xyz_in + f*v_xyz_out
        sxyz = v.scene_position * vxyz
        d = distance(sxyz, xyz_in)
        hits.append((d,sxyz,v))

    if len(hits) == 0:
        return None, None
    
    d,sxyz,v = min(hits, key=lambda h: h[0])
    return sxyz, v

# -----------------------------------------------------------------------------
#
def volume_plane_intercept(xyz_in, xyz_out, vlist):

    line = (xyz_in, xyz_out) # Scene coords
    hits = []
    from chimerax.geometry import distance
    for v in vlist:
        if not v.shown():
            continue
        plane = (v.single_plane() or
                 v.showing_image('orthoplanes') or
                 v.showing_image('box faces') or
                 v.showing_image('tilted slab'))
        if not plane:
            continue
        v_xyz_in, v_xyz_out = data_slice(v, line)
        if v_xyz_in is None:
            continue
        vxyz = .5 * v_xyz_in + .5 * v_xyz_out
        sxyz = v.scene_position * vxyz
        d = distance(sxyz, xyz_in)
        hits.append((d,sxyz,v))

    if len(hits) == 0:
        return None, None
    
    d,sxyz,v = min(hits, key=lambda h: h[0])
    return sxyz, v

# -----------------------------------------------------------------------------
#
def data_slice(v, line):

  if not v.shown():
    return None, None

  from chimerax.map import slice, tiltedslab
  if v.showing_image('orthoplanes') or v.showing_image('box faces'):
    xyz_in = xyz_out = slice.face_intercept_point(v, line)
  else:
    xyz_in, xyz_out = slice.volume_segment(v, line)
    if v.showing_image('tilted slab') and xyz_in is not None:
        xyz_in, xyz_out = tiltedslab.slab_segment(v, (xyz_in, xyz_out))

  return xyz_in, xyz_out


# -----------------------------------------------------------------------------
# Graph the data values along a line passing through a volume on a Tkinter
# canvas.  The volume data may have multiple components.  Each component is
# graphed using a Trace object.
#
def first_maximum_along_ray(volume, xyz_in, xyz_out, threshold):

  from chimerax.map.slice import slice_data_values
  trace = slice_data_values(volume, xyz_in, xyz_out)
  trace = tuple(trace)
  n = len(trace)
  for k in range(n):
    t, v = trace[k]
    if v >= threshold:
      if ((k-1 < 0 or trace[k-1][1] < v) and
          (k+1 >= n or trace[k+1][1] <= v)):
        return t

  return None
    
def _mouse_marker_settings(session, attr = None):
    if not hasattr(session, '_marker_settings'):
        session._marker_settings = {
            'marker set': None,
            'next_marker_num': None,
            'marker_chain_id': 'M',
            'marker color': (255,255,0,255),	# yellow
            'marker radius': 1.0,
            'link color': (101,156,239,255),	# cornflowerblue
            'link radius': 0.5,
            'link_new_markers': False,
        }
    s = session._marker_settings
    return s if attr is None else s[attr]

def _mouse_markerset(session):
    ms = _mouse_marker_settings(session)
    m = ms['marker set']
    if m is None or m.was_deleted:
        from .markers import MarkerSet
        mlist = [m for m in session.models.list(type = MarkerSet)]
        if mlist:
            m = mlist[0]
        else:
            m = MarkerSet(session, 'markers')
            session.models.add([m])
        ms['marker set'] = m
    return m
    
def _mouse_place_marker(session, center, link_to_selected = False,
                        select = True, log = True, on_model = None):
    '''Center is in scene coordinates.'''
    m = _mouse_markerset(session)
    ms = _mouse_marker_settings(session)
    mcenter = m.scene_position.inverse() * center
    marker_num = ms['next_marker_num']
    a = m.create_marker(mcenter, ms['marker color'], ms['marker radius'], marker_num)
    if on_model:
        _set_marker_frame_number(a, on_model)
    if log:
        _log_place_marker(m, center, ms['marker color'], ms['marker radius'])
    ms['next_marker_num'] = (a.residue.number+1) if marker_num is None else (marker_num+1)
    session.logger.status('Placed marker')
    if link_to_selected:
        from chimerax.atomic import selected_atoms
        atoms = selected_atoms(session)
        if len(atoms) == 1:
            al = atoms[0]
            if a.structure == al.structure and a is not al:
                from .markers import create_link
                create_link(al, a, radius = ms['link radius'], rgba = ms['link color'], log=log)
    if select:
        session.selection.clear()
        a.selected = True

def _set_marker_frame_number(marker, model):
    from chimerax.map import Volume
    if not isinstance(model, Volume):
        return
    series = getattr(model, 'series', None)
    if series is None:
        return
    from chimerax.map_series import MapSeries
    if not isinstance(series, MapSeries):
        return
    marker.frame = series.maps.index(model)
    marker.structure.save_marker_attribute_in_sessions('frame', int)
    
def _log_place_marker(mset, center, color, radius):
    c = '%.4g,%.4g,%.4g' % tuple(center)
    from chimerax.core.colors import color_name
    cmd = 'marker %s position %s color %s radius %.4g' % (mset.atomspec, c, color_name(color), radius)
    from chimerax.core.commands import log_equivalent_command
    log_equivalent_command(mset.session, cmd)

def _log_marker_delete(m):
    mset = m.structure
    cmd = 'marker delete %s:%d' % (mset.atomspec, m.residue.number)
    from chimerax.core.commands import log_equivalent_command
    log_equivalent_command(mset.session, cmd)

def _log_link_delete(l):
    mset = l.structure
    m1, m2 = l.atoms
    cmd = 'marker delete %s:%d,%d linksOnly true' % (mset.atomspec, m1.residue.number, m2.residue.number)
    from chimerax.core.commands import log_equivalent_command
    log_equivalent_command(mset.session, cmd)

def _log_marker_move(m):
    mset = m.structure
    pos = '%.4g,%.4g,%.4g' % tuple(m.scene_coord)
    cmd = 'marker change %s:%d position %s' % (mset.atomspec, m.residue.number, pos)
    from chimerax.core.commands import log_equivalent_command
    log_equivalent_command(mset.session, cmd)
    
def _log_marker_resize(m):
    mset = m.structure
    cmd = 'marker change %s:%d radius %.4g' % (mset.atomspec, m.residue.number, m.radius)
    from chimerax.core.commands import log_equivalent_command
    log_equivalent_command(mset.session, cmd)

def _log_link_resize(l):
    mset = l.structure
    m1, m2 = l.atoms
    cmd = ('marker change %s:%d,%d radius %.4g markers false'
           % (mset.atomspec, m1.residue.number, m2.residue.number, l.radius))
    from chimerax.core.commands import log_equivalent_command
    log_equivalent_command(mset.session, cmd)
    
# -----------------------------------------------------------------------------
#
def register_mousemode(session):
    marker_modes = [MarkMaximumMouseMode,
                    MarkPlaneMouseMode,
                    MarkSurfaceMouseMode,
                    MarkCenterMouseMode,
                    MarkPointMouseMode,
                    LinkMarkersPointMouseMode,
                    MoveMarkersPointMouseMode,
                    ResizeMarkersPointMouseMode,
                    DeleteMarkersPointMouseMode]
    mm = session.ui.mouse_modes
    for marker_class in marker_modes:
        mm.add_mode(marker_class(session))
