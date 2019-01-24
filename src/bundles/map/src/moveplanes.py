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

# Mouse mode to move map planes or move faces of bounding box.
from chimerax.mouse_modes import MouseMode
class RegionMouseMode(MouseMode):

    def __init__(self, session):

        MouseMode.__init__(self, session)

        self.bound_button = None
        
        self.map = None
        self.matching_maps = []	# Adjust region for these maps too.
        self.ijk = None         # Clicked grid point.
        self.axis = None        # Clicked face normal axis
        self.side = None        # 0 or 1 for min/max box face along axis
        self.xy_last = None
        self.frac_istep = 0
        
    def mouse_down(self, event):
        self.xy_last = (x,y) = event.position()
        v = self.session.main_view
        line = v.clip_plane_points(x,y)    # scene coordinates
        self._choose_box_face(line)

    def _choose_box_face(self, line):
        from .volume import Volume
        maps = [m for m in self.session.models.list() if isinstance(m, Volume) and m.shown()]
        from .slice import nearest_volume_face
        v, self.axis, self.side, self.ijk = nearest_volume_face(line, maps)
        self.map = v
        if v:
            v.set_parameters(show_outline_box = True)
            self.matching_maps = matching_maps(v, maps)
            if (not self.move_faces and v.is_full_region(any_step = True)
                and not v.showing_orthoplanes() and not v.showing_box_faces()):
                self._show_single_plane(v, self.axis)

    def _show_single_plane(self, v, axis):
        ijk_min, ijk_max, ijk_step = [list(b) for b in v.region]
        p = (ijk_min[axis] + ijk_max[axis])//2
        ijk_min[axis] = p
        ijk_max[axis] = p + ijk_step[axis] - 1
        v.new_region(ijk_min, ijk_max)
        print ('show single', ijk_min, ijk_max, axis)
        v.set_representation('solid')

    def mouse_drag(self, event):
        v = self.map
        if v is None or self.xy_last is None:
            self.mouse_down(event)
            return

        xl, yl = self.xy_last
        x,y = event.position()
        dx, dy = (x - xl, yl - y)
        if dx == 0 and dy == 0:
            return
        speed = 0.1 if event.shift_down() else 1
        view = self.session.main_view
        step = speed * drag_distance(v, self.ijk, self.axis, dx, dy, view)
        sa = v.data.step[self.axis]
        istep = step / sa      # grid units
        self.xy_last = (x,y)
        self._move_plane(istep)

    def _move_plane(self, istep):
        # Remember fractional grid step for next move.
        istep += self.frac_istep
        rstep = int(round(istep))
        if rstep == 0:
            self.frac_istep = istep
            return
        self.frac_istep = istep - rstep
        v = self.map
        if self.move_faces:
            move_face(v, self.axis, self.side, rstep)
        else:
            move_slab(v, self.axis, self.side, rstep)
        for m in self.matching_maps:
            m.new_region(*tuple(v.region), adjust_step = False, adjust_voxel_limit = False)
            if v.showing_orthoplanes() and m.showing_orthoplanes():
                m.set_parameters(orthoplane_positions = v.rendering_options.orthoplane_positions)
        # Make sure new plane is shown before another mouse event shows another plane.
        self.session.update_loop.update_graphics_now()

    def wheel(self, event):
        self.mouse_down(event)
        v = self.map
        if v:
            d = event.wheel_value()
            self._move_plane(d)

    def mouse_up(self, event = None):
        self.map = None
        self.ijk = None
        self.xy_last = None
        self.frac_istep = 0
        return

    def laser_click(self, xyz1, xyz2):
        line = (xyz1, xyz2)
        self._choose_box_face(line)
        
    def drag_3d(self, position, move, delta_z):
        if position is None:
            self.mouse_up()
        elif move is not None:
            v = self.map
            if v:
                trans = move * position.origin() - position.origin()
                dxyz = v.position.inverse() * trans
                dijk = v.data.xyz_to_ijk_transform.transform_vector(dxyz)
                istep = dijk[self.axis]
                self._move_plane(istep)

class PlanesMouseMode(RegionMouseMode):
    name = 'move planes'
    icon_file = 'moveplanes.png'
    move_faces = False

class CropMouseMode(RegionMouseMode):
    name = 'crop volume'
    icon_file = 'crop.png'
    move_faces = True
    
def matching_maps(v, maps):
    mm = []
    vd = v.data
    vp = v.scene_position
    for m in maps:
        d = m.data
        if (m is not v and
            tuple(d.size) == tuple(vd.size) and
            d.xyz_to_ijk_transform.same(vd.xyz_to_ijk_transform) and
            m.scene_position.same(vp) and
            same_orthoplanes(m, v)):
            mm.append(m)
    for vc in v.other_channels():
        if vc not in mm:
            mm.append(vc)
    return mm

def same_orthoplanes(v1, v2):
    s1 = v1.showing_orthoplanes()
    s2 = v2.showing_orthoplanes()
    if s1 == s2:
        if v1.rendering_options.orthoplane_positions == v2.rendering_options.orthoplane_positions:
            return True
    return False
    
def move_face(v, axis, side, istep):

    ijk_min, ijk_max, ijk_step = [list(u) for u in v.region]
    amax = v.data.size[axis]-1
    minsep = ijk_step[axis]-1
    
    if side == 0:
        istep = max(istep, -ijk_min[axis])
        istep = min(istep, amax - (ijk_min[axis]+minsep))
    else:
        istep = max(istep, -(ijk_max[axis]-minsep))
        istep = min(istep, amax - ijk_max[axis])
    (ijk_min, ijk_max)[side][axis] += istep

    # If face hits opposing face then move opposing face too
    if ijk_max[axis] - ijk_min[axis] < minsep:
        if side == 0:
            ijk_max[axis] = ijk_min[axis] + minsep
        else:
            ijk_min[axis] = ijk_max[axis] - minsep

    v.new_region(ijk_min, ijk_max, ijk_step, adjust_step = False, adjust_voxel_limit = False)
    
def move_slab(v, axis, side, istep):

    if v.showing_orthoplanes():
        move_orthoplane(v, axis, istep)
        return

    ijk_min, ijk_max, ijk_step = [list(u) for u in v.region]
    amax = v.data.size[axis]-1
    istep = max(istep, -ijk_min[axis])              # clamp step
    istep = min(istep, amax - ijk_max[axis])
    ijk_min[axis] += istep
    ijk_max[axis] += istep

    v.new_region(ijk_min, ijk_max, ijk_step, adjust_step = False, adjust_voxel_limit = False)

def move_orthoplane(v, axis, istep):

    ijk = list(v.rendering_options.orthoplane_positions)
    ijk[axis] += istep
    ijk_min, ijk_max = v.region[:2]
    if ijk[axis] < ijk_min[axis]:
        ijk[axis] = ijk_min[axis]
    elif ijk[axis] > ijk_max[axis]:
        ijk[axis] = ijk_max[axis]
    v.set_parameters(orthoplane_positions = tuple(ijk))

def drag_distance(v, ijk, axis, dx, dy, viewer, clamp_speed = 3):
    from math import sqrt
    d = sqrt(dx*dx + dy*dy)
    face_normal = v.axis_vector(axis)    # global coords
    m2c = viewer.camera.position.inverse()
    nx,ny,nz = m2c.transform_vector(face_normal)
    if ((dx == 0 and abs(dy) == 1 and abs(nx) > abs(ny)) or
        (dy == 0 and abs(dx) == 1 and abs(ny) > abs(nx))):
        # Slow mouse drags generate single pixel steps (1,0), (0,1), ...
        # and these will cause the motion to jitter back and forth when
        # they are in opposite directions when projected onto the plane normal.
        # This avoids the jitter.
        return 0
    psize = viewer.pixel_size(v.ijk_to_global_xyz(ijk))
    nxy = sqrt(nx*nx + ny*ny)
    cosa = (dx*nx + dy*ny) / (d*nxy) if d*nxy > 0 else sign(dy)
    nstep = psize * d * cosa / max(nxy, 1.0/clamp_speed)  # physical units
    return nstep

def sign(x):
    return 1 if x >= 0 else -1

def register_mousemode(session):
    mm = session.ui.mouse_modes
    mm.add_mode(PlanesMouseMode(session))
    mm.add_mode(CropMouseMode(session))
