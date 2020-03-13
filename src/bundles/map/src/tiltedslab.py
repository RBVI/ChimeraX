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
class RotateSlabMouseMode(MouseMode):
    name = 'rotate slab'
    icon_file = 'icons/rotateslab.png'

    def __init__(self, session):

        MouseMode.__init__(self, session)

        self._map = None
        self._matching_maps = []	# Adjust region for these maps too.
        self._xy_last = None
        self._center = None		# Center of rotation
        
    def mouse_down(self, event):
        self._xy_last = (x,y) = event.position()
        v = self.session.main_view
        line = v.clip_plane_points(x,y)    # scene coordinates
        if line[0] is None or line[1] is None:
            return  # Camera does not support ray casting, for example VR or stereo cameras.
        self._choose_map(line)

    def _choose_map(self, line):
        from .volume import Volume
        maps = [m for m in self.session.models.list() if isinstance(m, Volume) and m.shown()]
        from .slice import nearest_volume_face
        v, axis, side, ijk = nearest_volume_face(line, maps)
        self._map = v
        if v:
            from .moveplanes import matching_maps
            self._matching_maps = matching_maps(v, maps)
            ro = v.rendering_options
            if ro.image_mode != 'tilted slab':
                self._set_initial_slab()
            self._center = self._center_of_rotation(v, line)

    def _set_initial_slab(self):
        set_initial_tilted_slab(self._map, matching_maps = self._matching_maps)

    def _center_of_rotation(self, v, line):
        # Use point under mouse on middle slice of slab.
        ro = v.rendering_options
        spacing = ro.tilted_slab_spacing
        thickness =  (ro.tilted_slab_plane_count - 1) * spacing
        offset = ro.tilted_slab_offset + 0.5*thickness
        from . import box_cuts
        va, ta = box_cuts(v.corners(), ro.tilted_slab_axis, offset, spacing, 1)
        if len(va) == 0:
            # Middle slice is outside volume box.
            c = v.center()
        else:
            vxyz0, vxyz1 = v.scene_position.inverse() * line
            from chimerax.geometry import closest_triangle_intercept
            f, t = closest_triangle_intercept(va, ta, vxyz0, vxyz1)
            if f is None:
                # No intercept with middle slice.  Use center of slice.
                c = va.mean(axis = 0)
            else:
                # Use intercept point with slice.
                c = (1-f)*vxyz0 + f*vxyz1
            
        return c

    def mouse_drag(self, event):
        v = self._map
        if v is None or self._xy_last is None:
            self.mouse_down(event)
            return

        xl, yl = self._xy_last
        x,y = event.position()
        dx, dy = (x - xl, yl - y)
        if dx == 0 and dy == 0:
            return
        
        camera = self.session.main_view.camera
        if event.shift_down():
            # Translate slab
            ro = v.rendering_options
            spacing = ro.tilted_slab_spacing
            slab_normal = v.scene_position.transform_vector(ro.tilted_slab_axis)
            move_dir = camera.position.transform_vector((dx,dy,0))
            from chimerax.geometry import inner_product, norm
            sign = 1 if inner_product(move_dir, slab_normal) > 0 else -1
            dist = sign * norm(move_dir) * spacing
            self._move_slab(dist)
        else:
            # Rotate slab
            from math import sqrt
            dn = sqrt(dx*dx + dy*dy)
            rangle = dn
            raxis = camera.position.transform_vector((-dy/dn, dx/dn, 0))
            self._rotate_slab(raxis, rangle)
        self._xy_last = (x,y)

    def _rotate_slab(self, axis, angle, center = None):
        v = self._map
        vaxis = v.scene_position.inverse().transform_vector(axis)
        ro = v.rendering_options
        saxis, soffset = ro.tilted_slab_axis, ro.tilted_slab_offset
        thickness = ro.tilted_slab_spacing * (ro.tilted_slab_plane_count - 1)
        if center is None:
            rcenter = self._center
        else:
            rcenter = v.scene_position.inverse() * center
        from chimerax.geometry import rotation, inner_product
        axis = rotation(vaxis, angle).transform_vector(saxis)
#        offset = inner_product(axis, rcenter) - 0.5*thickness
        offset = soffset + inner_product(axis-saxis, rcenter)
        offset = keep_tilted_slab_in_box(v, offset, axis = axis)
        for m in [v] + self._matching_maps:
            m.set_parameters(tilted_slab_axis = axis,
                             tilted_slab_offset = offset)

        # Make sure new plane is shown before another mouse event shows another plane.
        self.session.update_loop.update_graphics_now()
        
    def _move_slab(self, distance):
        v = self._map
        offset = v.rendering_options.tilted_slab_offset + distance
        offset = keep_tilted_slab_in_box(v, offset)
        for m in [v] + self._matching_maps:
            m.set_parameters(tilted_slab_offset = offset)
        # Make sure new plane is shown before another mouse event shows another plane.
        self.session.update_loop.update_graphics_now()

    def wheel(self, event):
        self.mouse_down(event)
        v = self._map
        if v:
            d = event.wheel_value()
            spacing = v.rendering_options.tilted_slab_spacing
            self._move_slab(d*spacing)

    def mouse_up(self, event = None):
        self.log_volume_command()
        self._map = None
        self._xy_last = None

    def vr_press(self, event):
        # Virtual reality hand controller button press.
        xyz1, xyz2 = event.picking_segment()
        line = (xyz1, xyz2)
        self._choose_map(line)
        
    def vr_motion(self, event):
        v = self._map
        if v is None:
            return

        trans = event.tip_motion
        dxyz = v.scene_position.inverse().transform_vector(trans)
        ro = v.rendering_options
        from chimerax.geometry import inner_product
        dist = inner_product(ro.tilted_slab_axis, dxyz)
        self._move_slab(dist)
        center = event.tip_position
        axis, angle = event.motion.rotation_axis_and_angle()
        self._rotate_slab(axis, angle, center)

    def vr_release(self, event):
        # Virtual reality hand controller button release.
        self.mouse_up()

    def log_volume_command(self):
        v = self._map
        if v:
            ro = v.rendering_options
            options = ('tiltedSlabAxis %.4g,%.4g,%.4g' % tuple(ro.tilted_slab_axis) +
                       ' tiltedSlabOffset %.4g' % ro.tilted_slab_offset)
            command = 'volume #%s %s' % (v.id_string, options)
            from chimerax.core.commands import log_equivalent_command
            log_equivalent_command(v.session, command)

def set_initial_tilted_slab(volume, matching_maps = []):
    v = volume
    vdir = volume.session.main_view.camera.view_direction()
    axis = -v.scene_position.inverse().transform_vector(vdir)
    from chimerax.geometry import inner_product
    offset = inner_product(axis, v.center())
    spacing = min(v.data.step)
    v.expand_single_plane()
    plane_count = max(1, min(v.matrix_size()) // 5)
    offset -= 0.5*plane_count*spacing	# Center slab at volume center.
    for m in [v] + list(matching_maps):
        m.expand_single_plane()
        m.set_parameters(image_mode = 'tilted slab',
                         tilted_slab_axis = axis,
                         tilted_slab_offset = offset,
                         tilted_slab_spacing = spacing,
                         tilted_slab_plane_count = plane_count,
                         show_outline_box = True)

def move_tilted_slab_face(v, side, istep):
    ro = v.rendering_options
    offset, planes = ro.tilted_slab_offset, ro.tilted_slab_plane_count
    if side == 0:
        offset += istep * ro.tilted_slab_spacing
        planes -= istep
        if planes < 1:
            planes = 1
    else:
        planes += istep
        if planes < 1:
            offset += (planes-1) * ro.tilted_slab_spacing
            planes = 1
    offset = keep_tilted_slab_in_box(v, offset, planes = planes)
    v.set_parameters(tilted_slab_plane_count = planes,
                     tilted_slab_offset = offset)

def move_tilted_slab(v, istep):
    ro = v.rendering_options
    offset = ro.tilted_slab_offset + istep * ro.tilted_slab_spacing
    offset = keep_tilted_slab_in_box(v, offset)
    v.set_parameters(tilted_slab_offset = offset)

def keep_tilted_slab_in_box(v, offset, axis = None, planes = None):
    ro = v.rendering_options
    if planes is None:
        planes = ro.tilted_slab_plane_count
    if axis is None:
        axis = ro.tilted_slab_axis
    spacing = ro.tilted_slab_spacing
        
    from . import offset_range
    omin, omax = offset_range(v.corners(), axis)

    if offset < omin:
        noffset = omin
    elif offset + planes*spacing > omax:
        noffset = omax - planes*spacing
    else:
        noffset = offset

    return noffset

def slab_segment(v, line):
    '''Line segment and returned segment are in volume coordinates'''
    ro = v.rendering_options
    offset1 = ro.tilted_slab_offset
    thickness = ro.tilted_slab_spacing * (ro.tilted_slab_plane_count - 1)
    offset2 = offset1 + thickness
    from chimerax.geometry import clip_segment
    cline = clip_segment(line, ro.tilted_slab_axis, offset1, offset2)
    return cline

def register_mousemode(session):
    mm = session.ui.mouse_modes
    mm.add_mode(RotateSlabMouseMode(session))
