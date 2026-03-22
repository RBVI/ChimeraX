# vim: set expandtab ts=4 sw=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2022 Regents of the University of California. All rights reserved.
# The ChimeraX application is provided pursuant to the ChimeraX license
# agreement, which covers academic and commercial uses. For more details, see
# <https://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
#
# This particular file is part of the ChimeraX library. You can also
# redistribute and/or modify it under the terms of the GNU Lesser General
# Public License version 2.1 as published by the Free Software Foundation.
# For more details, see
# <https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html>
#
# THIS SOFTWARE IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND, EITHER
# EXPRESSED OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
# OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. ADDITIONAL LIABILITY
# LIMITATIONS ARE DESCRIBED IN THE GNU LESSER GENERAL PUBLIC LICENSE
# VERSION 2.1
#
# This notice must be embedded in or attached to all copies, including partial
# copies, of the software or any revisions or derivations thereof.
# === UCSF ChimeraX Copyright ===

# TODO:
#   1) Might be nice to show surface instead of ortho-planes in box.
#   2) Allow session saving
#   3) Allow picking the particle and creating a permanent outline box.
#   4) Allow exporting density for all picked particles, and a file of orientations and centers.
#   5) Make choosing new particle size in panel update orthoview immediately.

# -------------------------------------------------------------------------
#
from chimerax.mouse_modes import MouseMode
class OrthoPickMode(MouseMode):
    name = 'orthopick'
    icon_file = 'orthopick.png'

    def __init__(self, session):
        MouseMode.__init__(self, session)
        self._active_orthoview_axis = None, None, None	# OrthoView and axis for mouse press
        self._last_orthoview = None
        self._adjust_box_volume_contour = None
        self._box_volume_level = None
        
    @property
    def settings(self):
        return orthopick_panel(self.session)

    def enable(self):
        self.settings.show()

    def mouse_down(self, event):
        MouseMode.mouse_down(self, event)	# Record position for using mouse_motion()

        bv = self._picked_box_volume(event)
        if bv:
            self._adjust_box_volume_contour = bv
            return
        
        self._active_orthoview_axis = ov, axis, zfacing = self._orthoview_pick(event)
        if ov is None:
            self._create_orthoview(event)

    def _picked_box_volume(self, event):
        ov = self.current_orthoview
        if ov is None or ov.deleted:
            return None
        bv = ov.box_volume
        if bv is None or bv.deleted:
            return None
        x,y = event.position()
        pick = self.session.main_view.picked_object(x, y)
        from chimerax.map import PickedMap
        if isinstance(pick, PickedMap) and pick.map == bv:
            return bv
        return None
        
    def _create_orthoview(self, event):
        v, sxyz = self._volume_pick(event)
        if sxyz is not None:
            self._replace_orthoview(v, sxyz)

    def _replace_orthoview(self, volume, center):
        self._close_current_orthoview()
        ov = OrthoView(self.session, volume, center, box_size = self.settings.box_size,
                       slab_thickness = self._slab_thickness(),
                       membrane_surface = self.settings.membrane_surface,
                       box_volume_level = self._box_volume_level)
        self._last_orthoview = ov
        return ov

    def recreate_orthoview(self):
        '''Box size or orientation changed.'''
        ov = self.current_orthoview
        if ov:
            show_bv = ov.box_volume_shown()
            ov = self._replace_orthoview(ov.volume, ov.center)
            if show_bv:
                ov.show_box_volume()
        
    def mouse_drag(self, event):
        dx, dy = self.mouse_motion(event)
        bv = self._adjust_box_volume_contour
        if bv:
            from chimerax.map.mouselevel import adjust_threshold_level
            adjust_threshold_level(bv, -0.001*dy)
            
        ov, axis, zfacing = self._active_orthoview_axis
        if ov is None or axis is None:
            return

        psize = self.pixel_size(ov.center)
        cam_shift = (psize*dx,-psize*dy,0)
        camera = self.session.main_view.camera
        scene_shift = camera.position.transform_vector(cam_shift)
        # Find drag displacement in orthoview coordinates
        ov_shift = ov.ortho_position.inverse().transform_vector(scene_shift)
        shift = ov_shift[1] if axis == 2 and zfacing else ov_shift[axis]
        if event.shift_down():
            ov.shift_center(axis, shift)
        else:
            ov.shift_plane(axis, shift)

    def mouse_up(self, event = None):
        self._active_orthoview_axis = None, None, None
        self._adjust_box_volume_contour = None

    def _orthoview_pick(self, event):
        x,y = event.position()
        view = self.session.main_view
        pick = view.picked_object(x, y, max_transparent_layers = 0, exclude = _exclude_volumes)
        from chimerax.core.models import PickedModel
        if isinstance(pick, PickedModel) and isinstance(pick.model, OrthoView):
            if hasattr(pick, 'picked_triangle'):
                pt = pick.picked_triangle
                d = pt.drawing()
                axis = getattr(d, 'ortho_axis', None)
                zfacing = getattr(d, 'zfacing', None)
                return pick.model, axis, zfacing
        return None, None, None

    def _volume_pick(self, event):
        x,y = event.position()
        view = self.session.main_view
        xyz1, xyz2 = view.clip_plane_points(x, y)
        from chimerax.map import Volume
        vlist = self.session.models.list(type = Volume)
        from chimerax.markers.mouse import volume_plane_intercept
        sxyz, v = volume_plane_intercept(xyz1, xyz2, vlist)
        return v, sxyz

    def _slab_thickness(self):
        t = self.settings.slab_thickness
        return 0 if t is None else t

    def update_slab_thickness(self):
        lov = self._last_orthoview
        if lov and not lov.deleted:
            lov.update_slab_thickness(self._slab_thickness())

    @property
    def current_orthoview(self):
        lov = self._last_orthoview
        if lov and lov.deleted:
            lov = self._last_orthoview = None
        return lov

    def _close_current_orthoview(self):
        lov = self._last_orthoview
        if lov is not None and not lov.deleted:
            bv = lov.box_volume
            if bv and not bv.deleted and len(bv.surfaces) > 0:
                self._box_volume_level = bv.surfaces[0].level
            self.session.models.close([lov])

# -------------------------------------------------------------------------
#
def _exclude_volumes(drawing):
    if not drawing.pickable:
        return True
    from chimerax.map import Volume
    return isinstance(drawing, Volume)

# -------------------------------------------------------------------------
#
def _closest_surface_point(sxyz, msurf):
    if msurf.vertices is None or msurf.triangles is None or len(msurf.triangles) == 0:
        return None
    from chimerax.surface import surface_distance
    from numpy import array, float32
    points = array(sxyz, float32).reshape((1,3))
    vertices = msurf.vertices.copy()
    msurf.scene_position.transform_points(vertices, in_place = True)
    triangles = msurf.triangles
    dist = surface_distance(points, vertices, triangles)
    closest_point = dist[0,1:4]
    return closest_point

# -------------------------------------------------------------------------
#
from chimerax.core.models import Model
class OrthoView(Model):

    def __init__(self, session, volume, center, box_size = 200, oversample = 2.0,
                 slab_thickness = 0, membrane_surface = None, box_volume_level = None):
        Model.__init__(self, 'orthopick', session)
        self.volume = volume
        self.center = center		# scene coordinates
        self.box_size = box_size
        self.oversample = oversample
        self.slab_thickness = slab_thickness
        self.membrane_surface = membrane_surface
        self.ortho_position = None	# Place for origin ortho drawing position
        self._ortho_drawings = {}	# Map axis 0,1,2 to ortho Drawing
        self._plane_drawings = {}	# Map axis 0,1,2 to plane Drawing
        self._ortho_color = {}		# Map axis to VolumeColor instance
        self.box_volume = None		# Volume for pick box
        self._box_volume_level = box_volume_level	# Prefered box volume contour level

         # Map axis to border color
        self._border_color = {0: (0,0,255,255), 1:(0,255,0,255), 2:(255,0,0,255)}

        self._create_drawings()

        # Would like to add OrthoView as child of volume,
        # but Volume does not allow picking of child models.
        self.scene_position = volume.scene_position

        self._update_plane_colors()

        session.models.add([self])

        volume.add_volume_change_callback(self._volume_changed)

    def _create_drawings(self):
        width = self.box_size
        from math import ceil
        grid_size = max(2, int(ceil(self.oversample * width / self.volume.data.step[0])))

        # Make ortho planes
        self.ortho_position = oplace = self._orthoview_placement()
        slab_offsets = self._slab_offsets(self.slab_thickness)
        for name, axis, normal in [('z outline', 2, (0,0,1)),
                                   ('y outline', 1, (0,-1,0)),
                                   ('x outline', 0, (1,0,0))]:
            plane = _make_square_drawing(name, width, normal = normal, grid_size = grid_size,
                                         border_color = self._border_color[axis])
            plane.position = oplace
            plane.ortho_axis = plane.outline.ortho_axis = axis
            plane.zfacing = plane.outline.zfacing = False
            self.add_drawing(plane)
            self._ortho_drawings[axis] = plane
            from chimerax.surface.colorvol import VolumeColor
            vc = VolumeColor(plane, self.volume, palette = self._colormap(axis),
                             offset = slab_offsets, auto_recolor = False)
            self._ortho_color[axis] = vc
            
        # Make flat planes
        fplace = self._orthoview_placement(tilt = False)
        pad = self.volume.data.step[0]
        zo = pad
        o = width + pad
        for name, axis, offset in [('z view', 2, (o, -o, zo)),
                                   ('y view', 1, (0, -o, zo)),
                                   ('x view', 0, (o, 0, zo))]:
            view = _make_square_drawing(name, width, normal = (0,0,1), grid_size = grid_size,
                                        border_color = self._border_color[axis])
            from chimerax.geometry import translation
            view.position = fplace * translation(offset)
            view.ortho_axis = view.outline.ortho_axis = axis
            view.zfacing = view.outline.zfacing = True
            self.add_drawing(view)
            self._plane_drawings[axis] = view

    def _colormap(self, axis):
        modulation_color = [.8 + .2*c/255 for c in self._border_color[axis]]
        colormap = _volume_colormap(self.volume, modulation_color)
        return colormap
        
    def _orthoview_placement(self, tilt = True):
        vcenter = self.volume.scene_position.inverse() * self.center
        vnormal = self._membrane_normal()
        if vnormal is None:
            from chimerax.geometry import translation
            p = translation(vcenter)
        else:
            if not tilt:
                vnormal = (vnormal[0], vnormal[1], 0)
            from chimerax.geometry import orthonormal_frame, rotation, translation
            p = (translation(vcenter) *
                 orthonormal_frame(vnormal, ydir = (0,0,-1)) *
                 rotation((1,0,0), 90))
        return p

    def _membrane_normal(self):
        msurf = self.membrane_surface
        if msurf is None:
            return None
        
        center = self.center
        mxyz = _closest_surface_point(center, msurf)
        if mxyz is None:
            return None

        from chimerax.geometry import distance
        if distance(mxyz, center) > 2*self.box_size:
            return None
        
        # Map orthoview y axis to negative membrane normal,
        # map orthoview z axis to be in plane of volume z-axis and membrane normal.
        s2v = self.volume.scene_position.inverse()
        vnormal = s2v.transform_vector(mxyz - center) # Membrane normal in volume coords

        return vnormal

    def shift_plane(self, axis, shift):
        shift_vector = [0,0,0]
        shift_vector[axis] = shift
        from chimerax.geometry import translation
        od = self._ortho_drawings[axis]
        od.position *= translation(shift_vector)
        self._update_plane_colors(axis)

    def shift_center(self, axis, shift):
        shift_vector = [0,0,0]
        shift_vector[axis] = shift
        od_shift = self._ortho_drawings[0].position.transform_vector(shift_vector)
        self.center += od_shift
        from chimerax.geometry import translation
        od_trans = translation(od_shift)
        for axis in (0,1,2):
            od = self._ortho_drawings[axis]
            od.position = od_trans * od.position
        self.ortho_position = od_trans * self.ortho_position
        self._update_plane_colors()

    def _update_plane_colors(self, axis = None):
        if axis is None:
            for axis in (0,1,2):
                self._update_plane_colors(axis)
        else:
            od = self._ortho_drawings[axis]
            oc = self._ortho_color[axis]
            oc.set_vertex_colors(report_stats = False)
            pd = self._plane_drawings[axis]
            pd.vertex_colors = od.vertex_colors

    def _update_plane_colormap(self):
        for axis in (0,1,2):
            self._ortho_color[axis].colormap = self._colormap(axis)
        self._update_plane_colors()

    def _volume_changed(self, volume, change_type):
        if change_type == 'thresholds changed':
            self._update_plane_colormap()

    def update_slab_thickness(self, thickness):
        offsets = self._slab_offsets(thickness)
        self.slab_offsets = offsets
        for axis in (0,1,2):
            oc = self._ortho_color[axis]
            oc.offset = offsets
        self._update_plane_colors()

    def _slab_offsets(self, thickness):
        step = min(self.volume.data.step)
        offset_step = 0.5*step
        if 0.5*thickness < offset_step:
            return 0
        steps = int(0.5*thickness / offset_step)
        offsets = tuple(s*offset_step for s in range(-steps, steps+1))
        return offsets

    def _show_ortho_planes(self, show = True):
        for od in self._ortho_drawings.values():
            if show:
                if hasattr(od, '_keep_plane_triangles'):
                    od.set_geometry(od.vertices, od.normals, od._keep_plane_triangles)
                    self._update_plane_colors()
            else:
                if not hasattr(od, '_keep_plane_triangles'):
                    od._keep_plane_triangles = od.triangles
                from numpy import empty, int32
                no_triangles = empty((0,3),int32)
                od.set_geometry(od.vertices, od.normals, no_triangles)

    def show_box_volume(self, show = True):
        v = self.box_volume
        if show:
            if v is None:
                v = self._create_box_volume()
            v.show()
            self._show_ortho_planes(False)
        else:
            if v is not None:
                v.show(show = False)
                self._show_ortho_planes(True)

    def box_volume_shown(self):
        v = self.box_volume
        return v and not v.deleted and v.display

    def _create_box_volume(self):
        v = self.box_volume
        if v is not None and not v.deleted:
            return v
        width = self.box_size
        from math import ceil
        grid_size = max(2, int(ceil(self.oversample * width / self.volume.data.step[0])))
        grid_spacing = width / (grid_size - 1)
        pos = self.ortho_position
        c = -grid_spacing * (grid_size - 1)/2
        corner = (c,c,c)
        origin = pos * corner
        v_id = self.id + (1,)
        from chimerax.map_filter.vopcommand import volume_new, volume_add
        v = volume_new(self.session, 'box map', origin = origin, size = (grid_size, grid_size, grid_size),
                       grid_spacing = (grid_spacing, grid_spacing, grid_spacing), model_id = v_id)
        v.scene_position = self.volume.scene_position
        v.data.set_rotation(pos.matrix[:,:3])
        v.set_parameters(show_outline_box = True, cap_faces = False)
        level = self._box_volume_level
        if level is not None:
            v.set_parameters(surface_levels = [level])
        volume_add(self.session, [v, self.volume], in_place = True, hide_maps = False)
        v.show()
        self.box_volume = v
        v.update_drawings()	# Create surfaces so dust can be hidden
        self._hide_dust()
        self._show_ortho_planes(False)
        return v

    def _hide_dust(self):
        v = self.box_volume
        if v is None or v.deleted:
            return
        metric = 'volume rank'
        size = 1
        from chimerax.surface import dust
        for surf in v.surfaces:
            dust.hide_dust(surf, metric, size, auto_update = True)

    def save_box_volume(self):
        v = self._create_box_volume()
        from chimerax.shortcuts.shortcuts import unused_file_name
        path = unused_file_name('~/Desktop', 'particle', '.mrc')
        from chimerax.core.commands import run
        run(self.session, f'save {path} model #{v.id_string}')

# -------------------------------------------------------------------------
#
def _make_square_drawing(name, width, normal = (0,0,1), grid_size = 2,
                         border_color = (255,255,255,255)):
    from chimerax.graphics import Drawing
    d = Drawing(name)
    from chimerax.geometry import translation, vector_rotation
    d.color = border_color
    d.use_lighting = False
    from chimerax.shape import shape
    varray, tarray = shape.rectangle_geometry(width, width, grid_size, grid_size)
    # Rotate
    vector_rotation((0,0,1), normal).transform_points(varray, in_place = True)
    from numpy import empty, float32
    narray = empty((len(varray),3), float32)
    narray[:,:] = normal
    d.set_geometry(varray, narray, tarray)
    od = _make_outline_drawing(width, normal, border_color)
    d.add_drawing(od)
    d.outline = od
    return d
        
# -------------------------------------------------------------------------
#
def _make_outline_drawing(width, normal = (0,0,1), color = (255,255,255,255)):
    from chimerax.graphics import Drawing
    d = Drawing('outline')
    d.color = color
    h = width/2
    from numpy import array, float32, int32, uint8
    vertices = array([(h,h,0), (-h,h,0), (-h,-h,0), (h,-h,0)], float32)
    from chimerax.geometry import vector_rotation
    vector_rotation((0,0,1), normal).transform_points(vertices, in_place = True)
    normals = None
    triangles = array([(0,1,2), (2,3,0)], int32)
    d.set_geometry(vertices, normals, triangles)
    d.display_style = d.Mesh
    d.edge_mask = array((0x3, 0x3), uint8)
    return d
    
# -------------------------------------------------------------------------
#
def _volume_colormap(v, modulation_color = None):
    lc = list(zip(v.image_levels, v.image_colors))
    lc.sort(key = lambda level_intensity_color: level_intensity_color[0][0])
    levels = [level for (level,intensity), color in lc]
    colors = [[intensity*c for c in color[:3]]+[1]
              for (level,intensity), color in lc]  # Opaque colors
    if modulation_color:
        colors = [[mc*cc for cc,mc in zip(modulation_color,c)] for c in colors]
    ro = v.rendering_options
    extend_kw = {}
    if not ro.colormap_extend_right:
        extend_kw['color_above_value_range'] = (0,0,0,1)  # black
    if not ro.colormap_extend_left:
        extend_kw['color_below_value_range'] = (0,0,0,1)  # black
    from chimerax.core.colors import Colormap
    cmap = Colormap(levels, colors, **extend_kw)
    return cmap

# -----------------------------------------------------------------------------
# Panel for orthopick mouse mode settings.
#
from chimerax.core.tools import ToolInstance
class OrthoPickSettings(ToolInstance):
    help = "help:user/tools/orthopick.html"

    def __init__(self, session, tool_name):
        
        ToolInstance.__init__(self, session, tool_name)

        self.display_name = 'Orthopick Mouse Mode Settings'

        from chimerax.ui import MainToolWindow
        tw = MainToolWindow(self)
        self.tool_window = tw
        parent = tw.ui_area

        from chimerax.ui.widgets import vertical_layout
        layout = vertical_layout(parent, margins = (5,0,0,0))

        from chimerax.ui.widgets import row_frame, EntriesRow
        ps = EntriesRow(parent, 'Particle size (Angstroms)', 200.0)
        self._box_size = ps.values[0]
        self._box_size.return_pressed.connect(self._box_size_changed)

        f, olayout = row_frame(parent)
        cc = EntriesRow(f, False, 'Orient to membrane surface')
        self._orient_to_membrane = cc.values[0]
        self._orient_to_membrane.changed.connect(self._orient_to_membrane_changed)
                
        from chimerax.ui.widgets import ModelMenu
        from chimerax.core.models import Surface
        smenu = ModelMenu(session, f, label = '', model_types = [Surface])
        self._membrane_surface = smenu
        olayout.addWidget(smenu.frame)
        olayout.addStretch(1)   # Extra space at end

        st = EntriesRow(parent, False, 'Average slab density, thickness (Angstroms)', 50.0)
        self._average_slab, self._slab_thickness = st.values
        self._average_slab.changed.connect(self._slab_thickness_changed)
        self._slab_thickness.return_pressed.connect(self._slab_thickness_changed)

        from chimerax.ui.widgets import button_row
        button_row(parent, [('Show particle surface', self._show_particle_surface),
                            ('Save particle map', self._save_particle_map),
                            ('Help', self._show_help)])

        layout.addStretch(1)    # Extra space at end

        tw.manage(placement="side")

    @classmethod
    def get_singleton(self, session, create=True):
        from chimerax.core import tools
        return tools.get_singleton(session, OrthoPickSettings, 'Orthopick Mouse Mode Settings', create=create)

    def show(self):
        self.tool_window.shown = True

    def hide(self):
        self.tool_window.shown = False

    def _show_help(self):
        from chimerax.core.commands import run
        run(self.session, 'help %s' % self.help)

    def _show_particle_surface(self):
        oview = self._orthoview
        if oview is not None:
            oview.show_box_volume(not oview.box_volume_shown())

    def _save_particle_map(self):
        oview = self._orthoview
        if oview is not None:
            oview.save_box_volume()
        
    @property
    def membrane_surface(self):
        if self._orient_to_membrane.value:
            return self._membrane_surface.value
        return None

    def _orient_to_membrane_changed(self, *args):
        self._orthopick_mode.recreate_orthoview()

    @property
    def box_size(self):
        return self._box_size.value

    def _box_size_changed(self, *args):
        self._orthopick_mode.recreate_orthoview()

    @property
    def slab_thickness(self):
        if self._average_slab.value:
            return self._slab_thickness.value
        return None

    def _slab_thickness_changed(self, *args):
        self._orthopick_mode.update_slab_thickness()

    @property
    def _orthopick_mode(self):
        return self.session.ui.mouse_modes.named_mode('orthopick')

    @property
    def _orthoview(self):
        return self._orthopick_mode.current_orthoview

# -------------------------------------------------------------------------
#
def orthopick_panel(session, create = True):
    return OrthoPickSettings.get_singleton(session, create)

# -------------------------------------------------------------------------
#
def register_mousemode(session):
    mm = session.ui.mouse_modes
    mm.add_mode(OrthoPickMode(session))
