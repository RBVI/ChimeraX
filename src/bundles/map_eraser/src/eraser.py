# vim: set expandtab ts=4 sw=4:

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

# -------------------------------------------------------------------------
#
def volume_erase(session, volumes, center, radius, coordinate_system = None,
                 outside = False, value = 0):
    '''Erase a volume inside or outside a sphere.'''
    ev = []
    cscene = center.scene_coordinates(coordinate_system)
    for volume in volumes:
        cvol = volume.scene_position.inverse() * cscene
        v = volume.writable_copy()
        ev.append(v)
        if outside:
            _set_data_outside_sphere(v.data, cvol, radius, value)
        else:
            _set_data_in_sphere(v.data, cvol, radius, value)
    return ev[0] if len(ev) == 1 else ev

# -----------------------------------------------------------------------------
#
def _set_data_in_sphere(grid_data, center, radius, value = 0):

    # Optimization: Mask only subregion containing sphere.
    ijk_min, ijk_max = _sphere_grid_bounds(grid_data, center, radius)
    from chimerax.map_data import GridSubregion, zone_mask
    subgrid = GridSubregion(grid_data, ijk_min, ijk_max)

    mask = zone_mask(subgrid, [center], radius)

    dmatrix = subgrid.full_matrix()

    from numpy import putmask
    putmask(dmatrix, mask, value)

    grid_data.values_changed()

# -----------------------------------------------------------------------------
#
def _set_data_outside_sphere(grid_data, center, radius, value = 0):

    from chimerax.map_data import zone_mask
    mask = zone_mask(grid_data, [center], radius, invert_mask = True)

    dmatrix = grid_data.full_matrix()

    from numpy import putmask
    putmask(dmatrix, mask, value)

    grid_data.values_changed()

# -----------------------------------------------------------------------------
#
def _sphere_grid_bounds(grid_data, center, radius):

    ijk_center = grid_data.xyz_to_ijk(center)
    spacings = grid_data.plane_spacings()
    ijk_size = [radius/s for s in spacings]
    from math import floor, ceil
    ijk_min = [max(int(floor(c-s)), 0) for c,s in zip(ijk_center,ijk_size)]
    ijk_max = [min(int(ceil(c+s)), m-1) for c, s, m in zip(ijk_center, ijk_size, grid_data.size)]
    return ijk_min, ijk_max
    
# -------------------------------------------------------------------------
#
def register_volume_erase_command(logger):
    from chimerax.core.commands import CmdDesc, register, FloatArg, CenterArg, CoordSysArg, BoolArg
    from chimerax.map import MapsArg
    desc = CmdDesc(
        required = [('volumes', MapsArg),],
        keyword = [('center', CenterArg),
                   ('radius', FloatArg),
                   ('coordinate_system', CoordSysArg),
                   ('outside', BoolArg),
                   ('value', FloatArg),
        ],
        required_arguments = ['center', 'radius'],
        synopsis = 'Set map values to zero inside a sphere'
    )
    register('volume erase', desc, volume_erase, logger=logger)

# -------------------------------------------------------------------------
#
from chimerax.mouse_modes import MouseMode
class MapEraser(MouseMode):
    name = 'map eraser'
    icon_file = 'eraser.png'

    def __init__(self, session):
        MouseMode.__init__(self, session)

    @property
    def settings(self):
        return map_eraser_panel(self.session)
    
    def enable(self):
        self.settings.show()
        
    def mouse_down(self, event):
        MouseMode.mouse_down(self, event)

    def mouse_drag(self, event):
        dx, dy = self.mouse_motion(event)
        settings = self.settings
        # Compute motion in scene coords of sphere center.
        c = settings.sphere_center
        v = self.session.main_view
        s = v.pixel_size(c)
        if event.shift_down():
            shift = (0,0,s*dy)	# Move in z if shift key held.
        else:
            shift = (s*dx, -s*dy, 0)
        
        dxyz = v.camera.position.transform_vector(shift)
        settings.move_sphere(dxyz)

    def mouse_up(self, event):
        MouseMode.mouse_up(self, event)

    def vr_motion(self, event):
        settings = self.settings
        c = settings.sphere_center
        delta_xyz = event.motion*c - c
        settings.move_sphere(delta_xyz)

# -----------------------------------------------------------------------------
# Panel for erasing parts of map in sphere with map eraser mouse mode.
#
from chimerax.core.tools import ToolInstance
class MapEraserSettings(ToolInstance):
    help = "help:user/tools/maperaser.html"

    def __init__(self, session, tool_name):

        self._default_color = (255,153,204,128)		# Transparent pink
        self._max_slider_value = 1000		# QSlider only handles integer values
        self._max_slider_radius = 100.0		# Float maximum radius value, scene units
        self._block_text_update = False		# Avoid radius slider and text continuous updating each other.
        self._block_slider_update = False	# Avoid radius slider and text continuous updating each other.

        b = session.main_view.drawing_bounds()
        vradius = 100 if b is None else b.radius()
        self._max_slider_radius = vradius
        center = b.center() if b else (0,0,0)
        self._sphere_model = SphereModel('eraser sphere', session, self._default_color, center, 0.2*vradius)
        
        ToolInstance.__init__(self, session, tool_name)

        self.display_name = 'Map Eraser'

        from chimerax.ui import MainToolWindow
        tw = MainToolWindow(self)
        self.tool_window = tw
        parent = tw.ui_area

        from Qt.QtWidgets import QVBoxLayout, QHBoxLayout, QFrame, QCheckBox, QLabel, QPushButton, QLineEdit, QSlider
        from Qt.QtCore import Qt

        layout = QVBoxLayout(parent)
        layout.setContentsMargins(0,0,0,0)
        layout.setSpacing(0)
        parent.setLayout(layout)

        sf = QFrame(parent)
        layout.addWidget(sf)
        
        slayout = QHBoxLayout(sf)
        slayout.setContentsMargins(0,0,0,0)
        slayout.setSpacing(10)

        self._show_eraser = se = QCheckBox('Show map eraser sphere', sf)
        se.setCheckState(Qt.Checked)
        se.stateChanged.connect(self._show_eraser_cb)
        slayout.addWidget(se)
        from chimerax.ui.widgets import ColorButton
        self._sphere_color = sc = ColorButton(sf, max_size = (16,16), has_alpha_channel = True)
        sc.color = self._default_color
        sc.color_changed.connect(self._change_color_cb)
        slayout.addWidget(sc)
        slayout.addStretch(1)    # Extra space at end

        rf = QFrame(parent)
        layout.addWidget(rf)
        rlayout = QHBoxLayout(rf)
        rlayout.setContentsMargins(0,0,0,0)
        rlayout.setSpacing(4)

        rl = QLabel('Radius', rf)
        rlayout.addWidget(rl)
        self._radius_entry = rv = QLineEdit('', rf)
        rv.setMaximumWidth(40)
        rv.returnPressed.connect(self._radius_changed_cb)
        rlayout.addWidget(rv)
        self._radius_slider = rs = QSlider(Qt.Horizontal, rf)
        smax = self._max_slider_value
        rs.setRange(0,smax)
        rs.valueChanged.connect(self._radius_slider_moved_cb)
        rlayout.addWidget(rs)

        rv.setText('%.4g' % self._sphere_model.radius)
        self._radius_changed_cb()
        
        ef = QFrame(parent)
        layout.addWidget(ef)
        elayout = QHBoxLayout(ef)
        elayout.setContentsMargins(0,0,0,0)
        elayout.setSpacing(30)
        eb = QPushButton('Erase inside sphere', ef)
        eb.clicked.connect(self._erase_in_sphere)
        elayout.addWidget(eb)
        eo = QPushButton('Erase outside sphere', ef)
        eo.clicked.connect(self._erase_outside_sphere)
        elayout.addWidget(eo)
        rb = QPushButton('Reduce map bounds', ef)
        rb.clicked.connect(self._crop_map)
        elayout.addWidget(rb)
        elayout.addStretch(1)    # Extra space at end

        layout.addStretch(1)    # Extra space at end

        tw.manage(placement="side")

        # When displayed models change update radius slider range.
        from chimerax.core.models import MODEL_DISPLAY_CHANGED
        h = session.triggers.add_handler(MODEL_DISPLAY_CHANGED, self._model_display_change)
        self._model_display_change_handler = h

    def delete(self):
        ses = self.session
        ses.triggers.remove_handler(self._model_display_change_handler)
        sm = self._sphere_model
        if sm and not sm.deleted:
            ses.models.close([sm])
        self._sphere_model = None
        ToolInstance.delete(self)
        
    @classmethod
    def get_singleton(self, session, create=True):
        from chimerax.core import tools
        return tools.get_singleton(session, MapEraserSettings, 'Map Eraser', create=create)

    def show(self):
        self.tool_window.shown = True

    def hide(self):
        self.tool_window.shown = False

    @property
    def sphere_model(self):
        sm = self._sphere_model
        if sm is None or sm.deleted:
            b = self.session.main_view.drawing_bounds()
            center = b.center() if b else (0,0,0)
            sm = SphereModel('eraser sphere', self.session, self._sphere_color.color,
                             center, self._radius_value())
            self._sphere_model = sm
        return sm
    
    def _radius_changed_cb(self):
        if self._block_text_update:
            return
        r = self._radius_value()
        self.sphere_model.radius = r
        sval = int((r / self._max_slider_radius) * self._max_slider_value)
        self._block_text_update = True
        self._radius_slider.setValue(sval)
        self._block_text_update = False

    def _radius_value(self):
        rt = self._radius_entry.text()
        try:
            r = float(rt)
        except ValueError:
            self.session.logger.warning('Cannot parse map eraser radius value "%s"' % rt)
            return 10
        return r

    def _radius_slider_moved_cb(self, event):
        sval = self._radius_slider.value()
        r = (sval / self._max_slider_value) * self._max_slider_radius
        # Setting text does not invoke _radius_changed_cb.
        self._radius_entry.setText('%.4g' % r)
        self.sphere_model.radius = r

    def _model_display_change(self, name, data):
        v = self._shown_volume()
        if v:
            self._adjust_slider_range(v)
        
    def _adjust_slider_range(self, volume):
        xyz_min, xyz_max = volume.xyz_bounds(subregion = 'all')
        rmax = 0.5 * max([x1-x0 for x0,x1 in zip(xyz_min, xyz_max)])
        if rmax != self._max_slider_radius:
            self._max_slider_radius = rmax
            self._radius_changed_cb()
        
    @property
    def sphere_center(self):
        return self.sphere_model.scene_position.origin()

    def move_sphere(self, delta_xyz):
        sm = self.sphere_model
        dxyz = sm.scene_position.inverse().transform_vector(delta_xyz)	# Transform to sphere local coords.
        from chimerax.geometry import translation
        sm.position = sm.position * translation(dxyz)

    def _show_eraser_cb(self, show):
        self.sphere_model.display = show
        
    def _erase_in_sphere(self):
        self._erase()

    def _erase(self, outside = False):
        v, center, radius = self._eraser_region()
        if v is None:
            return
        c = '%.5g,%.5g,%.5g' % tuple(center)
        cmd = 'volume erase #%s center %s radius %.5g' % (v.id_string, c, radius)
        if outside:
            cmd += ' outside true'
        from chimerax.core.commands import run
        run(self.session, cmd)

    def _eraser_region(self):
        from Qt.QtCore import Qt
        if self._show_eraser.checkState() != Qt.Checked:
            return None, None, None

        v = self._shown_volume()
        if v is None:
            self.session.logger.warning('Can only have one displayed volume when erasing')
            return None, None, None

        sm = self.sphere_model
        center = sm.scene_position.origin()
        radius = sm.radius
        
        return v, center, radius

    def _erase_outside_sphere(self):
        self._erase(outside = True)

    def _crop_map(self):
        v, center, radius = self._eraser_region()
        if v is None:
            return
        vcenter = v.scene_position.inverse() * center
        ijk_min, ijk_max, ijk_step = v.bounding_region([vcenter], radius,
                                                       step = 1, cubify = True)
        region = ','.join(['%d,%d,%d' % tuple(ijk_min),
                           '%d,%d,%d' % tuple(ijk_max)])
        cmd = 'volume copy #%s subregion %s' % (v.id_string, region)
        from chimerax.core.commands import run
        run(self.session, cmd)
        
    def _shown_volume(self):
        ses = self.session
        from chimerax.map import Volume
        vlist = [m for m in ses.models.list(type = Volume) if m.visible]
        v = vlist[0] if len(vlist) == 1 else None
        return v

    def _change_color_cb(self, color):
        self.sphere_model.color = color
        
# -------------------------------------------------------------------------
#
from chimerax.core.models import Surface
class SphereModel(Surface):
    SESSION_SAVE = False
    
    def __init__(self, name, session, color, center, radius):
        self._num_triangles = 1000
        Surface.__init__(self, name, session)
        from chimerax.surface import sphere_geometry2
        va, na, ta = sphere_geometry2(self._num_triangles)
        self._unit_vertices = va
        self.set_geometry(radius*va, na, ta)
        self.color = color
        from chimerax.geometry import translation
        self.position = translation(center)
        self._radius = radius
        session.models.add([self])

    def _get_radius(self):
        return self._radius
    def _set_radius(self, r):
        if r != self._radius:
            self._radius = r
            self.set_geometry(r*self._unit_vertices, self.normals, self.triangles)
    radius = property(_get_radius, _set_radius)
        
# -------------------------------------------------------------------------
#
def map_eraser_panel(session, create = True):
    return MapEraserSettings.get_singleton(session, create)

# -------------------------------------------------------------------------
#
def register_mousemode(session):
    mm = session.ui.mouse_modes
    mm.add_mode(MapEraser(session))
