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

# -----------------------------------------------------------------------------
#
def connected_component(tarray, t):

    from chimerax import surface
    tlist = surface.connected_triangles(tarray, t)
    vlist = surface.triangle_vertices(tarray, tlist)
    return vlist, tlist

# -----------------------------------------------------------------------------
#
def color_blob(surface, vlist, rgba8):

    vc = surface.vertex_colors
    if vc is None:
        varray, tarray = surface.vertices, surface.triangles
        n = len(varray)
        from numpy import empty, uint8
        vc = empty((n, 4), uint8)
        vc[:] = surface.color

    vc[vlist,:] = rgba8

    surface.vertex_colors = vc

# -----------------------------------------------------------------------------
#
def blob_geometry(surface, vlist, tlist):

    varray, tarray = surface.vertices, surface.triangles
    
    vbarray = varray.take(vlist, axis=0)
    tbarray = tarray.take(tlist, axis=0)

    # Remap vertex indices in triangle array to use new vertex list.
    from numpy import zeros, intc, arange
    vmap = zeros(varray.shape[0], intc)
    vmap[vlist] = arange(len(vlist), dtype = intc)
    tbarray[:,:] = vmap[tbarray]
        
    return vbarray, tbarray

# -------------------------------------------------------------------------
#
def principle_axes_box(varray, tarray):

  from chimerax import surface
  weights = surface.vertex_areas(varray, tarray)
  from chimerax.std_commands.measure_inertia import moments_of_inertia
  axes, d2e, center = moments_of_inertia([(varray, weights)])
  from chimerax.geometry import Place, point_bounds
  axes_points = Place(axes = axes).inverse().transform_points(varray)
  bounds = point_bounds(axes_points)
  return axes, bounds
      
# -------------------------------------------------------------------------
#
from chimerax.core.models import Surface
class BlobOutlineBox(Surface):
    def __init__(self, session):
        Surface.__init__(self, 'blob outline box', session)
        self.pickable = False

    @classmethod
    def create_box(cls, session, axes, bounds, rgba = (0,255,0,255)):
        '''Create a new outline box model.'''
        bob = cls(session)
        bob._create_outline(axes, bounds, rgba)
        return bob
        
    def _create_outline(self, axes, bounds, rgba = (0,255,0,255)):        
        # Compute corners
        vlist = []
        b = (bounds.xyz_min, bounds.xyz_max)
        for ci in ((0,0,0),(0,0,1),(0,1,0),(0,1,1),
                   (1,0,0),(1,0,1),(1,1,0),(1,1,1)):
            v = [sum([b[ci[a]][a]*axes[a][c] for a in (0,1,2)]) for c in (0,1,2)]
            vlist.append(v)
        tlist = ((0,4,5), (5,1,0), (0,2,6), (6,4,0),
                 (0,1,3), (3,2,0), (7,3,1), (1,5,7),
                 (7,6,2), (2,3,7), (7,5,4), (4,6,7))
        e = 2 + 1    # Bit mask, edges are bits 4,2,1
        hide_diagonals = (e,e,e,e,e,e,e,e,e,e,e,e)

        s = self
        s.display = True
        s.display_style = s.Mesh
        s.use_lighting = False
        s.outline_box = True # Do not cap clipped outline box.
        from numpy import array, float32, int32
        s.set_geometry(array(vlist,float32), None, array(tlist,int32))
        s.edge_mask = hide_diagonals
        s.color = rgba

    def take_snapshot(self, session, flags):
        return Surface.save_geometry(self, session, flags)

    @classmethod
    def restore_snapshot(cls, session, data):
        return BlobOutlineBox(session).restore_geometry(session, data)
    
# -------------------------------------------------------------------------
#
def boundary_lengths(varray, tarray):

    from chimerax import surface
    loops = surface.boundary_loops(tarray)
    return [loop_length(loop, varray) for loop in loops]

# -------------------------------------------------------------------------
#
def loop_length(vindices, varray):

    p = varray.take(vindices, axis=0)
    p0 = p[0,:].copy()
    p[:-1,:] -= p[1:,:]
    p[-1,:] -= p0
    p *= p
    from numpy import sqrt
    d = sqrt(p.sum(axis=1)).sum()
    return d

# -------------------------------------------------------------------------
#
def measure_blob(session, surface, triangle_number, color = None,
                 outline = False, outline_color = (0,255,0,255),
                 report_size = True):

    nt = len(surface.triangles)
    if triangle_number < 0 or triangle_number >= nt:
        from chimerax.core.errors import UserError
        raise UserError('Triangle number %d out of range for surface #%s with %d triangles'
                        % (triangle_number, surface.id_string, nt))
    
    vlist, tlist = connected_component(surface.triangles, triangle_number)
    if color is not None:
        from chimerax.core.colors import Color
        rgba = color.uint8x4() if isinstance(color, Color) else color
        color_blob(surface, vlist, rgba)

    log = session.logger if report_size else None
    axes, bounds, msg = blob_size(surface, vlist, tlist, log = log)
        
    pbp = pick_blobs_panel(session, create = False)
    if pbp:
        pbp.message(msg)

    if outline:
        from chimerax.core.colors import Color
        rgba = outline_color.uint8x4() if isinstance(outline_color, Color) else outline_color
        bob = BlobOutlineBox.create_box(session, axes, bounds, rgba = rgba)
        surface.parent.add([bob])

# -------------------------------------------------------------------------
#
def register_measure_blob_command(logger):
    from chimerax.core.commands import CmdDesc, register, SurfaceArg, IntArg, ColorArg, BoolArg
    desc = CmdDesc(
        required = [('surface', SurfaceArg),],
        keyword = [('triangle_number', IntArg),
                   ('color', ColorArg),
                   ('outline', BoolArg),
                   ('outline_color', ColorArg),
                   ('report_size', BoolArg)],
        required_arguments = ['triangle_number'],
        synopsis = 'Measure and color connected parts of surfaces'
    )
    register('measure blob', desc, measure_blob, logger=logger)

# -------------------------------------------------------------------------
#
from chimerax.mouse_modes import MouseMode
class PickBlobs(MouseMode):
    name = 'pick blobs'
    icon_file = 'pickblobs.png'

    def __init__(self, session):
        MouseMode.__init__(self, session)
        
    @property
    def settings(self):
        return pick_blobs_panel(self.session)

    def enable(self):
        self.settings.show()
        
    def mouse_down(self, event):
        x,y = event.position()
        view = self.session.main_view
        pick = view.picked_object(x, y, max_transparent_layers = 0)
        self._pick_blob(pick)
    
    def _pick_blob(self, pick):
        from chimerax.map.volume import PickedMap
        if not isinstance(pick , PickedMap) or not hasattr(pick, 'triangle_pick'):
            return

        tpick = pick.triangle_pick
        t = tpick.triangle_number
        surface = tpick.drawing()

        cmd = 'measure blob #!%s triangle %d'  % (surface.id_string, t)
        settings = self.settings
        if settings.color_blob:
            cmd += ' color %s' % hex_color(settings.blob_color)
            if settings.change_color:
                settings.new_color()
        if settings.show_box:
            cmd += ' outline true'
            if tuple(settings.box_color) != (0,255,0,255):
                cmd += ' outlineColor %s' % hex_color(settings.box_color)

        from chimerax.core.commands import run
        run(surface.session, cmd)

    def vr_press(self, event):
        # Virtual reality hand controller button press.
        pick = event.picked_object(self.view)
        self._pick_blob(pick)

def hex_color(rgba8):
    return '#%02x%02x%02x%02x' % tuple(rgba8)

# -----------------------------------------------------------------------------
# Panel for coloring connected pieces of a surface chosen with mouse.
#
from chimerax.core.tools import ToolInstance
class PickBlobSettings(ToolInstance):
    help = "help:user/tools/measureblobs.html"

    def __init__(self, session, tool_name):

        self._default_color = (0,0,204,255)
        self._default_box_color = (0,255,0,255)
        
        ToolInstance.__init__(self, session, tool_name)

        self.display_name = 'Measure and Color Blobs'

        from chimerax.ui import MainToolWindow
        tw = MainToolWindow(self)
        self.tool_window = tw
        parent = tw.ui_area

        from Qt.QtWidgets import QVBoxLayout, QHBoxLayout, QFrame, QCheckBox, QLabel, QPushButton, QSizePolicy
        from Qt.QtCore import Qt

        layout = QVBoxLayout(parent)
        layout.setContentsMargins(0,0,0,0)
        layout.setSpacing(0)
        parent.setLayout(layout)

        cf = QFrame(parent)
#        cf.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        layout.addWidget(cf)
        
        clayout = QHBoxLayout(cf)
        clayout.setContentsMargins(0,0,0,0)
        clayout.setSpacing(10)

        self._color_blob = cb = QCheckBox('Color blob', cf)
        cb.setCheckState(Qt.Checked)
        clayout.addWidget(cb)
        from chimerax.ui.widgets import ColorButton
        self._blob_color = cbut = ColorButton(cf, max_size = (16,16), has_alpha_channel = True)
        cbut.color = self._default_color
        clayout.addWidget(cbut)
        clayout.addSpacing(10)

        self._change_color = cc = QCheckBox('Change color automatically', cf)
        cc.setCheckState(Qt.Checked)
        clayout.addWidget(cc)
        clayout.addStretch(1)    # Extra space at end

        af = QFrame(parent)
#        af.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        layout.addWidget(af)
        
        aflayout = QHBoxLayout(af)
        aflayout.setContentsMargins(0,0,0,0)
        aflayout.setSpacing(10)

        self._show_box = bb = QCheckBox('Show principal axes box', af)
        bb.setCheckState(Qt.Checked)
        aflayout.addWidget(bb)
        self._box_color = bc = ColorButton(af, max_size = (16,16), has_alpha_channel = True)
        bc.color = self._default_box_color
        aflayout.addWidget(bc)
        aflayout.addSpacing(10)
        
        eb = QPushButton('Erase boxes', af)
        eb.clicked.connect(self._erase_boxes)
        aflayout.addWidget(eb)
        aflayout.addStretch(1)    # Extra space at end
        
        self._message_label = ml = QLabel(parent)
        layout.addWidget(ml)
        layout.addStretch(1)    # Extra space at end

        tw.manage(placement="side")

    @classmethod
    def get_singleton(self, session, create=True):
        from chimerax.core import tools
        return tools.get_singleton(session, PickBlobSettings, 'Measure and Color Blobs', create=create)

    def show(self):
        self.tool_window.shown = True

    def hide(self):
        self.tool_window.shown = False

    def message(self, text):
        self._message_label.setText(text)

    @property
    def color_blob(self):
        from Qt.QtCore import Qt
        return self._color_blob.checkState() == Qt.Checked
    
    @property
    def blob_color(self):
        return self._blob_color.color

    @property
    def change_color(self):
        from Qt.QtCore import Qt
        return self._change_color.checkState() == Qt.Checked
        
    def new_color(self):
        from random import random as r
        from chimerax.geometry import normalize_vector
        rgba = tuple(normalize_vector((r(), r(), r()))) + (1,)
        self._blob_color.color = rgba

    @property
    def show_box(self):
        from Qt.QtCore import Qt
        return self._show_box.checkState() == Qt.Checked
    
    @property
    def box_color(self):
        return self._box_color.color
    
    def _erase_boxes(self):
        models = self.session.models
        outlines = [m for m in models.list() if isinstance(m, BlobOutlineBox)]
        models.close(outlines)

# -------------------------------------------------------------------------
#
def pick_blobs_panel(session, create = True):
    return PickBlobSettings.get_singleton(session, create)

# -------------------------------------------------------------------------
#
def blob_size(surface, vlist, tlist, log = None):

  # Report enclosed volume and area
  varray, tarray = blob_geometry(surface, vlist, tlist)
  from chimerax.surface import enclosed_volume, surface_area
  v, h = enclosed_volume(varray, tarray)
  blen = None
  if v == None:
    vstr = 'undefined (non-oriented surface)'
  else:
    vstr = '%.5g' % v
    if h > 0:
      vstr += ' (%d holes)' % h
      blen = boundary_lengths(varray, tarray)
  area = surface_area(varray, tarray)

  axes, bounds = principle_axes_box(varray, tarray)
  size = bounds.size()

  vstr = 'volume = %s' % (vstr,)
  astr = 'area = %.5g' % (area,)
  szstr = 'size = %.5g %.5g %.5g' % tuple(size)
  stats = (vstr, astr, szstr)
  if blen:
    blstr = 'boundary = %.5g' % sum(blen)
    stats += (blstr,)
    if len(blen) > 1:
      blen.sort()
      blen.reverse()
      llstr = 'loop lengths = ' + ', '.join('%.5g' % b for b in blen[:8])
      if len(blen) > 8:
        llstr += ', ...'
      stats += (llstr,)

  name = surface.name
  if name == 'surface':
      name = surface.parent.name
  msg = ('Surface %s #%s blob:\n  %s' %
         (name, surface.id_string, '\n  '.join(stats)))

  if log:
      log.info(msg + '\n')
      log.status(', '.join(stats))

  return axes, bounds, msg

# -------------------------------------------------------------------------
#
def register_mousemode(session):
    mm = session.ui.mouse_modes
    mm.add_mode(PickBlobs(session))
