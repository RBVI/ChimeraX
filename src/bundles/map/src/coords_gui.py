# vim: set expandtab ts=4 sw=4:

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

# -----------------------------------------------------------------------------
# User interface for placement of volume data array in xyz coordinate space.
# Displays origin, voxel size, cell angles (for skewed x-ray data), rotation.
#
from chimerax.core.tools import ToolInstance
class CoordinatesPanel(ToolInstance):

  def __init__(self, session, tool_name):

    ToolInstance.__init__(self, session, tool_name)

    from chimerax.ui import MainToolWindow
    tw = MainToolWindow(self)
    self.tool_window = tw
    parent = tw.ui_area

    from Qt.QtWidgets import QVBoxLayout, QLabel
    layout = QVBoxLayout(parent)
    layout.setContentsMargins(0,0,0,0)
    layout.setSpacing(0)
    parent.setLayout(layout)

    # Heading
    heading = QLabel('Placement of data array in x,y,z coordinate space:', parent)
    layout.addWidget(heading)
    
    # Make menus to choose molecule and map for fitting
    mf = self._create_map_menu(parent)
    layout.addWidget(mf)

    # GUI for origin, step, angles, axis settings.
    options = self._create_parameters_gui(parent)
    layout.addWidget(options)

    # Apply button
#    bf = self._create_action_buttons(parent)
#    layout.addWidget(bf)
  
    # Status line
    self._status_label = sl = QLabel(parent)
    layout.addWidget(sl)
        
    layout.addStretch(1)    # Extra space at end

    self._update_gui_values()

    tw.manage(placement="side")

  # ---------------------------------------------------------------------------
  #
  def _create_map_menu(self, parent):

    from Qt.QtWidgets import QFrame, QHBoxLayout, QLabel
        
    mf = QFrame(parent)
    mlayout = QHBoxLayout(mf)
    mlayout.setContentsMargins(0,0,0,0)
    mlayout.setSpacing(10)
        
    fl = QLabel('Map', mf)
    mlayout.addWidget(fl)

    from chimerax.map import Volume
    from chimerax.ui.widgets import ModelMenuButton
    self._map_menu = mm = ModelMenuButton(self.session, class_filter = Volume)
    vlist = self.session.models.list(type = Volume)
    vdisp = [v for v in vlist if v.display]
    if vdisp:
      mm.value = vdisp[0]
    elif vlist:
      mm.value = vlist[0]
    mm.value_changed.connect(self._update_gui_values)
    mlayout.addWidget(mm)
    mlayout.addStretch(1)    # Extra space at end

    return mf

  # ---------------------------------------------------------------------------
  #
  def _create_parameters_gui(self, parent):

    from Qt.QtWidgets import QFrame, QVBoxLayout
        
    f = QFrame(parent)
    layout = QVBoxLayout(f)
    layout.setContentsMargins(0,0,0,0)
    layout.setSpacing(2)

    from chimerax.ui.widgets import EntriesRow
    
    oe = EntriesRow(f, 'Origin index', '',
                    ('center', self._center_origin),
                    ('reset', self._reset_origin))
    self._origin = o = oe.values[0]
    o.return_pressed.connect(self._origin_changed)
    import sys
    if sys.platform == 'darwin':
      layout.addSpacing(-12)	# Reduce Mac push button vertical space.
    
    vse = EntriesRow(f, 'Voxel size', '',
                     ('reset', self._reset_voxel_size))
    self._voxel_size = vs = vse.values[0]
    vs.return_pressed.connect(self._voxel_size_changed)
    if sys.platform == 'darwin':
      layout.addSpacing(-8)	# Reduce Mac push button vertical space.
    
    cae = EntriesRow(f, 'Cell angles', '')
    self._cell_angles = ca = cae.values[0]
    ca.return_pressed.connect(self._cell_angles_changed)
    
    rae = EntriesRow(f, 'Rotation axis', '', 'angle', 0.0)
    self._rotation_axis, self._rotation_angle = ra, ran = rae.values
    ra.return_pressed.connect(self._rotation_changed)
    ran.return_pressed.connect(self._rotation_changed)

    return f

  # ---------------------------------------------------------------------------
  #
  @classmethod
  def get_singleton(self, session, create=True):
    from chimerax.core import tools
    return tools.get_singleton(session, CoordinatesPanel, 'Map Coordinates', create=create)

  # ---------------------------------------------------------------------------
  #
  def _status(self, message, log = False):
    self._status_label.setText(message)
    if log:
      self.session.logger.info(message)

  # ---------------------------------------------------------------------------
  #
  @property
  def _map(self):
      return self._map_menu.value
  
  # ---------------------------------------------------------------------------
  #
  def _update_gui_values(self):

    v = self._map
    if v is None:
      self._origin.value = ''
      self._voxel_size.value = ''
      self._cell_angles.value = ''
      self._rotation_axis.value = ''
      self._rotation_angle.value = 0
    else:
      data = v.data
      from .volume_viewer import vector_value_text
      self._origin.value = vector_value_text(data.xyz_to_ijk((0,0,0)))
      self._voxel_size.value = vector_value_text(data.step)
      self._cell_angles.value = vector_value_text(data.cell_angles)
      from chimerax.geometry import Place
      from numpy import transpose
      axis, angle = Place(axes = transpose(data.rotation)).rotation_axis_and_angle()
      from .volume_viewer import float_format
      self._rotation_axis.value = ' '.join([float_format(x,5) for x in axis])
      self._rotation_angle.value = angle

    self._status('')
    
  # ---------------------------------------------------------------------------
  #
  def _origin_changed(self):

    v = self._map
    if v is None:
        return

    data = v.data
    orig = self._origin.value
    dorigin = data.xyz_to_ijk((0,0,0))
    from .volume_viewer import vector_value
    origin = vector_value(orig, dorigin, allow_singleton = True)
    if origin is None:
       self._status('Invalid origin value. Must be one number or 3 numbers separated by spaces.')
    elif tuple(origin) != tuple(dorigin):
      xyz_origin = [x0-x for x0,x in zip(data.ijk_to_xyz((0,0,0)),data.ijk_to_xyz(origin))]
      data.set_origin(xyz_origin)
      self._redraw_regions(data)
      self._status('Set origin %s' % orig)

  # ---------------------------------------------------------------------------
  #
  def _voxel_size_changed(self):

    v = self._map
    if v is None:
        return

    data = v.data
    
    vs = self._voxel_size.value
    from .volume_viewer import vector_value
    vsize = vector_value(vs, data.step, allow_singleton = True)
    if vsize is None:
      self._status('Invalid voxel size. Must be one number or 3 numbers separated by spaces.')
    elif vsize != data.step:
      if min(vsize) <= 0:
        self._status('Voxel size must be positive, got %g,%g,%g.' % vsize)
        return
      # Preserve index origin.
      index_origin = data.xyz_to_ijk((0,0,0))
      data.set_step(vsize)
      xyz_origin = [x0-x for x0,x in zip(data.ijk_to_xyz((0,0,0)),data.ijk_to_xyz(index_origin))]
      data.set_origin(xyz_origin)
      self._redraw_regions(data)
      self._status('Set voxel size %s' % vs)

  # ---------------------------------------------------------------------------
  #
  def _cell_angles_changed(self):

    v = self._map
    if v is None:
        return

    data = v.data

    ca = self._cell_angles.value
    from .volume_viewer import vector_value
    cell_angles = vector_value(ca, data.cell_angles, allow_singleton = True)
    if cell_angles is None:
      self._status('Invalid cell angles. Must be one number or 3 numbers separated by spaces.')
      return
    if [a for a in cell_angles if a <= 0 or a >= 180]:
      self._status('Cell angles must be between 0 and 180 degrees')
      return
    if cell_angles != data.cell_angles:
      data.set_cell_angles(cell_angles)
      self._redraw_regions(data)
      self._status('Set cell angles %s' % ca)

  # ---------------------------------------------------------------------------
  #
  def _rotation_changed(self):

    v = self._map
    if v is None:
        return

    data = v.data

    if data.rotation == ((1,0,0),(0,1,0),(0,0,1)):
      axis, angle = (0,0,1), 0
    else:
      from chimerax.geometry import Place
      axis, angle = Place(axes = data.rotation).rotation_axis_and_angle()

    rax = self._rotation_axis.value
    from .volume_viewer import vector_value
    naxis = vector_value(rax, axis)
    if naxis is None:
      self._status('Invalid rotation axis.  Must be 3 numbers separated by spaces.')
      return
    if tuple(naxis) == (0,0,0):
      self._status('Rotation axis must be non-zero')
      return

    nangle = self._rotation_angle.value

    if tuple(naxis) != tuple(axis) or nangle != angle:
      from chimerax.geometry import rotation
      r = rotation(naxis, nangle)
      data.set_rotation(r.matrix[:,:3])
      # Have to change xyz origin for index origin to remain the same.
      self._origin_changed()
      self._redraw_regions(data)
      self._status('Set rotation axis %s, angle %.3g' % (rax, nangle))

  # ---------------------------------------------------------------------------
  #
  def _redraw_regions(self, data):

    from .volume import Volume
    vlist = [v for v in self.session.models.list(type = Volume) if v.data is data]
    for v in vlist:
      if v.shown():
        v.show()

  # ---------------------------------------------------------------------------
  #
  def _center_origin(self):

    v = self._map
    if v is None:
      return

    data = v.data
    index_origin = [0.5*(s-1) for s in data.size]
    xyz_origin = [x0-x for x0,x in zip(data.ijk_to_xyz((0,0,0)),data.ijk_to_xyz(index_origin))]
    data.set_origin(xyz_origin)
    self._redraw_regions(data)
    self._update_gui_values()

  # ---------------------------------------------------------------------------
  #
  def _reset_origin(self):

    v = self._map
    if v is None:
      return

    data = v.data
    # To get original index origin need to use original xyz origin, original step,
    # and original rotation (which we don't know so we assume original rotation is identity).
    data.set_origin(data.original_origin)
    step = data.step
    data.set_step(data.original_step)
    rotation = data.rotation
    data.set_rotation(((1,0,0),(0,1,0),(0,0,1)))
    index_origin = data.xyz_to_ijk((0,0,0))
    data.set_step(step)         # Restore step
    data.set_rotation(rotation) # Restore rotation
    xyz_origin = [x0-x for x0,x in zip(data.ijk_to_xyz((0,0,0)),data.ijk_to_xyz(index_origin))]
    data.set_origin(xyz_origin)
    self._redraw_regions(data)
    self._update_gui_values()

  # ---------------------------------------------------------------------------
  #
  def _reset_voxel_size(self):

    v = self._map
    if v == None:
      return

    data = v.data
    # Preserve index origin.
    index_origin = data.xyz_to_ijk((0,0,0))
    data.set_step(data.original_step)
    xyz_origin = [x0-x for x0,x in zip(data.ijk_to_xyz((0,0,0)),data.ijk_to_xyz(index_origin))]
    data.set_origin(xyz_origin)
    self._redraw_regions(data)
    self._update_gui_values()

# -----------------------------------------------------------------------------
#
def show_coords_panel(session):

  return CoordinatesPanel.get_singleton(session)

