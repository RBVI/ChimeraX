
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

# -----------------------------------------------------------------------------
# Manages surface and volume display for a region of a data set.
# Holds surface and solid thresholds, color, and transparency and brightness
# factors.
#
from chimerax.core.models import Model
class Volume(Model):
  '''
  A Volume is a rendering of a 3-d image GridData object.  It includes
  color, display styles including surface, mesh and grayscale, contouring levels,
  brightness and transparency for grayscale rendering, region bounds for display
  a subregion including single plane display, subsampled display of every Nth data
  value along each axis, outline box display.
  '''
  def __init__(self, session, data, region = None, rendering_options = None):
    '''Supported API. Create a volume model from a GridData instance.'''
    
    Model.__init__(self, data.name, session)

    self.session = session

    ds = default_settings(session)
    self.pickable = ds['pickable']

    self.change_callbacks = []

    self.data = data
    data.add_change_callback(self.data_changed_cb)
    self.path = data.path

    if region is None:
      region = full_region(data.size)
    self.region = clamp_region(region, data.size)

    if rendering_options is None:
      rendering_options = Rendering_Options()
    self.rendering_options = rendering_options

    self.message_cb = session.logger.status
    
    self.matrix_stats = None
    self.matrix_id = 1          # Incremented when shape or values change.

    rlist = Region_List()
    ijk_min, ijk_max = self.region[:2]
    rlist.insert_region(ijk_min, ijk_max)
    self.region_list = rlist

    self._channels = None	# MapChannels object
    
    self.representation = 'surface'

    self.solid = None
    self._keep_displayed_data = None

    self.initialized_thresholds = False

    # Surface display parameters
    self._surfaces = []				# VolumeSurface instances
    self.surface_brightness_factor = 1
    self.transparency_factor = 0                # for surface/mesh
    self.outline_box = Outline_Box(self)

    # Solid display parameters
    self.solid_levels = []                      # list of (threshold, scale)
    self.solid_colors = []
    self.transparency_depth = 0.5               # for solid
    self.solid_brightness_factor = 1

    self.default_rgba = data.rgba if data.rgba else (.7,.7,.7,1)


#    from chimera import addModelClosedCallback
#    addModelClosedCallback(self, self.model_closed_cb)

#    from chimera import triggers
#    h = triggers.addHandler('SurfacePiece', self.surface_piece_changed_cb, None)
#    self.surface_piece_change_handler = h
    self.surface_piece_change_handler = None

  # ---------------------------------------------------------------------------
  #
  def message(self, text, **kw):

    if self.message_cb:
      self.message_cb(text, **kw)

  # ---------------------------------------------------------------------------
  # Update data name when model name changes, so it gets written to cmap files.
  #
  def _set_data_name(self, name):
    if hasattr(self, 'data') and name != self.data.name:
      self.data.name = name
    Model.name.fset(self, name)
  name = property(Model.name.fget, _set_data_name)
    
  # ---------------------------------------------------------------------------
  #
  def full_name(self):

    return self.name
    
  # ---------------------------------------------------------------------------
  #
  def name_with_id(self):

    return '%s #%s' % (self.name, self.id_string)

  # ---------------------------------------------------------------------------
  #
  def add_volume_change_callback(self, cb):

    self.change_callbacks.append(cb)

  # ---------------------------------------------------------------------------
  #
  def remove_volume_change_callback(self, cb):

    self.change_callbacks.remove(cb)


  # ---------------------------------------------------------------------------
  #
  def added_to_session(self, session):
    if len(session.models.list()) == 1:
      from chimerax.std_commands.lighting import lighting
      lighting(session, 'full')	# Use full lighting for initial map display

  # ---------------------------------------------------------------------------
  #
  def call_change_callbacks(self, change_types):

    if isinstance(change_types, str):
      change_types = [change_types]

    for cb in self.change_callbacks:
      for ct in change_types:
        cb(self, ct)
  
  # ---------------------------------------------------------------------------
  #
  def replace_data(self, data):

    d = self.data
    cc = (data.origin != d.origin or
          data.step != d.step or
          data.cell_angles != d.cell_angles)
    dc = self.data_changed_cb
    d.remove_change_callback(dc)
    self.data = data
    data.add_change_callback(dc)
    dc('values changed')
    if cc:
      dc('coordinates changed')
    
  # ---------------------------------------------------------------------------
  #
  @property
  def surfaces(self):
    '''Supported API.  Return a list of VolumeSurface instances for this Volume.'''
    return self._surfaces
  
  # ---------------------------------------------------------------------------
  #
  def add_surface(self, level, rgba = None):
    '''Supported API.  Create and add a new VolumeSurface with specified contour level and color.'''
    ses = self.session
    s = VolumeSurface(self, level, rgba)
    self._surfaces.append(s)
    if self.id is None:
      self.add([s])
    else:
      ses.models.add([s], parent = self)
    self._drawings_need_update()
    return s
  
  # ---------------------------------------------------------------------------
  #
  def remove_surfaces(self, surfaces = None):
    '''
    Supported API.  Remove a list of VolumeSurface instances from this Volume.
    If surfaces is None then all current surfaces are removed.
    '''
    surfs = self._surfaces if surfaces is None else surfaces
    if self.id is None:
      self.remove_drawings(surfs)
    else:
      self.session.models.close(surfs)

  # ---------------------------------------------------------------------------
  #
  @property
  def minimum_surface_level(self):
    return min((s.level for s in self.surfaces), default = None)
  @property
  def maximum_surface_level(self):
    return max((s.level for s in self.surfaces), default = None)
  
  # ---------------------------------------------------------------------------
  #
  def set_parameters(self, **kw):
    '''
    Set volume display parameters.  The following keyword parameters are valid.
  
      surface_levels
      surface_colors              (rgb or rgba values)
      surface_brightness_factor
      transparency_factor
      solid_levels
      solid_colors                (rgb or rgba values)
      transparency_depth
      solid_brightness_factor
  
      Any rendering option attribute names can also be used.
    '''

    parameters = ('surface_levels',
                  'surface_colors',
                  'surface_brightness_factor',
                  'transparency_factor',
                  'solid_levels',
                  'solid_colors',
                  'solid_brightness_factor',
                  'transparency_depth',
                  'default_rgba',
                  )

    def rgb_to_rgba(color):
      if len(color) == 3:
        return tuple(color) + (1,)
      return color

    if 'surface_colors' in kw:
      kw['surface_colors'] = [rgb_to_rgba(c) for c in kw['surface_colors']]
    if 'solid_colors' in kw:
      kw['solid_colors'] = [rgb_to_rgba(c) for c in kw['solid_colors']]

    if ('surface_levels' in kw and
        not 'surface_colors' in kw and
        len(kw['surface_levels']) != len(self.surfaces)):
      kw['surface_colors'] = [self.default_rgba] * len(kw['surface_levels'])
    if ('solid_levels' in kw and
        not 'solid_colors' in kw and
        len(kw['solid_levels']) != len(self.solid_colors)):
      rgba = saturate_rgba(self.default_rgba)
      kw['solid_colors'] = [rgba] * len(kw['solid_levels'])

    if 'default_rgba' in kw:
      self.default_rgba = kw['default_rgba'] = rgb_to_rgba(kw['default_rgba'])

    # Make copies of lists.
    for param in ('surface_levels', 'surface_colors',
                  'solid_levels', 'solid_colors'):
      if param in kw:
        kw[param] = list(kw[param])

    threaded_surf_calc = kw.get('threaded_surface_calculation', False)
    for param in parameters:
      if param in kw:
        values = kw[param]
        if param == 'surface_levels':
          if len(values) == len(self.surfaces):
            for s,level in zip(self.surfaces, values):
              s.set_level(level, use_thread = threaded_surf_calc)
          else:
            self.remove_surfaces()
            for level in values:
              s = self.add_surface(level)
              s.set_level(level, use_thread = threaded_surf_calc)
        elif param == 'surface_colors':
          if len(values) == len(self.surfaces):
            for s,color in zip(self.surfaces, values):
              s.rgba = color
          else:
            raise ValueError('Number of surface colors (%d) does not match number of surfaces (%d)'
                             % (len(values), len(self.surfaces)))
        else:
          setattr(self, param, values)

    # Update rendering options.
    option_changed = False
    if ('orthoplanes_shown' in kw and not 'box_faces' in kw and
        true_count(kw['orthoplanes_shown']) > 0):
      # Turn off box faces if orthoplanes enabled.
      kw['box_faces'] = False
    ro = self.rendering_options
    box_faces_toggled = ('box_faces' in kw and kw['box_faces'] != ro.box_faces)
    orthoplanes_toggled = ('orthoplanes_shown' in kw and
                           kw['orthoplanes_shown'] != ro.orthoplanes_shown and
                           (true_count(kw['orthoplanes_shown']) == 0 or
                            true_count(ro.orthoplanes_shown) == 0))
    adjust_step = (self.representation == 'solid' and
                   (box_faces_toggled or orthoplanes_toggled))
    for k,v in kw.items():
      if k in ro.__dict__:
        setattr(self.rendering_options, k, v)
        option_changed = True
    if adjust_step:
      r = self.region
      self.new_region(r[0], r[1], r[2], adjust_step = True)

    if 'surface_levels' in kw or 'solid_levels' in kw:
      self.call_change_callbacks('thresholds changed')

    if ('surface_colors' in kw or 'solid_colors' in kw or
        'surface_brightness_factor' in kw or 'transparency_factor' in kw or
        'solid_brightness_factor' in kw or 'transparency_depth' in kw):
      self.call_change_callbacks('colors changed')

    if option_changed:
      self.call_change_callbacks('rendering options changed')

    if kw:
      self._drawings_need_update()

  # ---------------------------------------------------------------------------
  #
  def set_color(self, rgba):
    for s in self.surfaces:
      s.rgba = rgba
      s.vertex_colors = None
    self.solid_colors = [rgba]*len(self.solid_levels)
    self._drawings_need_update()
    self.call_change_callbacks('colors changed')

  # ---------------------------------------------------------------------------
  #
  def _get_single_color(self):
    rep = self.representation
    from chimerax.core.colors import rgba_to_rgba8
    if rep in ('surface', 'mesh'):
      surfs = self.surfaces
      if surfs:
        return min(surfs, key = lambda s: s.level).color
    elif rep == 'solid':
      lev = self.solid_levels
      if lev:
        from numpy import argmin
        i = argmin([v for v,b in lev])
        return rgba_to_rgba8(self.solid_colors[i])
    drgba = self.data.rgba
    if drgba:
      return rgba_to_rgba8(drgba)
    return None
  def _set_single_color(self, color):
    from chimerax.core.colors import rgba8_to_rgba
    self.set_color(rgba8_to_rgba(color))
  single_color = property(_get_single_color, _set_single_color)

  
  # ---------------------------------------------------------------------------
  #
  def set_transparency(self, alpha):
    '''Alpha values in range 0-255. Only changes current style (surface/mesh or solid).'''
    a1 = alpha/255
    if self.representation in ('surface', 'mesh'):
      for s in self.surfaces:
        r,g,b,a = s.rgba
        s.rgba = (r,g,b,a1)
    else:
      self.set_parameters(solid_colors = [(r,g,b,a1) for r,g,b,a in self.solid_colors])
    self._drawings_need_update()
    
    # Update transparency on per-vertex coloring
    for s in self.surfaces:
      vc = s.vertex_colors
      if vc is not None:
        vc[:,3] = alpha
        s.vertex_colors = vc

  # ---------------------------------------------------------------------------
  #
  def new_region(self, ijk_min = None, ijk_max = None, ijk_step = None,
                 adjust_step = True, adjust_voxel_limit = True):
    '''
    Set new display region and optionally shows it.
    '''

    if ijk_min is None:
      ijk_min = self.region[0]
    if ijk_max is None:
      ijk_max = self.region[1]

    # Make bounds integers.
    from math import ceil, floor
    ijk_min = [int(ceil(x)) for x in ijk_min]
    ijk_max = [int(floor(x)) for x in ijk_max]

    # Make it lie within dota bounds.
    (ijk_min, ijk_max) = clamp_region((ijk_min, ijk_max), self.data.size)

    # Determine ijk_step.
    if ijk_step == None:
      if self.region:
        ijk_step = self.region[2]
      else:
        ijk_step = (1,1,1)
    else:
      ijk_step = [int(ceil(x)) for x in ijk_step]

    # Adjust ijk_step to meet voxel limit.
    ro = self.rendering_options
    fpa = faces_per_axis(self.representation, ro.box_faces,
                         ro.any_orthoplanes_shown())
    adjusted_ijk_step = ijk_step_for_voxel_limit(ijk_min, ijk_max, ijk_step,
                                                 fpa, ro.limit_voxel_count,
                                                 ro.voxel_limit)
    if adjust_step:
      ijk_step = adjusted_ijk_step
    elif adjust_voxel_limit and tuple(ijk_step) != tuple(adjusted_ijk_step):
      # Change automatic step adjustment voxel limit.
      vc = subarray_size(ijk_min, ijk_max, ijk_step, fpa)
      ro.voxel_limit = (1.01*vc) / (2**20)  # Mvoxels rounded up for gui value
      self.call_change_callbacks('voxel limit changed')

    region = (ijk_min, ijk_max, ijk_step)
    if self.same_region(region, self.region):
      return False

    self.region = region
    self.matrix_changed()

    self._drawings_need_update()
    
    self.call_change_callbacks('region changed')

    return True

  # ---------------------------------------------------------------------------
  #
  def full_region(self):
    return full_region(self.data.size)

  # ---------------------------------------------------------------------------
  #
  def is_full_region(self, region = None):

    if region is None:
      region = self.region
    elif region == 'all':
      return True
    ijk_min, ijk_max,ijk_step = region
    dmax = tuple([s-1 for s in self.data.size])
    full = (tuple(ijk_min) == (0,0,0) and
            tuple(ijk_max) == dmax and
            tuple(ijk_step) == (1,1,1))
    return full

  # ---------------------------------------------------------------------------
  # Either data values or subregion has changed.
  #
  def matrix_changed(self):

    self.matrix_stats = None
    self.matrix_id += 1
    self._drawings_need_update()
      
  # ---------------------------------------------------------------------------
  # Handle ijk_min, ijk_max, ijk_step as lists or tuples.
  #
  def same_region(self, r1, r2):

    for i in range(3):
      if tuple(r1[i]) != tuple(r2[i]):
        return False
    return True
    
  # ---------------------------------------------------------------------------
  #
  @property
  def showing_one_plane(self):
    ijk_min, ijk_max, ijk_step = self.region
    for i0, i1, step in zip(ijk_min, ijk_max, ijk_step):
      if i1//step - i0//step == 0:
        return True
    return False
  
  # ---------------------------------------------------------------------------
  #
  def has_thresholds(self):

    return len(self.surfaces) > 0 and len(self.solid_levels) > 0

  # ---------------------------------------------------------------------------
  #
  def initialize_thresholds(self, first_time_only = True,
                            vfrac = (0.01, 0.90), mfrac = None,
                            replace = False):
    '''
    Set default initial surface and solid style rendering thresholds.
    The thresholds are only changed the first time this method is called or if
    the replace option is True.  Returns True if thresholds are changed.
    '''
    if not replace:
      if first_time_only and self.initialized_thresholds:
        return False
      if self.has_thresholds():
        self.initialized_thresholds = True
        return False

#     from chimera import CancelOperation
    try:
      s = self.matrix_value_statistics()
    except CancelOperation:
      return False

    polar = (hasattr(self.data, 'polar_values') and self.data.polar_values)

    if replace or len(self.surfaces) == 0:
      if mfrac is None:
        v = s.rank_data_value(1-vfrac[0])
      else:
        v = s.mass_rank_data_value(1-mfrac[0])
      rgba = self.default_rgba
      if polar:
        levels = [-v,v]
        neg_rgba = tuple([1-c for c in rgba[:3]] + [rgba[3]])
        colors = [neg_rgba,rgba]
      else:
        levels = [v]
        colors = [rgba]
      self.remove_surfaces()
      for lev, c in zip(levels, colors):
        self.add_surface(lev, rgba = c)

    if replace or len(self.solid_levels) == 0:
      ilow, imid, imax = 0, 0.8, 1
      if mfrac is None:
        vlow = s.rank_data_value(1-vfrac[1])
        vmid = s.rank_data_value(1-vfrac[0])
      else:
        vlow = s.mass_rank_data_value(1-mfrac[1])
        vmid = s.mass_rank_data_value(1-mfrac[0])
      vmax = s.maximum
      rgba = saturate_rgba(self.default_rgba)
      if polar:
        self.solid_levels = ((s.minimum,imax), (max(-vmid,s.minimum),imid), (0,ilow),
                             (0,ilow), (vmid,imid), (vmax,imax))
        neg_rgba = tuple([1-c for c in rgba[:3]] + [rgba[3]])
        self.solid_colors = (neg_rgba,neg_rgba,neg_rgba, rgba,rgba,rgba)
      else:
        if vlow < vmid and vmid < vmax:
          self.solid_levels = ((vlow,ilow), (vmid,imid), (vmax,imax))
        else:
          self.solid_levels = ((vlow,ilow), (0.9*vlow+0.1*vmax,imid), (vmax,imax))
        self.solid_colors = [rgba]*len(self.solid_levels)

    self.initialized_thresholds = True
    self._drawings_need_update()

    self.call_change_callbacks('thresholds changed')

    return True

  # ---------------------------------------------------------------------------
  #
  def set_representation(self, rep):
    '''
    Set display style to "surface", "mesh", or "solid".
    '''
    if rep == self.representation:
      return
    
    self.redraw_needed()  # Switch to solid does not change surface until draw
    if rep == 'solid' or self.representation == 'solid':
      ro = self.rendering_options
      adjust_step = (ro.box_faces or ro.any_orthoplanes_shown())
    else:
      adjust_step = False
    self.representation = rep
    if adjust_step:
      ijk_min, ijk_max = self.region[:2]
      self.new_region(ijk_min, ijk_max)

    # Show or hide surfaces
    surfshow = (rep == 'surface' or rep == 'mesh')
    for s in self.surfaces:
      s.display = surfshow

    # Show or hide solid
    so = self.solid
    if so:
      so.drawing.display = (rep == 'solid')

    self.call_change_callbacks('representation changed')

  # ---------------------------------------------------------------------------
  #
  def _set_display(self, display):
    if display == self.display:
      return
    Model._set_display(self, display)
    self.call_change_callbacks('displayed')
    if not display:
      self._keep_displayed_data = None	# Allow data to be freed from cache.
  display = Model.display.setter(_set_display)

  # ---------------------------------------------------------------------------
  #
  def show(self, representation = None, rendering_options = None, show = True):
    '''
    Deprecated: Use v.display = True.
    Display the volume using the current parameters.
    '''
    if representation:
      self.set_representation(representation)
    rep = self.representation

    if rendering_options:
      self.rendering_options = rendering_options

    # Prevent cached matrix for displayed data from being freed.
    self._keep_displayed_data = self.displayed_matrices() if show else None

    # Update surface or solid
    if show:
      self._drawings_need_update()

    # Show or hide volume
    self.display = show

  # ---------------------------------------------------------------------------
  #
  def _drawings_need_update(self):
    s = self.session
    vm = getattr(s, '_volume_update_manager', None)
    if vm is None:
      s._volume_update_manager = vm = VolumeUpdateManager(s)
    vm.add(self)

  # ---------------------------------------------------------------------------
  #
  def update_drawings(self):

    rep = self.representation
    if rep == 'surface' or rep == 'mesh':
      self._update_surfaces()
    elif rep == 'solid':
      self._update_solid()

    # Prevent cached matrix for displayed data from being freed.
    self._keep_displayed_data = self.displayed_matrices()

  # ---------------------------------------------------------------------------
  #
  def _update_surfaces(self):

    show_mesh = (self.representation == 'mesh')
    ro = self.rendering_options
    try:
      for s in self.surfaces:
        s.update_surface(show_mesh, ro)
    except CancelOperation:
      pass

    self.show_outline_box(ro.show_outline_box, ro.outline_box_rgb,
                          ro.outline_box_linewidth)

  # ---------------------------------------------------------------------------
  #
  def _remove_contour_surface(self, surf):
    self.session.models.close([surf])
    
  # ---------------------------------------------------------------------------
  # Rank method ignores tolerance and uses a histogram of data values to
  # estimate the level that will contain the fraction of grid points corresponding
  # to the requested volume.
  #
  def surface_level_for_enclosed_volume(self, volume, tolerance = 1e-3,
                                        max_bisections = 30, rank_method = False):

    cell_volume = self.data.voxel_volume()

    if rank_method:
      ms = self.matrix_value_statistics()
      nx, ny, nz = self.data.size
      box_volume = cell_volume * nx * ny * nz
      r = 1.0 - (volume / box_volume)
      level = ms.rank_data_value(r)
      return level

    gvolume = volume / cell_volume
    matrix = self.full_matrix()
    from . import data
    try:
      level = data.surface_level_enclosing_volume(matrix, gvolume, tolerance, max_bisections)
    except MemoryError as e:
      self.session.warning(str(e))
      level = None
    return level

  # ---------------------------------------------------------------------------
  #
  def show_outline_box(self, show, rgb, linewidth):
    '''
    Show an outline box enclosing the displayed subregion of the volume.
    '''
    if show and rgb is not None:
      from .data import box_corners
      ijk_corners = box_corners(*self.ijk_bounds())
      corners = self.data.ijk_to_xyz_transform * ijk_corners
      if self.showing_orthoplanes():
        ro = self.rendering_options
        planes = ro.orthoplanes_shown
        center = self.data.ijk_to_xyz(ro.orthoplane_positions)
        crosshair_width = self.data.step
      else:
        planes, center, crosshair_width = None, None, None
      self.outline_box.show(corners, rgb, linewidth, center, planes,
                            crosshair_width)
    else:
      self.outline_box.erase_box()

  # ---------------------------------------------------------------------------
  #
  def _update_solid(self):

    s = self.solid
    if s is None:
      s = self._make_solid()
      self.solid = s

    ro = self.rendering_options
    s.set_options(ro.color_mode, ro.projection_mode,
                  ro.dim_transparent_voxels,
                  ro.bt_correction, ro.minimal_texture_memory,
                  ro.maximum_intensity_projection, ro.linear_interpolation,
                  ro.show_outline_box, ro.outline_box_rgb,
                  ro.outline_box_linewidth, ro.box_faces,
                  ro.orthoplanes_shown,
                  self.matrix_index(ro.orthoplane_positions))

    s.set_transform(self.matrix_indices_to_xyz_transform())


    tf = self.transfer_function()
    s.set_colormap(tf, self.solid_brightness_factor, self.transparency_depth)
    s.set_matrix(self.matrix_size(), self.data.value_type, self.matrix_id, self.matrix_plane)

    from . import grayscale
    bm = grayscale.blend_manager(self.session)
    s.update_drawing(self, bm)

    self.show_outline_box(ro.show_outline_box, ro.outline_box_rgb,
                          ro.outline_box_linewidth)
    return s

  # ---------------------------------------------------------------------------
  #
  def _make_solid(self):

    from . import solid
    name = self.name + ' solid'
    msize = self.matrix_size()
    value_type = self.data.value_type

    transform = self.matrix_indices_to_xyz_transform()
    align = self.surface_model()
    s = solid.Solid(name, msize, value_type, self.matrix_id, self.matrix_plane,
                    transform, align, self.message)
    if hasattr(self, 'mask_colors'):
      s.mask_colors = self.mask_colors
    return s

  # ---------------------------------------------------------------------------
  #
  def shown(self):

    surf_disp = len([s for s in self.surfaces if s.display]) > 0
    solid_disp = (self.solid and self.solid.drawing.display)
    return self.display and (surf_disp or solid_disp)
    
  # ---------------------------------------------------------------------------
  #
  def showing_orthoplanes(self):

    return bool(self.representation == 'solid' and
                self.rendering_options.any_orthoplanes_shown())

  # ---------------------------------------------------------------------------
  #
  def shown_orthoplanes(self):

    ro = self.rendering_options
    shown = ro.orthoplanes_shown
    ijk_min, ijk_max = self.region[:2]
    ijk = clamp_ijk(ro.orthoplane_positions, ijk_min, ijk_max)
    return tuple((axis, ijk[axis]) for axis in (0,1,2) if shown[axis])
    
  # ---------------------------------------------------------------------------
  #
  def showing_box_faces(self):

    return bool(self.representation == 'solid' and
                self.rendering_options.box_faces)
    
  # ---------------------------------------------------------------------------
  #
  def principal_channel(self):

    c = self._channels
    return c is None or self == c.first_channel
    
  # ---------------------------------------------------------------------------
  #
  def other_channels(self):

    c = self._channels
    if c is None:
      vc = []
    else:
      vc = [v for v in c.maps if v is not self]
    return vc
  
  # ---------------------------------------------------------------------------
  #
  def copy(self):

    v = volume_from_grid_data(self.data, self.session, self.representation,
                              show_dialog = False)
    v.copy_settings_from(self)
    return v

  # ---------------------------------------------------------------------------
  #
  def copy_settings_from(self, v,
                         copy_style = True,
                         copy_thresholds = True,
                         copy_colors = True,
                         copy_rendering_options = True,
                         copy_region = True,
                         copy_xform = True,
                         copy_active = True,
                         copy_zone = True):

    if copy_style:
      # Copy display style
      self.set_representation(v.representation)

    if copy_thresholds:
      # Copy thresholds
      self.set_parameters(
        surface_levels = [s.level for s in v.surfaces],
        solid_levels = v.solid_levels,
        )
    if copy_colors:
      # Copy colors
      self.set_parameters(
        surface_colors = [s.rgba for s in v.surfaces],
        surface_brightness_factor = v.surface_brightness_factor,
        transparency_factor = v.transparency_factor,
        solid_colors = v.solid_colors,
        transparency_depth = v.transparency_depth,
        solid_brightness_factor = v.solid_brightness_factor,
        default_rgba = v.default_rgba
        )

    if copy_rendering_options:
      # Copy rendering options
      self.set_parameters(**v.rendering_options.__dict__)

    if copy_region:
      # Copy region bounds
      ijk_min, ijk_max, ijk_step = v.region
      self.new_region(ijk_min, ijk_max, ijk_step)

    if copy_xform:
      # Copy position and orientation
      self.position = v.position

    # if copy_active:
    #   # Copy movability
    #   self.surface_model().openState.active = v.surface_model().openState.active

    # if copy_zone:
    #   # Copy surface zone
    #   sm = v.surface_model()
    #   sm_copy = self.surface_model()
    #   import SurfaceZone
    #   if sm and SurfaceZone.showing_zone(sm) and sm_copy:
    #     points, distance = SurfaceZone.zone_points_and_distance(sm)
    #     SurfaceZone.surface_zone(sm_copy, points, distance, True)

    # TODO: Should copy color zone too.

  # ---------------------------------------------------------------------------
  # If volume data is not writable then make a writable copy.
  #
  def writable_copy(self, require_copy = False,
                    unshow_original = True, model_id = None,
                    subregion = None, step = (1,1,1), name = None,
                    copy_colors = True, value_type = None):

    r = self.subregion(step, subregion)
    if not require_copy and self.data.writable and self.is_full_region(r):
      return self

    g = self.region_grid(r, value_type)
    g.array[:,:,:] = self.region_matrix(r)

    if name:
      g.name = name
    elif self.name.endswith('copy'):
      g.name = self.name
    else:
      g.name = self.name + ' copy'

    v = volume_from_grid_data(g, self.session, self.representation, model_id = model_id,
                              show_dialog = False)
    v.copy_settings_from(self, copy_region = False, copy_colors = copy_colors)

    if unshow_original:
      self.display = False

    return v

  # ---------------------------------------------------------------------------
  #
  def region_grid(self, r, value_type = None):

    shape = self.matrix_size(region = r, clamp = False)
    shape.reverse()
    d = self.data
    if value_type is None:
      value_type = d.value_type
    from numpy import zeros
    m = zeros(shape, value_type)
    origin, step = self.region_origin_and_step(r)
    from .data import ArrayGridData
    g = ArrayGridData(m, origin, step, d.cell_angles, d.rotation)
    g.rgba = d.rgba           # Copy default data color.
    return g

  # ---------------------------------------------------------------------------
  #
  def surface_bounds(self):
    '''Surface bounds in volume coordinate system.'''
    from chimerax.core.geometry import union_bounds
    return union_bounds([s.geometry_bounds() for s in self.surfaces])
      
  # ---------------------------------------------------------------------------
  # The xyz bounding box encloses the subsampled grid with half a step size
  # padding on all sides.
  #
  def xyz_bounds(self, step = None, subregion = None):

    ijk_min_edge, ijk_max_edge = self.ijk_bounds(step, subregion)
    
    from .data import box_corners, bounding_box
    ijk_corners = box_corners(ijk_min_edge, ijk_max_edge)
    data = self.data
    xyz_min, xyz_max = bounding_box([data.ijk_to_xyz(c) for c in ijk_corners])
    
    return (xyz_min, xyz_max)

  # ---------------------------------------------------------------------------
  #
  def first_intercept(self, mxyz1, mxyz2, exclude = None):

    if exclude is not None and exclude(self):
      return None
      
    if self.representation == 'solid':
      ro = self.rendering_options
      if not ro.box_faces and ro.orthoplanes_shown == (False,False,False):
        vxyz1, vxyz2 = self.position.inverse() * (mxyz1, mxyz2)
        from . import slice
        xyz_in, xyz_out = slice.box_line_intercepts((vxyz1, vxyz2), self.xyz_bounds())
        if xyz_in is None or xyz_out is None:
          return None
        from chimerax.core.geometry import norm
        f = norm(0.5*(xyz_in+xyz_out) - mxyz1) / norm(mxyz2 - mxyz1)
        return PickedMap(self, f)

    from chimerax.core.graphics import Drawing
    pd = Drawing.first_intercept(self, mxyz1, mxyz2, exclude)
    if pd:
      d = pd.drawing()
      detail = d.name
      p = PickedMap(self, pd.distance, detail)
      p.triangle_pick = pd
      if d.display_style == d.Mesh or hasattr(pd, 'is_transparent') and pd.is_transparent():
        # Try picking opaque object under transparent map
        p.pick_through = True
    else:
      p = None

    return p

  # ---------------------------------------------------------------------------
  #
  def planes_pick(self, planes, exclude=None):
    picks = Model.planes_pick(self, planes, exclude)
    if picks:
      picks = [PickedMap(self)]
    return picks

  # ---------------------------------------------------------------------------
  #
  def _set_selected(self, sel, *, fire_trigger=True):
    Model.set_selected(self, sel, fire_trigger=fire_trigger)
    for s in self.surfaces:
      s.set_selected(sel)
  selected = property(Model.selected.fget, _set_selected)

  # ---------------------------------------------------------------------------
  #
  def clear_selection(self):
    for s in self.surfaces:
      s.selected = False
    self.selected = False

  # ---------------------------------------------------------------------------
  # The data ijk bounds with half a step size padding on all sides.
  #
  def ijk_bounds(self, step = None, subregion = None, integer = False):

    ijk_origin, ijk_size, ijk_step = self.ijk_aligned_region(step, subregion)
    mat_size = [(a+b-1)//b for a,b in zip(ijk_size, ijk_step)]
    ijk_last = [a+b*(c-1) for a,b,c in zip(ijk_origin, ijk_step, mat_size)]

    ijk_min_edge = [a - .5*b for a,b in zip(ijk_origin, ijk_step)]
    ijk_max_edge = [a + .5*b for a,b in zip(ijk_last, ijk_step)]
    if integer:
      r = self.integer_region(ijk_min_edge, ijk_max_edge, step)
      ijk_min_edge, ijk_max_edge = r[:2]

    return ijk_min_edge, ijk_max_edge

  # ---------------------------------------------------------------------------
  #
  def integer_region(self, ijk_min, ijk_max, step = None):

    if step is None:
      step = self.region[2]
    elif isinstance(step, int):
      step = (step,step,step)
    from math import floor, ceil
    ijk_min = [int(floor(i/s)*s) for i,s in zip(ijk_min,step)]
    ijk_max = [int(ceil(i/s)*s) for i,s in zip(ijk_max,step)]
    r = (ijk_min, ijk_max, step)
    return r

  # ---------------------------------------------------------------------------
  # Points must be in volume local coordinates.
  #
  def bounding_region(self, points, padding = 0, step = None, clamp = True, cubify = False):

    d = self.data
    from .data import points_ijk_bounds
    ijk_fmin, ijk_fmax = points_ijk_bounds(points, padding, d)
    r = self.integer_region(ijk_fmin, ijk_fmax, step)
    if cubify:
      ijk_min, ijk_max = r[:2]
      s = max(a-b+1 for a, b in zip(ijk_max, ijk_min)) if isinstance(cubify,bool) else cubify
      for a in (0,1,2):
        sa = ijk_max[a] - ijk_min[a] + 1
        if sa < s:
          ds = s-sa
          o = (ds+1)//2 if ijk_fmax[a] - ijk_max[a] > ijk_min[a] - ijk_fmin[a] else ds//2
          ijk_max[a] += o
          ijk_min[a] -= ds - o
    if clamp:
      r = clamp_region(r, d.size)
    return r

  # ---------------------------------------------------------------------------
  #
  def ijk_to_global_xyz(self, ijk):

    pm = self.data.ijk_to_xyz(ijk)
    p = self.position * pm
    return p

  # ---------------------------------------------------------------------------
  # Axis vector in global coordinates.
  #
  def axis_vector(self, axis):

    d = self.data
    va = {0:(1,0,0), 1:(0,1,0), 2:(0,0,1)}[axis]
    lv = d.ijk_to_xyz(va) - d.ijk_to_xyz((0,0,0))
    v = self.position * lv
    from chimerax.core.geometry import normalize_vector
    vn = normalize_vector(v)
    return vn

  # ---------------------------------------------------------------------------
  #
  def matrix(self, read_matrix = True, step = None, subregion = None):

    r = self.subregion(step, subregion)
    m = self.region_matrix(r, read_matrix)
    return m

  # ---------------------------------------------------------------------------
  #
  def full_matrix(self, read_matrix = True, step = (1,1,1)):

    m = self.matrix(read_matrix, step, full_region(self.data.size)[:2])
    return m

  # ---------------------------------------------------------------------------
  # Region includes ijk_min and ijk_max points.
  #
  def region_matrix(self, region = None, read_matrix = True):

    if region is None:
      region = self.region
    origin, size, step = self.step_aligned_region(region)
    d = self.data
    operation = 'reading %s' % d.name
    from .data import ProgressReporter
    progress = ProgressReporter(operation, size, d.value_type.itemsize,
                                message = self.message_cb)
    from_cache_only = not read_matrix
    m = d.matrix(origin, size, step, progress, from_cache_only)
    return m

  # ---------------------------------------------------------------------------
  # Size of matrix for subsampled subregion returned by matrix().
  #
  def matrix_size(self, step = None, subregion = None, region = None,
                  clamp = True):

    if region is None:
      region = self.subregion(step, subregion)
    origin, size, step = self.step_aligned_region(region, clamp)
    mat_size = [(a+b-1)//b for a,b in zip(size, step)]
    return mat_size

  # ---------------------------------------------------------------------------
  #
  def matrix_index(self, vijk):

    ijk_min, ijk_step = self.region[0], self.region[2]
    mijk = tuple((i-((m+s-1)//s)*s)//s for i,m,s in zip(vijk, ijk_min, ijk_step))
    # clamp
    msize = self.matrix_size()
    mijk = tuple(max(0, min(i,s-1)) for i,s in zip(mijk, msize))
    return mijk

  # ---------------------------------------------------------------------------
  # Slice specifying portion of full matrix shown (uses ijk order).
  #
  def matrix_slice(self, step = None, subregion = None, region = None,
                  clamp = True):

    if region is None:
      region = self.subregion(step, subregion)
    origin, size, step = self.step_aligned_region(region, clamp)
    slc = [slice(o,o+st*(sz-1)+1,st) for o,st,sz in zip(origin,step,size)]
    return slc

  # ---------------------------------------------------------------------------
  # Return 2d array for one plane of matrix for current region.  The plane
  # is specified as an axis and a matrix index.  This is used for solid
  # style rendering in box mode, orthoplane mode, and normal mode.
  #
  def matrix_plane(self, axis, mplane, read_matrix = True):

    if axis is None:
      return self.matrix()

    ijk_min, ijk_max, ijk_step = [list(b) for b in self.region]
    ijk_min[axis] += mplane*ijk_step[axis]
    ijk_max[axis] = ijk_min[axis]
    m = self.region_matrix((ijk_min, ijk_max, ijk_step), read_matrix)
    if m is None:
      return None
    s = [slice(None), slice(None), slice(None)]
    s[2-axis] = 0
    m2d = m[tuple(s)]
    return m2d

  # ---------------------------------------------------------------------------
  #
  def single_plane(self):

    return len([s for s in self.matrix_size() if s == 1]) > 0
    
  # ---------------------------------------------------------------------------
  #
  def expand_single_plane(self):

    if self.single_plane():
        ijk_min, ijk_max = [list(i) for i in self.region[:2]]
        msize = self.matrix_size()
        for a in (0,1,2):
            if msize[a] == 1:
                ijk_min[a] = 0
                ijk_max[a] = self.data.size[a]-1
        self.new_region(ijk_min, ijk_max)

  # ---------------------------------------------------------------------------
  # Transform mapping matrix indices to xyz.  The matrix indices are not the
  # same as the data indices since the matrix includes only the current
  # subregion and subsampled data values.
  #
  def matrix_indices_to_xyz_transform(self, step = None, subregion = None):

    ijk_origin, ijk_size, ijk_step = self.ijk_aligned_region(step, subregion)

    data = self.data
    xo, yo, zo = data.ijk_to_xyz(ijk_origin)
    io, jo, ko = ijk_origin
    istep, jstep, kstep = ijk_step
    xi, yi, zi = data.ijk_to_xyz((io+istep, jo, ko))
    xj, yj, zj = data.ijk_to_xyz((io, jo+jstep, ko))
    xk, yk, zk = data.ijk_to_xyz((io, jo, ko+kstep))
    from chimerax.core.geometry import Place
    tf = Place(((xi-xo, xj-xo, xk-xo, xo),
                (yi-yo, yj-yo, yk-yo, yo),
                (zi-zo, zj-zo, zk-zo, zo)))
    return tf

  # ---------------------------------------------------------------------------
  #
  def data_origin_and_step(self, step = None, subregion = None):

    r = self.subregion(step, subregion)
    return self.region_origin_and_step(r)

  # ---------------------------------------------------------------------------
  #
  def region_origin_and_step(self, region):

    ijk_origin, ijk_max, ijk_step = region
    dorigin = self.data.ijk_to_xyz(ijk_origin)
    dstep = [a*b for a,b in zip(self.data.step, ijk_step)]
    return dorigin, dstep

  # ---------------------------------------------------------------------------
  # Data values or coordinates have changed.
  # Surface / solid rendering is not automatically redrawn when data values
  # change.
  #
  def data_changed_cb(self, type):

    if type == 'values changed':
      self.data.clear_cache()
      self.matrix_changed()
      self._drawings_need_update()
      self.call_change_callbacks('data values changed')
      # TODO: should this automatically update the data display?
    elif type == 'coordinates changed':
      self.call_change_callbacks('coordinates changed')
    elif type == 'path changed':
      self.name = utf8_string(self.data.name)

  # ---------------------------------------------------------------------------
  # Return the origin and size of the subsampled submatrix to be read.
  #
  def ijk_aligned_region(self, step = None, subregion = None):

    r = self.subregion(step, subregion)
    return self.step_aligned_region(r)

  # ---------------------------------------------------------------------------
  # Return the origin aligned to a multiple of step and size of the region and step.
  #
  def step_aligned_region(self, region, clamp = True):

    ijk_min, ijk_max, ijk_step = region
    
    # Samples always have indices divisible by step, so increase ijk_min if
    # needed to make it a multiple of ijk_step.
    origin = [s*((i+s-1)//s) for i,s in zip(ijk_min, ijk_step)]

    # If region is non-empty but contains no step multiple decrease ijk_min.
    for a in range(3):
      if origin[a] > ijk_max[a] and ijk_min[a] <= ijk_max[a]:
        origin[a] -= ijk_step[a]

    end = [s*(i+s)//s for i,s in zip(ijk_max, ijk_step)]
    if clamp:
      origin = [max(i,0) for i in origin]
      end = [min(i,lim) for i,lim in zip(end, self.data.size)]
    size = [e-o for e,o in zip(end, origin)]

    return tuple(origin), tuple(size), tuple(ijk_step)

  # ---------------------------------------------------------------------------
  # Applying point_xform to points gives Chimera world coordinates.  If the
  # point_xform is None then the points are in local volume coordinates.
  # The returned values are float32.
  #
  def interpolated_values(self, points, point_xform = None,
                          out_of_bounds_list = False, subregion = 'all',
                          step = (1,1,1), method = 'linear'):

    matrix, p2m_transform = self.matrix_and_transform(point_xform,
                                                      subregion, step)
    from .data import interpolate_volume_data
    values, outside = interpolate_volume_data(points, p2m_transform,
                                              matrix, method)

    if out_of_bounds_list:
      return values, outside
    
    return values
    
  # ---------------------------------------------------------------------------
  # Applying point_xform to points gives Chimera world coordinates.  If the
  # point_xform is None then the points are in local volume coordinates.
  #
  def interpolated_gradients(self, points, point_xform = None,
                             out_of_bounds_list = False,
                             subregion = 'all', step = (1,1,1),
                             method = 'linear'):

    matrix, v2m_transform = self.matrix_and_transform(point_xform,
                                                      subregion, step)

    from .data import interpolate_volume_gradient
    gradients, outside = interpolate_volume_gradient(points, v2m_transform,
                                                     matrix, method)
    if out_of_bounds_list:
      return gradients, outside
    
    return gradients

  # ---------------------------------------------------------------------------
  # Add values from another volume interpolated at grid positions
  # of this volume.  The subregion and step arguments allow interpolating
  # not using the full map v.
  #
  def add_interpolated_values(self, v, subregion = 'all', step = (1,1,1), scale = 1):

    self.combine_interpolated_values(v, 'add', subregion, step, scale)

  # ---------------------------------------------------------------------------
  # Combine values from another volume interpolated at grid positions
  # of this volume by adding, subtracting, taking maximum, multiplying....
  # The subregion and step arguments allow interpolating not using the full map v.
  #
  def combine_interpolated_values(self, v, operation = 'add',
                                  subregion = 'all', step = (1,1,1), scale = 1):

    if scale == 0:
      return

    values, const_values = v.interpolate_on_grid(self, subregion, step)

    # Add scaled copy of array.
    d = self.data
    m = d.full_matrix()
    if scale == 'minrms':
      level = v.minimum_surface_level
      if level is None:
        level = 0
      scale = -minimum_rms_scale(values, m, level)
      log = self.session.logger
      log.info('Minimum RMS scale factor for "%s" above level %.5g is %.5g\n'
               % (v.name_with_id(), level, scale))
    if scale != 1:
      if const_values:
        # Copy array only if scaling and the values come from another map
        # without interpolation because the grids matched, and values should
        # not be modified.
        from numpy import float32
        values = values.astype(float32)
      values *= scale
    if values.dtype != m.dtype:
      values = values.astype(m.dtype, copy=False)
    if operation == 'add':
      m[:,:,:] += values
    elif operation == 'subtract':
      m[:,:,:] -= values
    elif operation == 'maximum':
      from numpy import maximum
      maximum(m, values, m)
    elif operation == 'minimum':
      from numpy import minimum
      minimum(m, values, m)
    elif operation == 'multiply':
      m[:,:,:] *= values
    d.values_changed()

  # ---------------------------------------------------------------------------
  # Returns 3-d array of values interpolated on the full grid of another map.
  # Step and subregion are for the interpolated map, not for the grid.
  # Currently there is no option to interpolate on a subregion grid.
  #
  def interpolate_on_grid(self, vgrid, subregion = 'all', step = (1,1,1)):

    same = same_grid(vgrid, full_region(vgrid.data.size),
                     self, self.subregion(step, subregion))
    if same:
      # Optimization: avoid interpolation for identical grids.
      values = self.matrix(step = step, subregion = subregion)
    else:
      size_limit = 2 ** 22          # 4 Mvoxels
      isize, jsize, ksize = vgrid.matrix_size(step = 1, subregion = 'all')
      shape = (ksize, jsize, isize)
      if isize*jsize*ksize > size_limit:
        # Calculate plane by plane to save memory with grid point array
        from numpy import empty, single as floatc
        values = empty(shape, floatc)
        for z in range(ksize):
          points = vgrid.grid_points(self.model_transform().inverse(), z)
          values1d = self.interpolated_values(points, None,
                                              subregion = subregion,
                                              step = step)
          values[z,:,:] = values1d.reshape((jsize, isize))
      else:
        points = vgrid.grid_points(self.model_transform().inverse())
        values1d = self.interpolated_values(points, None,
                                            subregion = subregion,
                                            step = step)
        values = values1d.reshape(shape)

    return values, same

  # -----------------------------------------------------------------------------
  #
  def mean_sd_rms(self):
    return mean_sd_rms(self.matrix())

  # ---------------------------------------------------------------------------
  # Return xyz coordinates of grid points of volume data transformed to a
  # local coordinate system.
  #
  def grid_points(self, transform_to_local_coords, zplane = None):

    data = self.data
    size = data.size
    from numpy import float32
    from .data import grid_indices
    if zplane is None:
      points = grid_indices(size, float32)
    else:
      points = grid_indices((size[0],size[1],1), float32)  # Single z plane.
      points[:,2] = zplane
    mt = transform_to_local_coords * self.model_transform() * data.ijk_to_xyz_transform
    mt.transform_points(points, in_place = True)
    return points
  
  # ---------------------------------------------------------------------------
  # Returns 3-d numeric array and transformation from a given "source"
  # coordinate system to array indices.  The source_to_scene_transform transforms
  # from the source coordinate system to Chimera scene coordinates.
  # If the transform is None it means the source coordinates are the
  # same as the volume local coordinates.
  #
  def matrix_and_transform(self, source_to_scene_transform, subregion, step):
    
    m2s_transform = self.matrix_indices_to_xyz_transform(step, subregion)
    if source_to_scene_transform:
      # Handle case where vertices and volume have different model transforms.
      scene_to_source_tf = source_to_scene_transform.inverse()
      m2s_transform = scene_to_source_tf * self.scene_position * m2s_transform
      
    s2m_transform = m2s_transform.inverse()

    matrix = self.matrix(step=step, subregion=subregion)

    return matrix, s2m_transform

  # ---------------------------------------------------------------------------
  # Return currently displayed subregion.  If only a zone is being displayed
  # set all grid data values outside the zone to zero.
  #
  def grid_data(self, subregion = None, step = (1,1,1), mask_zone = True,
                region = None):

    if region is None:
      region = self.subregion(step, subregion)
    if self.is_full_region(region):
      sg = self.data
    else:
      ijk_min, ijk_max, ijk_step = region
      from .data import GridSubregion
      sg = GridSubregion(self.data, ijk_min, ijk_max, ijk_step)

    if mask_zone:
      surf_model = self.surface_model()
#      import SurfaceZone
#      if SurfaceZone.showing_zone(surf_model):
      if False:
        points, radius = SurfaceZone.zone_points_and_distance(surf_model)
        from .data import zone_masked_grid_data
        mg = zone_masked_grid_data(sg, points, radius)
        return mg
        
    return sg
  
  # ---------------------------------------------------------------------------
  #
  def subregion(self, step = None, subregion = None):

    if subregion is None:
      ijk_min, ijk_max = self.region[:2]
    elif isinstance(subregion, str):
      if subregion == 'all':
        ijk_min, ijk_max = full_region(self.data.size)[:2]
      elif subregion == 'shown':
        ijk_min, ijk_max = self.region[:2]
      else:
        ijk_min, ijk_max = self.region_list.named_region_bounds(subregion)
        if ijk_min == None or ijk_max == None:
          ijk_min, ijk_max = self.region[:2]
    else:
      ijk_min, ijk_max = subregion

    if step is None:
      ijk_step = self.region[2]
    elif isinstance(step, int):
      ijk_step = (step, step, step)
    else:
      ijk_step = step

    r = (ijk_min, ijk_max, ijk_step)
    return r
  
  # ---------------------------------------------------------------------------
  #
  def copy_zone(self, outside = False):

    import SurfaceZone
    if not SurfaceZone.showing_zone(self):
      return None

    points, radius = SurfaceZone.zone_points_and_distance(self)
    from .data import zone_masked_grid_data
    masked_data = zone_masked_grid_data(self.data, points, radius,
                                        invert_mask = outside)
    if outside: name = 'outside zone'
    else: name = 'zone'
    masked_data.name = self.name + ' ' + name
    mv = volume_from_grid_data(masked_data, self.session)
    mv.copy_settings_from(self, copy_region = False, copy_zone = False)
    return mv
  
  # ---------------------------------------------------------------------------
  #
  def matrix_value_statistics(self, read_matrix = True):

    ms = self.matrix_stats
    if ms:
      return ms

    matrices = self.displayed_matrices(read_matrix)
    if len(matrices) == 0:
      return None

    min_status_message_voxels = 2**27
    nvox = sum(m.size for m in matrices)
    if nvox >= min_status_message_voxels:
      self.message('Computing histogram for %s' % self.name)
    from . import data
    self.matrix_stats = ms = data.MatrixValueStatistics(matrices)
    if nvox >= min_status_message_voxels:    
      self.message('')

    return ms
  
  # ---------------------------------------------------------------------------
  #
  def displayed_matrices(self, read_matrix = True):

    matrices = []
    if self.representation == 'solid':
      ro = self.rendering_options
      if ro.box_faces:
        msize = self.matrix_size()
        for axis in (0,1,2):
          matrices.append(self.matrix_plane(axis, 0, read_matrix))
          matrices.append(self.matrix_plane(axis, msize[axis]-1, read_matrix))
      elif ro.any_orthoplanes_shown():
        omijk = self.matrix_index(ro.orthoplane_positions)
        for axis in (0,1,2):
          if ro.orthoplanes_shown[axis]:
            matrices.append(self.matrix_plane(axis, omijk[axis], read_matrix))

    if len(matrices) == 0:
      matrices.append(self.matrix(read_matrix))

    matrices = [m for m in matrices if not m is None]
    return matrices
  
  # ---------------------------------------------------------------------------
  # Apply surface/mesh transparency factor.
  #
  def _modulated_surface_color(self, rgba):

    r,g,b,a = rgba

    bf = self.surface_brightness_factor

    ofactor = 1 - self.transparency_factor
    ofactor = max(0, ofactor)
    ofactor = min(1, ofactor)
      
    return (r * bf, g * bf, b * bf, a * ofactor)
  
  # ---------------------------------------------------------------------------
  # Without brightness and transparency adjustment.
  #
  def transfer_function(self):

    tf = [tuple(ts) + tuple(c) for ts,c in zip(self.solid_levels, self.solid_colors)]
    tf.sort()

    return tf
  
  # ---------------------------------------------------------------------------
  #
  def write_file(self, path, format = None, options = {}, temporary = False):

    from .data import save_grid_data
    d = self.grid_data()
    format = save_grid_data(d, path, self.session, format, options, temporary)

  # ---------------------------------------------------------------------------
  #
  def showing_transparent(self):
    if not self.display:
      return False
    if self.representation == 'solid' and self.solid:
      return 'a' in self.solid.color_mode
    from chimerax.core.graphics import Drawing
    return Drawing.showing_transparent(self)
      
  # ---------------------------------------------------------------------------
  #
  def view_models(self, view, representation = None):

    mlist = []
    
    if representation in (None, 'surface', 'mesh'):
      mlist.append(self)
      self.display = view

    if representation in (None, 'solid'):
      s = self.solid
      if s:
        m = s.model()
        if m:
          mlist.append(m)
          m.display = view

    return len(mlist) > 0
  
  # ---------------------------------------------------------------------------
  #
  def models(self):

    mlist = [self]
    return mlist
  
  # ---------------------------------------------------------------------------
  #
  def surface_model(self):

    return self
  
  # ---------------------------------------------------------------------------
  #
  def solid_model(self):

    s = self.solid
    if s:
      return s.model()
    return None
    
  # ---------------------------------------------------------------------------
  #
  def model_transform(self):

    return self.position
  
  # ---------------------------------------------------------------------------
  #
  def close_models(self):

    self.close_solid()
    self.close_surface()
  
  # ---------------------------------------------------------------------------
  #
  def close_surface(self):

    self.remove_all_drawings()
      
  # ---------------------------------------------------------------------------
  #
  def close_solid(self):

    s = self.solid
    if s:
      s.close_model(self)
      self.solid = None
      
  # ---------------------------------------------------------------------------
  #
  def delete(self):

    d = self.data
    if d:
      d.clear_cache()
    self.close_models()
    Model.delete(self)
      
  # ---------------------------------------------------------------------------
  #
  def model_closed_cb(self, model):

    if self.data:
      self.data.remove_change_callback(self.data_changed_cb)
      self.data = None
      self._keep_displayed_data = None
      self.outline_box = None   # Remove reference loops
      from chimera import triggers
      triggers.deleteHandler('SurfacePiece', self.surface_piece_change_handler)
      self.surface_piece_change_handler = None


  # State save/restore in ChimeraX
  def take_snapshot(self, session, flags):
    from .session import state_from_map, grid_data_state
    from chimerax.core.state import State
    include_maps = bool(flags & State.INCLUDE_MAPS)
    data = {
      'model state': Model.take_snapshot(self, session, flags),
      'volume state': state_from_map(self),
      'grid data state': grid_data_state(self.data, session, include_maps=include_maps),
      'version': 1,
    }
    return data

  @staticmethod
  def restore_snapshot(session, data):
    grid_data = data['grid data state'].grid_data
    if grid_data is None:
      return None	# Map file not available.
    v = Volume(session, grid_data)
    Model.set_state_from_snapshot(v, session, data['model state'])
    from .session import set_map_state
    set_map_state(data['volume state'], v)
    v._drawings_need_update()
    show_volume_dialog(session)
    return v

# -----------------------------------------------------------------------------
#
from chimerax.core.models import Surface
class VolumeSurface(Surface):

  def __init__(self, volume, level, rgba = (1.0,1.0,1.0,1.0)):
    name = 'level %.3g' % level
    Surface.__init__(self, name, volume.session)
    self.volume = volume
    self._level = level
    self.rgba = rgba
    self._contour_settings = {}	         	# Settings for current surface geometry
    self._min_status_message_voxels = 2**24	# Show status messages only on big surface calculations
    self._use_thread = False			# Whether to compute next surface in thread
    self._surf_calc_thread = None
    
  def delete(self):
    self.volume._surfaces.remove(self)
    Surface.delete(self)

  def get_level(self):
    return self._level
  def set_level(self, level, use_thread = False):
    self._level = level
    self._use_thread = use_thread
  level = property(get_level, set_level)
  
  # ---------------------------------------------------------------------------
  #
  def update_surface(self, show_mesh, rendering_options):

    self._use_thread_result()
    
    if not self._geometry_changed(rendering_options):
      self._set_appearance(show_mesh, rendering_options)
      return
    
    v = self.volume
    matrix = v.matrix()
    level = self.level
    
    show_status = (matrix.size >= self._min_status_message_voxels)
    if show_status:
      v.message('Computing %s surface, level %.3g' % (v.data.name, level))

    if self._use_thread:
      self._use_thread = False
      self._calc_surface_in_thread(matrix, level, rendering_options, show_mesh)
      return
    else:
      # Don't use thread calculation started earlier since new non-threaded calculation has begun.
      self._surf_calc_thread = None
      
    try:
      va, na, ta, hidden_edges = self._calculate_contour_surface(matrix, level, rendering_options)
    except MemoryError:
      ses = v.session
      ses.warning('Ran out of memory contouring at level %.3g.\n' % level +
                  'Try a higher contour level.')
      return
    
    if show_status:
      v.message('Calculated %s surface, level %.3g, with %d triangles'
                   % (v.data.name, level, len(ta)), blank_after = 3.0)

    self._set_surface(va, na, ta, hidden_edges)
    self._set_appearance(show_mesh, rendering_options)

  # ---------------------------------------------------------------------------
  #
  def _calc_surface_in_thread(self, matrix, level, rendering_options, show_mesh):
    sct = self._surf_calc_thread
    new_thread = (sct is None or not sct.is_alive())
    if new_thread:
      from chimerax.core.threadq import WorkThread
      self._surf_calc_thread = sct = WorkThread(self._calculate_contour_surface_threaded,
                                                in_queue = (sct.in_queue if sct else None),
                                                out_queue = (sct.out_queue if sct else None))
      sct.daemon = True

    # Clear the input queue, only the most recent surface calculation is queued
    import queue
    try:
      while sct.in_queue.get_nowait():
        sct.in_queue.task_done()
    except queue.Empty:
      pass
    
    sct.in_queue.put((matrix, level, rendering_options, show_mesh))

    if new_thread:
      sct.start()	# Start surface calculation in separate thread
      
    self.volume._drawings_need_update()        # Check later for surface calc result
    
  # ---------------------------------------------------------------------------
  #
  def _use_thread_result(self):
    sct = self._surf_calc_thread
    if sct is None:
      return

    import queue
    try:
      result = sct.out_queue.get_nowait()
    except queue.Empty:
      if sct.is_alive():
        self.volume._drawings_need_update()        # Check later for surface calc result
      else:
        self._surf_calc_thread = None
      return

    va, na, ta, hidden_edges, matrix, level, rendering_options, show_mesh = result
    self._set_surface(va, na, ta, hidden_edges)
    self._set_appearance(show_mesh, rendering_options)
    
    show_status = (matrix.size >= self._min_status_message_voxels)
    if show_status:
      v = self.volume
      v.message('Calculated %s surface, level %.3g, with %d triangles'
                   % (v.data.name, level, len(ta)), blank_after = 3.0)
    
  # ---------------------------------------------------------------------------
  #
  def _calculate_contour_surface_threaded(self, matrix, level, rendering_options, show_mesh):
    va, na, ta, hidden_edges = self._calculate_contour_surface(matrix,level, rendering_options)
    return va, na, ta, hidden_edges, matrix, level, rendering_options, show_mesh
  
  # ---------------------------------------------------------------------------
  #
  def _calculate_contour_surface(self, matrix, level, rendering_options):

    # _map contour code does not handle single data planes.
    # Handle these by stacking two planes on top of each other.
    plane_axis = [a for a in (0,1,2) if matrix.shape[a] == 1]
    if plane_axis:
      for a in plane_axis:
        matrix = matrix.repeat(2, axis = a)

    from ._map import contour_surface
    varray, tarray, narray = contour_surface(matrix, level,
                                             cap_faces = rendering_options.cap_faces,
                                             calculate_normals = True)

    if plane_axis:
      for a in plane_axis:
        varray[:,2-a] = 0

    va, na, ta, hidden_edges = self._adjust_surface_geometry(varray, narray, tarray,
                                                             rendering_options, level)

    return va, na, ta, hidden_edges
      
  # ---------------------------------------------------------------------------
  #
  def _adjust_surface_geometry(self, varray, narray, tarray, rendering_options, level):

    ro = rendering_options
    if ro.flip_normals and level < 0:
      from chimerax.surface import invert_vertex_normals
      invert_vertex_normals(narray, tarray)

    # Preserve triangle vertex traversal direction about normal.
    v = self.volume
    transform = v.matrix_indices_to_xyz_transform()
    if transform.determinant() < 0:
      from ._map import reverse_triangle_vertex_order
      reverse_triangle_vertex_order(tarray)

    if ro.subdivide_surface:
      from chimerax.surface import subdivide_triangles
      for i in range(ro.subdivision_levels):
        varray, tarray, narray = subdivide_triangles(varray, tarray, narray)

    if ro.square_mesh:
      from numpy import empty, uint8
      hidden_edges = empty((len(tarray),), uint8)
      from . import _map
      _map.principle_plane_edges(varray, tarray, hidden_edges)
    else:
      hidden_edges = None
      
    if ro.surface_smoothing:
      sf, si = ro.smoothing_factor, ro.smoothing_iterations
      from chimerax.surface import smooth_vertex_positions
      smooth_vertex_positions(varray, tarray, sf, si)
      smooth_vertex_positions(narray, tarray, sf, si)

    # Transform vertices and normals from index coordinates to model coordinates
    transform.transform_points(varray, in_place = True)
    transform.transform_normals(narray, in_place = True)
    from chimerax.core.geometry import normalize_vectors
    normalize_vectors(narray)

    return varray, narray, tarray, hidden_edges

  # ---------------------------------------------------------------------------
  #
  def _set_surface(self, va, na, ta, hidden_edges):
    
    self.set_geometry(va, na, ta)

    self.edge_mask = hidden_edges

    # TODO: Clip cap offset for different contour levels is not related to voxel size.
    v = self.volume
    self.clip_cap = True
    self.clip_offset = .002* len([s for s in v.surfaces if self.level < s.level])

    self.name = 'level %.4g' % self.level

  # ---------------------------------------------------------------------------
  #
  def _set_appearance(self, show_mesh, rendering_options):

    # Update color
    v = self.volume
    rgba = v._modulated_surface_color(self.rgba)
    self.color = tuple(int(255*r) for r in rgba)
    if self.auto_recolor_vertices is None:
      self.vertex_colors = None

    # Update display style
    ro = rendering_options
    # OpenGL draws nothing for degenerate triangles where two vertices are
    # identical.  For 2d contours want to see these triangles so show as mesh.
    contour_2d = v.single_plane() and not ro.cap_faces
    style = self.Mesh if show_mesh or contour_2d else self.Solid
    self.display_style = style

    # Update lighting
    if contour_2d:  lit = False
    elif show_mesh: lit = ro.mesh_lighting
    else:           lit = True
    self.use_lighting = lit

#    self.twoSidedLighting = ro.two_sided_lighting
#    self.lineThickness = ro.line_thickness
#    self.smoothLines = ro.smooth_lines

#     if ro.dim_transparency:
#       bmode = self.SRC_ALPHA_DST_1_MINUS_ALPHA
#     else:
#       bmode = self.SRC_1_DST_1_MINUS_ALPHA
#     self.transparencyBlendMode = bmode
      
  # ---------------------------------------------------------------------------
  #
  def _geometry_changed(self, rendering_options):
    v = self.volume
    ro = rendering_options
    contour_settings = {'level': self.level,
                        'matrix_id': v.matrix_id,
                        'transform': v.matrix_indices_to_xyz_transform(),
                        'surface_smoothing': ro.surface_smoothing,
                        'smoothing_factor': ro.smoothing_factor,
                        'smoothing_iterations': ro.smoothing_iterations,
                        'subdivide_surface': ro.subdivide_surface,
                        'subdivision_levels': ro.subdivision_levels,
                        'square_mesh': ro.square_mesh,
                        'cap_faces': ro.cap_faces,
                        'flip_normals': ro.flip_normals,
                        }
    changed = (self._contour_settings != contour_settings)
    if changed:
      self._contour_settings = contour_settings
    return changed

  # State save/restore in ChimeraX
  def take_snapshot(self, session, flags):
    data = {
      'volume': self.volume,
      'level': self.level,
      'rgba': self.rgba,
      'model state': Surface.take_snapshot(self, session, flags),
      'version': 1
    }
    return data

  @staticmethod
  def restore_snapshot(session, data):
    v = data['volume']
    if v is None:
      return None	# Volume was not restored, e.g. file missing.
    s = VolumeSurface(v, data['level'], data['rgba'])
    Model.set_state_from_snapshot(s, session, data['model state'])
    v._surfaces.append(s)
    return s
      
# -----------------------------------------------------------------------------
#
def maps_pickable(session, pickable):
  for m in session.models.list(type = Volume):
    m.pickable = pickable
  session._maps_pickable = pickable

# -----------------------------------------------------------------------------
#
from chimerax.core.graphics import Pick
class PickedMap(Pick):
  def __init__(self, v, distance = None, detail = ''):
    Pick.__init__(self, distance)
    self.map = v
    self.detail = detail
  def description(self):
    return '%s %s %s' % (self.map.id_string, self.map.name, self.detail)
  def select(self, mode = 'add'):
    m = self.map
    if mode == 'add':
      sel = True
    elif mode == 'subtract':
      sel = False
    elif mode == 'toggle':
      sel = not m.selected
    m.selected = sel
    for s in m.surfaces:
      s.selected = sel
    
# -----------------------------------------------------------------------------
#
class Outline_Box:

  def __init__(self, surface_model):

    self.model = surface_model
    self.piece = None
    self.corners = None
    self.rgb = None
    self.linewidth = None
    self.center = None
    self.planes = None
    self.crosshair_width = None
    
  # ---------------------------------------------------------------------------
  # The center and planes option are for orthoplane outlines.
  #
  def show(self, corners, rgb, linewidth,
           center = None, planes = None, crosshair_width = None):

    if not corners is None and not rgb is None:
      from numpy import array_equal
      changed = (self.corners is None or
                 not array_equal(corners, self.corners) or
                 rgb != self.rgb or
                 linewidth != self.linewidth or
                 not array_equal(center, self.center) or
                 planes != self.planes or
                 crosshair_width != self.crosshair_width)
      if changed:
        self.erase_box()
        self.make_box(corners, rgb, linewidth, center, planes, crosshair_width)
      
  # ---------------------------------------------------------------------------
  #
  def make_box(self, corners, rgb, linewidth, center, planes, crosshair_width):

    if center is None or planes is None or not True in planes:
      vlist = corners
      tlist = ((0,4,5), (5,1,0), (0,2,6), (6,4,0),
               (0,1,3), (3,2,0), (7,3,1), (1,5,7),
               (7,6,2), (2,3,7), (7,5,4), (4,6,7))
    else:
      vlist = []
      tlist = []
      self.plane_outlines(corners, center, planes, vlist, tlist)
      self.crosshairs(corners, center, planes, crosshair_width, vlist, tlist)
          
    b = 8 + 2 + 1    # Bit mask, 8 = show triangle, edges are bits 4,2,1
    hide_diagonals = [b]*len(tlist)

    rgba = tuple(rgb) + (1,)
    p = self.model.new_drawing('outline box')
    p.display_style = p.Mesh
    p.lineThickness = linewidth
    p.use_lighting = False
    p.pickable = False
    p.no_cofr = True
    # Set geometry after setting outline_box attribute to avoid undesired
    # coloring and capping of outline boxes.
    from numpy import array
    p.set_geometry(array(vlist), None, array(tlist))
    p.edge_mask = hide_diagonals
    p.color = tuple(int(255*r) for r in rgba)

    self.piece = p
    self.corners = corners
    self.rgb = rgb
    self.linewidth = linewidth
    self.center = center
    self.planes = planes
    self.crosshair_width = crosshair_width
    
  # ---------------------------------------------------------------------------
  #
  def plane_outlines(self, corners, center, planes, vlist, tlist):

    for a,p in enumerate(planes):
      if p:
        if a == 0:
          vp = [(center[0],corners[c][1],corners[c][2]) for c in (0,2,3,1)]
        elif a == 1:
          vp = [(corners[c][0],center[1],corners[c][2]) for c in (0,1,5,4)]
        elif a == 2:
          vp = [(corners[c][0],corners[c][1],center[2]) for c in (0,4,7,2)]
        v0 = len(vlist)
        vlist.extend(vp)
        tp = [(i0+v0,i1+v0,i2+v0) for i0,i1,i2 in ((0,1,2),(2,3,0))]
        tlist.extend(tp)

  # ---------------------------------------------------------------------------
  #
  def crosshairs(self, corners, center, planes, width, vlist, tlist):

    hw0,hw1,hw2 = [0.5*w for w in width]
    btlist = ((0,4,5), (5,1,0), (0,2,6), (6,4,0),
              (0,1,3), (3,2,0), (7,3,1), (1,5,7),
              (7,6,2), (2,3,7), (7,5,4), (4,6,7))
    from .data import box_corners
    if planes[1] and planes[2]:
      x0, x1 = corners[0][0], corners[4][0]
      v0 = len(vlist)
      vlist.extend(box_corners((x0,center[1]-hw1,center[2]-hw2),
                               (x1,center[1]+hw1,center[2]+hw2)))
      tlist.extend((i0+v0,i1+v0,i2+v0) for i0,i1,i2 in btlist)
    if planes[0] and planes[2]:
      y0, y1 = corners[0][1], corners[2][1]
      v0 = len(vlist)
      vlist.extend(box_corners((center[0]-hw0,y0,center[2]-hw2),
                               (center[0]+hw0,y1,center[2]+hw2)))
      tlist.extend((i0+v0,i1+v0,i2+v0) for i0,i1,i2 in btlist)
    if planes[0] and planes[1]:
      z0, z1 = corners[0][2], corners[1][2]
      v0 = len(vlist)
      vlist.extend(box_corners((center[0]-hw0,center[1]-hw1,z0),
                               (center[0]+hw0,center[1]+hw1,z1)))
      tlist.extend((i0+v0,i1+v0,i2+v0) for i0,i1,i2 in btlist)

  # ---------------------------------------------------------------------------
  #
  def erase_box(self):

    p = self.piece
    if not p is None:
      if not p.was_deleted:
        self.model.remove_drawing(p)
      self.piece = None
      self.corners = None
      self.rgb = None
      self.center = None
      self.planes = None
      self.crosshair_width = None
      
# -----------------------------------------------------------------------------
# Compute scale factor f minimizing norm of (f*v + u) over domain v >= level.
#
#   f = -(v,u)/|v|^2 where v >= level.
#
def minimum_rms_scale(v, u, level):

  from numpy import greater_equal, multiply, dot as inner_product

  # Make copy of v with values less than level set to zero.
  vc = v.copy()
  greater_equal(vc, level, vc)
  multiply(v, vc, vc)
  vc = vc.ravel()

  # Compute factor
  vcu = inner_product(vc,u.ravel())
  vcvc = inner_product(vc,vc)
  if vcvc == 0:
    f = 1
  else:
    f = -vcu/vcvc
    
  return f
  
# -----------------------------------------------------------------------------
#
def same_grid(v1, region1, v2, region2):

  same = ([tuple(i) for i in region1] == [tuple(i) for i in region2] and
          v1.data.ijk_to_xyz_transform.same(v2.data.ijk_to_xyz_transform) and
          v1.model_transform().same(v2.model_transform()))
  return same
    
# -----------------------------------------------------------------------------
# Remember visited subregions.
#
class Region_List:

  def __init__(self):

    self.region_list = []               # history
    self.current_index = None
    self.max_list_size = 32

    self.named_regions = []

  # ---------------------------------------------------------------------------
  #
  def insert_region(self, ijk_min, ijk_max):

    ijk_min_max = (tuple(ijk_min), tuple(ijk_max))
    ci = self.current_index
    rlist = self.region_list
    if ci == None:
      ni = 0
    elif ijk_min_max == rlist[ci]:
      return
    else:
      ni = ci + 1
    self.current_index = ni
    if ni < len(rlist) and ijk_min_max == rlist[ni]:
      return
    rlist.insert(ni, ijk_min_max)
    self.trim_list()
    
  # ---------------------------------------------------------------------------
  #
  def trim_list(self):

    if len(self.region_list) <= self.max_list_size:
      return

    if self.current_index > self.max_list_size//2:
      del self.region_list[0]
      self.current_index -= 1
    else:
      del self.region_list[-1]

  # ---------------------------------------------------------------------------
  #
  def other_region(self, offset):

    if self.current_index == None:
      return None, None

    i = self.current_index + offset
    if i < 0 or i >= len(self.region_list):
      return None, None

    self.current_index = i
    return self.region_list[i]

  # ---------------------------------------------------------------------------
  #
  def where(self):

    ci = self.current_index
    if ci == None:
      return 0, 0

    from_beginning = ci
    from_end = len(self.region_list)-1 - ci
    return from_beginning, from_end

  # ---------------------------------------------------------------------------
  #
  def region_names(self):

    return [nr[0] for nr in self.named_regions]

  # ---------------------------------------------------------------------------
  #
  def add_named_region(self, name, ijk_min, ijk_max):

    self.named_regions.append((name, (tuple(ijk_min), tuple(ijk_max))))

  # ---------------------------------------------------------------------------
  #
  def find_named_region(self, ijk_min, ijk_max):

    ijk_min_max = (tuple(ijk_min), tuple(ijk_max))
    for name, named_ijk_min_max in self.named_regions:
      if named_ijk_min_max == ijk_min_max:
        return name
    return None

  # ---------------------------------------------------------------------------
  #
  def named_region_bounds(self, name):

    index = self.named_region_index(name)
    if index == None:
      return None, None
    return self.named_regions[index][1]

  # ---------------------------------------------------------------------------
  #
  def named_region_index(self, name):

    try:
      index = self.region_names().index(name)
    except ValueError:
      index = None
    return index

  # ---------------------------------------------------------------------------
  #
  def remove_named_region(self, index):

    del self.named_regions[index]

# -----------------------------------------------------------------------------
#
class Rendering_Options:

  def __init__(self):

    self.show_outline_box = True
    self.outline_box_rgb = (1,1,1)
    self.outline_box_linewidth = 1
    self.limit_voxel_count = True           # auto-adjust step size
    self.voxel_limit = 16                   # Mvoxels
    self.color_modes = (
      'auto4', 'auto8', 'auto12', 'auto16',
      'opaque4', 'opaque8', 'opaque12', 'opaque16',
      'rgba4', 'rgba8', 'rgba12', 'rgba16',
      'rgb4', 'rgb8', 'rgb12', 'rgb16',
      'la4', 'la8', 'la12', 'la16',
      'l4', 'l8', 'l12', 'l16')
    self.color_mode = 'auto8'         # solid rendering pixel formats
                                      #  (auto|opaque|rgba|rgb|la|l)(4|8|12|16)
    self.projection_modes = ('auto', '2d-xyz', '2d-x', '2d-y', '2d-z')
    self.projection_mode = 'auto'           # auto, 2d-xyz, 2d-x, 2d-y, 2d-z, 3d
    self.bt_correction = False              # brightness and transparency
    self.minimal_texture_memory = False
    self.maximum_intensity_projection = False
    self.linear_interpolation = True
    self.dim_transparency = True            # for surfaces
    self.dim_transparent_voxels = True      # for solid rendering
    self.line_thickness = 1
    self.smooth_lines = False
    self.mesh_lighting = True
    self.two_sided_lighting = True
    self.flip_normals = False
    self.subdivide_surface = False
    self.subdivision_levels = 1
    self.surface_smoothing = False
    self.smoothing_iterations = 2
    self.smoothing_factor = .3
    self.square_mesh = False
    self.cap_faces = True
    self.box_faces = False              # solid rendering
    self.orthoplanes_shown = (False, False, False)
    self.orthoplane_positions = (0,0,0) # solid rendering

  # ---------------------------------------------------------------------------
  #
  def any_orthoplanes_shown(self):

    return true_count(self.orthoplanes_shown) > 0

  # ---------------------------------------------------------------------------
  #
  def show_orthoplane(self, axis, p):

    shown = list(self.orthoplanes_shown)
    shown[axis] = True
    self.orthoplanes_shown = tuple(shown)
    ijk = list(self.orthoplane_positions)
    ijk[axis] = p
    self.orthoplane_positions = tuple(ijk)

  # ---------------------------------------------------------------------------
  #
  def copy(self):

    ro = Rendering_Options()
    for key, value in self.__dict__.items():
      if not key.startswith('_'):
        setattr(ro, key, value)
    return ro

# -----------------------------------------------------------------------------
#
def true_count(seq):

  count = 0
  for s in seq:
    if s:
      count += 1
  return count

# -----------------------------------------------------------------------------
#
def clamp_ijk(ijk, ijk_min, ijk_max):

  return tuple(max(ijk_min[a], min(ijk_max[a], ijk[a])) for a in (0,1,2))

# -----------------------------------------------------------------------------
#
def clamp_region(region, size):

  from . import data
  r = data.clamp_region(region[:2], size) + tuple(region[2:])
  return r

# ---------------------------------------------------------------------------
# Return ijk step size so that voxels displayed is at or below the limit.
# The given ijk step size may be increased or decreased by powers of 2.
#
def ijk_step_for_voxel_limit(ijk_min, ijk_max, ijk_step, faces_per_axis,
                             limit_voxel_count, mvoxel_limit):

  def voxel_count(step, ijk_min = ijk_min, ijk_max = ijk_max,
                  fpa = faces_per_axis):
    return subarray_size(ijk_min, ijk_max, step, fpa)

  step = limit_voxels(voxel_count, ijk_step, limit_voxel_count, mvoxel_limit)
  return step

# ---------------------------------------------------------------------------
# For box style solid display 2 planes per axis are used.  For orthoplane
# display 1 plane per axis is shown.  Returns 0 for normal (all planes)
# display style.
#
def faces_per_axis(representation, box_faces, any_orthoplanes_shown):

  fpa = 0
  if representation == 'solid':
    if box_faces:
      fpa = 2
    elif any_orthoplanes_shown:
      fpa = 1
  return fpa

# ---------------------------------------------------------------------------
# Return ijk step size so that voxels displayed is at or below the limit.
# The given ijk step size may be increased or decreased by powers of 2.
#
def limit_voxels(voxel_count, ijk_step, limit_voxel_count, mvoxel_limit):

    if not limit_voxel_count:
      return ijk_step

    if mvoxel_limit == None:
      return ijk_step
    
    voxel_limit = int(mvoxel_limit * (2 ** 20))
    if voxel_limit < 1:
      return ijk_step

    step = ijk_step

    # Make step bigger until voxel limit met.
    while voxel_count(step) > voxel_limit:
      new_step = [2*s for s in step]
      if voxel_count(new_step) >= voxel_count(step):
        break
      step = new_step

    # Make step smaller until voxel limit exceeded.
    while tuple(step) != (1,1,1):
      new_step = [max(1, s//2) for s in step]
      if voxel_count(new_step) > voxel_limit:
        break
      step = new_step
    
    return step

# ---------------------------------------------------------------------------
#
def show_planes(v, axis, plane, depth = 1, extend_axes = []):

  p = int(plane)
  ro = v.rendering_options
  orthoplanes = v.showing_orthoplanes()
  if orthoplanes:
    if depth == 1:
      ro.show_orthoplane(axis, p)
      v._drawings_need_update()
      if not extend_axes:
        return
    else:
      orthoplanes = False
      v.set_parameters(orthoplanes_shown = (False, False, False))

  # Make sure requested plane number is in range.
  dsize = v.data.size

  # Set new display plane
  ijk_min, ijk_max, ijk_step = [list(b) for b in v.region]
  for a in extend_axes:
    ijk_min[a] = 0
    ijk_max[a] = dsize[a]-1

  if orthoplanes:
    def set_plane_range(step):
      pass
  else:
    def set_plane_range(step, planes = dsize[axis]):
      astep = step[axis]
      ijk_min[axis] = max(0, min(p, planes - depth*astep))
      ijk_max[axis] = max(0, min(planes-1, p+depth*astep-1))

  fpa = faces_per_axis(v.representation, ro.box_faces,
                       ro.any_orthoplanes_shown())
  def voxel_count(step, fpa=fpa):
    set_plane_range(step)
    return subarray_size(ijk_min, ijk_max, step, fpa)

  # Adjust step size to limit voxel count.
  step = limit_voxels(voxel_count, ijk_step,
                      ro.limit_voxel_count, ro.voxel_limit)
  set_plane_range(step)

  changed = v.new_region(ijk_min, ijk_max, step)
  return changed

# -----------------------------------------------------------------------------
#
class cycle_through_planes:

  def __init__(self, v, session, axis, pstart, pend = None, pstep = 1, pdepth = 1):

    axis = {'x':0, 'y':1, 'z':2}.get(axis, axis)
    if pend is None:
      pend = pstart
    if pend < 0:
      pend = v.data.size[axis] + pend
    if pstart < 0:
      pstart = v.data.size[axis] + pstart
    if pend < pstart:
      pstep *= -1

    self.volume = v
    self.session = session
    self.axis = axis
    self.plane = pstart
    self.plast = pend
    self.step = pstep
    self.depth = pdepth

    self.handler = self.next_plane_cb
    session.triggers.add_handler('new frame', self.handler)

  def next_plane_cb(self, *_):
    
    p = self.plane
    if self.step * (self.plast - p) >= 0:
      self.plane += self.step
      show_planes(self.volume, self.axis, p, self.depth)
    else:
      self.handler = None
      from chimerax.core.triggerset import DEREGISTER
      return DEREGISTER

# -----------------------------------------------------------------------------
#
def subarray_size(ijk_min, ijk_max, step, faces_per_axis = 0):

  pi,pj,pk = [max(ijk_max[a]//step[a] - (ijk_min[a]+step[a]-1)//step[a] + 1, 1)
              for a in (0,1,2)]
  if faces_per_axis == 0:
    voxels = pi*pj*pk
  else:
    voxels = faces_per_axis * (pi*pj + pi*pk + pj*pk)
  return voxels

# -----------------------------------------------------------------------------
#
def full_region(size, ijk_step = [1,1,1]):

  ijk_min = [0, 0, 0]
  ijk_max = [n-1 for n in size]
  region = (ijk_min, ijk_max, ijk_step)
  return region

# -----------------------------------------------------------------------------
#
def is_empty_region(region):

  ijk_min, ijk_max, ijk_step = region
  for a,b in zip(ijk_max, ijk_min):
    if a - b + 1 <= 0:
      return True
  return False

# ---------------------------------------------------------------------------
# Adjust volume region to include a zone.  If current volume region is
# much bigger than that needed for the zone, then shrink it.  The purpose
# of this resizing is to keep the region small so that recontouring is fast,
# but not resize on every new zone radius.  Resizing on every new zone
# radius requires recontouring and redisplaying the volume histogram which
# slows down zone radius updates.
#
def resize_region_for_zone(data_region, points, radius, initial_resize = False):

  from .data import points_ijk_bounds
  ijk_min, ijk_max = points_ijk_bounds(points, radius, data_region.data)
  ijk_min, ijk_max = clamp_region((ijk_min, ijk_max, None),
                                  data_region.data.size)[:2]

  cur_ijk_min, cur_ijk_max = data_region.region[:2]

  volume_padding_factor = 2.0
  min_volume = 32768

  if not region_contains_region((cur_ijk_min, cur_ijk_max),
                                (ijk_min, ijk_max)):
    import math
    padding_factor = math.pow(volume_padding_factor, 1.0/3)
  else:
    cur_volume = region_volume(cur_ijk_min, cur_ijk_max)
    box_volume = region_volume(ijk_min, ijk_max)
    if cur_volume <= 2 * box_volume or cur_volume <= min_volume:
      return None, None
    import math
    if initial_resize:
      # Pad zone to grow or shrink before requiring another resize
      padding_factor = math.pow(volume_padding_factor, 1.0/6)
    else:
      padding_factor = 1
    new_box_volume = box_volume * math.pow(padding_factor, 3)
    if new_box_volume < min_volume and box_volume > 0:
      # Don't resize to smaller than the minimum volume
      import math
      padding_factor = math.pow(float(min_volume) / box_volume, 1.0/3)

  new_ijk_min, new_ijk_max = extend_region(ijk_min, ijk_max, padding_factor)

  return new_ijk_min, new_ijk_max

# -----------------------------------------------------------------------------
#
def region_contains_region(r1, r2):

  for a in range(3):
    if r2[0][a] < r1[0][a] or r2[1][a] > r1[1][a]:
      return False
  return True

# -----------------------------------------------------------------------------
#
def region_volume(ijk_min, ijk_max):

  vol = 1
  for a in range(3):
    vol *= ijk_max[a] - ijk_min[a] + 1
  return vol

# -----------------------------------------------------------------------------
#
def extend_region(ijk_min, ijk_max, factor):

  e_ijk_min = list(ijk_min)
  e_ijk_max = list(ijk_max)
  for a in range(3):
    pad = int(.5 * (factor - 1) * (ijk_max[a] - ijk_min[a]))
    e_ijk_min[a] -= pad
    e_ijk_max[a] += pad

  return tuple(e_ijk_min), tuple(e_ijk_max)

# -----------------------------------------------------------------------------
#
def maximum_data_diagonal_length(data):

    imax, jmax, kmax = [a-1 for a in data.size]
    ijk_to_xyz = data.ijk_to_xyz
    from chimerax.core.geometry import distance
    d = max(distance(ijk_to_xyz((0,0,0)), ijk_to_xyz((imax,jmax,kmax))),
            distance(ijk_to_xyz((0,0,kmax)), ijk_to_xyz((imax,jmax,0))),
            distance(ijk_to_xyz((0,jmax,0)), ijk_to_xyz((imax,0,kmax))),
            distance(ijk_to_xyz((0,jmax,kmax)), ijk_to_xyz((imax,0,0))))
    return d

# -----------------------------------------------------------------------------
#
def mean_sd_rms(m):

    from numpy import float64
    mean = m.mean(dtype=float64)
    sd = m.std(dtype=float64)
    from math import sqrt
    rms = sqrt(sd*sd + mean*mean)
    return mean, sd, rms

# -----------------------------------------------------------------------------
#
def transformed_points(points, tf):

  from numpy import array, single as floatc
  tf_points = array(points, floatc)
  tf.transform_points(tf_points, in_place = True)
  return tf_points
    
# -----------------------------------------------------------------------------
#
def saturate_rgba(rgba):

  mc = max(rgba[:3])
  if mc > 0:
    s = rgba[0]/mc, rgba[1]/mc, rgba[2]/mc, rgba[3]
  else:
    s = rgba
  return s

# ----------------------------------------------------------------------------
# Return utf-8 encoding in a plain (non-unicode) string.  This is needed
# for setting C++ model name.
#
def utf8_string(s):

#  if isinstance(s, unicode):
#  return s.encode('utf-8')
  return s

# ----------------------------------------------------------------------------
# Use a periodic unit cell map to create a new map that covers a PDB model
# plus some padding.  Written for Terry Lang.
#
def map_covering_atoms(atoms, pad, volume):

    ijk_min, ijk_max = atom_bounds(atoms, pad, volume)
    g = map_from_periodic_map(volume.data, ijk_min, ijk_max)
    v = volume_from_grid_data(g, volume.session)
    v.copy_settings_from(volume, copy_region = False)
    return v

# ----------------------------------------------------------------------------
#
def atom_bounds(atoms, pad, volume):

    # Get atom positions.
    xyz = atoms.scene_coords

    # Transform atom coordinates to volume ijk indices.
    tf = volume.data.xyz_to_ijk_transform * volume.position.inverse()
    tf.transform_points(xyz, in_place = True)
    ijk = xyz

    # Find integer bounds.
    from math import floor, ceil
    ijk_min = [int(floor(i-pad)) for i in ijk.min(axis=0)]
    ijk_max = [int(ceil(i+pad)) for i in ijk.max(axis=0)]

    return ijk_min, ijk_max

# ----------------------------------------------------------------------------
#
def map_from_periodic_map(grid, ijk_min, ijk_max):

    # Create new 3d array.
    ijk_size = [a-b+1 for a,b in zip(ijk_max, ijk_min)]
    kji_size = tuple(reversed(ijk_size))
    from numpy import zeros
    m = zeros(kji_size, grid.value_type)

    # Fill in new array using periodicity of original array.
    # Find all overlapping unit cells and copy needed subblocks.
    gsize = grid.size
    cell_min = [a//b for a,b in zip(ijk_min, gsize)]
    cell_max = [a//b for a,b in zip(ijk_max, gsize)]
    for kc in range(cell_min[2], cell_max[2]+1):
        for jc in range(cell_min[1], cell_max[1]+1):
            for ic in range(cell_min[0], cell_max[0]+1):
                ijkc = (ic,jc,kc)
                ijk0 = [max(a*c,b)-b for a,b,c in zip(ijkc, ijk_min, gsize)]
                ijk1 = [min((a+1)*c,b+1)-d for a,b,c,d in 
                        zip(ijkc, ijk_max, gsize, ijk_min)]
                size = [a-b for a,b in zip(ijk1, ijk0)]
                origin = [max(0, b-a*c) for a,b,c in zip(ijkc, ijk_min, gsize)]
                cm = grid.matrix(origin, size)
                m[ijk0[2]:ijk1[2],ijk0[1]:ijk1[1],ijk0[0]:ijk1[0]] = cm

    # Create volume data copy.
    xyz_min = grid.ijk_to_xyz(ijk_min)
    from .data import ArrayGridData
    g = ArrayGridData(m, xyz_min, grid.step, grid.cell_angles, grid.rotation,
                      name = grid.name)
    return g

# -----------------------------------------------------------------------------
# Open and display a map.
#
def open_volume_file(path, session, format = None, name = None, representation = None,
                     open_models = True, model_id = None,
                     show_data = True, show_dialog = True):

  from . import data
  try:
    glist = data.open_file(path, format)
  except data.FileFormatError as value:
    raise
    from os.path import basename
    if isinstance(path, (list,tuple)):
      descrip = '%s ... (%d files)' % (basename(path[0]), len(path))
    else:
      descrip = basename(path)
    msg = 'Error reading file ' + descrip
    if format:
      msg += ', format %s' % format
    msg += '\n%s\n' % str(value)
    from chimera.replyobj import error
    error(msg)
    return []

  if not name is None:
    for g in glist:
      g.name = name

  vlist = [volume_from_grid_data(g, session, representation, open_models,
                                 model_id, show_dialog)
            for g in glist]

  if not show_data:
    for v in vlist:
      v.display = False
      
  return vlist

# -----------------------------------------------------------------------------
#
def default_settings(session):
  if not hasattr(session, 'volume_defaults'):
    from . import defaultsettings
    session.volume_defaults = defaultsettings.VolumeDefaultSettings()
  return session.volume_defaults

# -----------------------------------------------------------------------------
#
def set_data_cache(grid_data, session):
  from .data import ArrayGridData
  if isinstance(grid_data, ArrayGridData):
    return	# No caching for in-memory maps

  grid_data.data_cache = data_cache(session)

# -----------------------------------------------------------------------------
#
def data_cache(session):
  dc = getattr(session, '_volume_data_cache', None)
  if dc is None:
    ds = default_settings(session)
    size = ds['data_cache_size'] * (2**20)
    from .data import datacache
    session._volume_data_cache = dc = datacache.Data_Cache(size = size)
  return dc

# -----------------------------------------------------------------------------
# Open and display a map using Volume Viewer.
#
def volume_from_grid_data(grid_data, session, representation = None,
                          open_model = True, model_id = None, show_dialog = True):
  '''
  Supported API.
  Create a new Volume model from a GridData instance and set its initial 
  display style and color and add it to the session open models.
  '''
  
  set_data_cache(grid_data, session)

  ds = default_settings(session)
  ro = ds.rendering_option_defaults()
  
  # Create volume model
  d = data_already_opened(grid_data.path, grid_data.grid_id, session)
  if d:
    grid_data = d
    
  v = Volume(session, grid_data, rendering_options = ro)
  
  # Set display style
  if representation is None:
    # Show single plane data in solid style.
    single_plane = [s for s in grid_data.size if s == 1]
    representation = 'solid' if single_plane else 'surface'
  v.set_representation(representation)
  
  if grid_data.rgba is None:
    set_initial_volume_color(v, session)

  if not model_id is None:
    v.id = model_id

  if open_model:
    session.models.add([v])

  if show_dialog:
    show_volume_dialog(session)

  return v

# -----------------------------------------------------------------------------
#
def show_volume_dialog(session):
  from .volume_viewer import show_volume_dialog
  show_volume_dialog(session)

# -----------------------------------------------------------------------------
#
class CancelOperation(BaseException):
  pass

# -----------------------------------------------------------------------------
# Decide whether a data region is small enough to show when opened.
#
def show_when_opened(v, show_on_open, max_voxels):

  if not show_on_open:
    return False
  
  if max_voxels == None:
    return False
  
  voxel_limit = int(max_voxels * (2 ** 20))
  ijk_origin, ijk_size, ijk_step = v.ijk_aligned_region()
  sx,sy,sz = [s//st for s,st in zip(ijk_size, ijk_step)]
  voxels = sx*sy*sz

  return (voxels <= voxel_limit)

# -----------------------------------------------------------------------------
# Decide whether a data region is large enough that only a single z plane
# should be shown.
#
def show_one_plane(size, show_plane, min_voxels):

  if not show_plane:
    return False
  
  if min_voxels == None:
    return False
  
  voxel_limit = int(min_voxels * (2 ** 20))
  voxels = float(size[0]) * float(size[1]) * float(size[2])

  return (voxels >= voxel_limit)
    
# ---------------------------------------------------------------------------
#
def set_initial_volume_color(v, session):

  ds = default_settings(session)
  if ds['use_initial_colors']:
    i = 0 if session.models.empty() else getattr(ds, '_next_color_index', 0)
    if not hasattr(v.data, 'series_index') or v.data.series_index == 0:
      ds._next_color_index = i+1
    icolors = ds['initial_colors']
    rgba = icolors[i%len(icolors)]
    v.set_parameters(default_rgba = rgba)

# ---------------------------------------------------------------------------
#
def data_already_opened(path, grid_id, session):

  if not path:
    return None
  
  for v in volume_list(session):
    d = v.data
    if not d.writable and d.path == path and d.grid_id == grid_id:
        return d
  return None

# -----------------------------------------------------------------------------
#
def volume_list(session):
  return [m for m in session.models.list() if isinstance(m, Volume)]

# -----------------------------------------------------------------------------
#
def open_map(session, stream, name = None, format = None, **kw):
    '''
    Supported API. Open a density map file having any of the known density map formats.
    '''
    if isinstance(stream, (str, list)):
      map_path = stream         # Batched paths
    else:
      map_path = stream.name
      stream.close()
    from os.path import basename
    name = basename(map_path if isinstance(map_path, str) else map_path[0])

    from . import data
    grids = data.open_file(map_path, file_type = format)

    if grids and isinstance(grids[0], (tuple, list)):
      # handle multiple channels.
      models = []
      for cgrids in grids:
        cmodels, msg = open_grids(session, cgrids, name, **kw)
        models.extend(cmodels)
    else:
      models, msg = open_grids(session, grids, name, **kw)
    return models, msg

# -----------------------------------------------------------------------------
#
def open_grids(session, grids, name, **kw):

    if kw.get('polar_values', False):
      for g in grids:
        g.polar_values = True
        if g.rgba is None:
          g.rgba = (0,1,0,1) # Green

    channel = kw.get('channel', None)
    if channel is not None:
      for g in grids:
        g.channel = channel
        
    series = kw.get('vseries', None)
    if series is not None:
      if series:
        for i,g in enumerate(grids):
          if tuple(g.size) != tuple(grids[0].size):
            gsizes = '\n'.join((g.name + (' %d %d %d' % g.size)) for g in grids)
            raise UserError('Cannot make series from volumes with different sizes:\n%s' % gsizes)
          g.series_index = i
      else:
        for g in grids:
          if hasattr(g, 'series_index'):
            delattr(g, 'series_index')

    maps = []
    show = kw.get('show', True)
    si = [d.series_index for d in grids if hasattr(d, 'series_index')]
    is_series = (len(si) == len(grids) and len(set(si)) > 1)
    cn = [d.channel for d in grids if d.channel is not None]
    is_multichannel = (len(cn) == len(grids) and len(set(cn)) > 1)
    for d in grids:
      show_data = show
      if is_series or is_multichannel:
        show_data = False	# MapSeries or MapChannelsModel classes will decide which to show
      vkw = {'show_dialog': False}
      if hasattr(d, 'initial_style') and d.initial_style in ('surface', 'mesh', 'solid'):
        vkw['representation'] = d.initial_style
      v = volume_from_grid_data(d, session, open_model = False, **vkw)
      maps.append(v)
      if not show_data:
        v.display = False
      set_initial_region_and_style(v)

    show_dialog = kw.get('show_dialog', True)
    if maps and show_dialog:
      show_volume_dialog(session)

    if is_series and is_multichannel:
      cmaps = {}
      for m in maps:
        cmaps.setdefault(m.data.channel,[]).append(m)
      if len(set(len(cm) for cm in cmaps.values())) > 1:
        session.logger.warning('Map channels have differing numbers of series maps: %s'
                               % ', '.join('%d (%d)' % (c,cm) for c, cm in cmaps.items()))
      from .series import MapSeries
      ms = [MapSeries('channel %d' % c, cm, session) for c, cm in cmaps.items()]
      mc = MultiChannelSeries(name, ms, session)
      msg = ('Opened multichannel map series %s, %d channels, %d images per channel'
             % (name, len(ms), len(maps)//len(ms)))
      models = [mc]
    elif is_series:
      from .series import MapSeries
      ms = MapSeries(name, maps, session)
      msg = 'Opened map series %s, %d images' % (name, len(maps))
      models = [ms]
    elif is_multichannel:
      mc = MapChannelsModel(name, maps, session)
      mc.show_n_channels(3)
      msg = 'Opened multi-channel map %s, %d channels' % (name, len(maps))
      models = [mc]
    else:
      # TODO: Handle multi-times and multi-channels.
      msg = 'Opened %s' % maps[0].name
      models = maps

    # Initialize thresholds before adding to session so that initial view can use corrrect bounds.
    for v in maps:
      if v.display:
        v.initialize_thresholds()
        
    m0 = maps[0]
    px,py,pz = m0.data.step
    psize = '%.3g' % px if py == px and pz == px else '%.3g,%.3g,%.3g' % (px,py,pz)
    msg += (', grid size %d,%d,%d' % tuple(m0.data.size) +
            ', pixel %s' % psize +
            ', shown at ')
    if m0.representation == 'surface':
      msg += 'level %s, ' % ','.join('%.3g' % s.level for s in m0.surfaces)
    sx,sy,sz = m0.region[2]
    step = '%d' % sx if sy == sx and sz == sx else '%d,%d,%d' % (sx,sy,sz)
    msg += 'step %s' % step
    msg += ', values %s' % m0.data.value_type.name
    
    return models, msg

# -----------------------------------------------------------------------------
#
def set_initial_region_and_style(v):
  ds = default_settings(v.session)

  if not show_when_opened(v, ds['show_on_open'], ds['voxel_limit_for_open']):
    v.display = False
 
  ro = v.rendering_options
  if getattr(v.data, 'polar_values', False):
    ro.flip_normals = True
    ro.cap_faces = False

  one_plane = show_one_plane(v.data.size, ds['show_plane'], ds['voxel_limit_for_plane'])
  if one_plane:
    v.set_representation('solid')
    
  # Determine initial region bounds and step.
  region = v.full_region()[:2]
  if one_plane:
    region[0][2] = region[1][2] = v.data.size[2]//2
    
  fpa = faces_per_axis(v.representation, ro.box_faces, ro.any_orthoplanes_shown())
  ijk_step = ijk_step_for_voxel_limit(region[0], region[1], (1,1,1), fpa,
                                      ro.limit_voxel_count, ro.voxel_limit)
  region = tuple(region) + (ijk_step,)
  v.new_region(*region, adjust_step = False)

# -----------------------------------------------------------------------------
#
class MapChannels:
  
  def __init__(self, maps):
    self.set_maps(maps)

  def set_maps(self, maps):
    for v in maps:
      v._channels = self
    self.maps = maps

  def show_n_channels(self, n):
    # Hide all but lowest N channels.
    # Allen Institute data sometimes has 8 channels, mostly segmentations.
    maps = self.maps
    if len(maps) == 0:
      return
    channels = [v.data.channel for v in maps]
    channels.sort()
    channel_show_max = channels[min(2,len(channels)-1)]
    for v in maps:
      v.display = (v.data.channel <= channel_show_max)
    
  @property
  def first_channel(self):
    return self.maps[0]

# -----------------------------------------------------------------------------
#
class MapChannelsModel(Model, MapChannels):
  
  def __init__(self, name, maps, session):
    Model.__init__(self, name, session)
    self.add(maps)
    MapChannels.__init__(self, maps)

  # State save/restore in ChimeraX
  def take_snapshot(self, session, flags):
    data = {'model state': Model.take_snapshot(self, session, flags),
            # Can't reference maps directly because it creates cyclic dependency.
            'map ids': [m.id for m in self.maps],
            'version': 1}
    return data

  @staticmethod
  def restore_snapshot(session, data):
    maps = []
    c = MapChannelsModel('channels', maps, session)
    Model.set_state_from_snapshot(c, session, data['model state'])

    # Parent models are always restored before child models.
    # Restore child map list after child maps are restored.
    def restore_maps(trigger_name, session, channels = c, map_ids = data['map ids']):
      idm = {m.id : m for m in channels.child_models()}
      maps = [idm[id] for id in map_ids if id in idm]
      channels.set_maps(maps)
      from chimerax.core.triggerset import DEREGISTER
      return DEREGISTER
    session.triggers.add_handler('end restore session', restore_maps)
    
    return c

# -----------------------------------------------------------------------------
#
class MultiChannelSeries(Model):
  
  def __init__(self, name, map_series, session):
    Model.__init__(self, name, session)
    self.add(map_series)
    self.set_map_series(map_series)

  def set_map_series(self, map_series):
    self.map_series = map_series
    if map_series:
      # For each time, group the map channels
      for maps in zip(*tuple(ms.maps for ms in map_series)):
        mc = MapChannels(maps)
        for m in maps:
          m._channels = mc

  # State save/restore in ChimeraX
  def take_snapshot(self, session, flags):
    data = {'model state': Model.take_snapshot(self, session, flags),
            # Can't reference maps directly because it creates cyclic dependency.
            'map series ids': [m.id for m in self.map_series],
            'version': 1}
    return data

  @staticmethod
  def restore_snapshot(session, data):
    map_series = []
    mcs = MultiChannelSeries('mcs', map_series, session)
    Model.set_state_from_snapshot(mcs, session, data['model state'])

    # Parent models are always restored before child models.
    # Restore child map list after child maps are restored.
    def restore_maps(trigger_name, session, mcs = mcs, map_ids = data['map ids']):
      idm = {m.id : m for m in mcs.child_models()}
      map_series = [idm[id] for id in map_ids if id in idm]
      channels.set_map_series(map_series)
      from chimerax.core.triggerset import DEREGISTER
      return DEREGISTER
    session.triggers.add_handler('end restore session', restore_maps)
    
    return mcs
  
# -----------------------------------------------------------------------------
#
def save_map(session, path, format_name, models = None, region = None, step = (1,1,1),
             mask_zone = True, chunk_shapes = None, append = None, compress = None,
             base_index = 1, **kw):
    '''
    Save a density map file having any of the known density map formats.
    '''
    if models is None:
        vlist = session.models.list(type = Volume)
        if len(vlist) != 1:
            from chimerax.core.errors import UserError
            raise UserError('No model specified for saving map')
    elif len(models) == 0:
        from chimerax.core.errors import UserError
        raise UserError('No volumes specified')
    else:
      vlist = [m for m in models if isinstance(m, Volume)]
      if len(vlist) == 0:
          mstring = ' (#%s)' % ','.join(model.id_string for m in models) if models else ''
          from chimerax.core.errors import UserError
          raise UserError('Specified models are not volumes' + mstring)

      
    from .data.fileformats import file_writer, file_writers
    if file_writer(path, format_name) is None:
        from chimerax.core.errors import UserError
        if format_name is None:
            msg = ('Unknown file suffix for "%s", known suffixes %s'
                   % (path, ', '.join(sum([fw[2] for fw in file_writers], []))))
        else:
            msg = ('Unknown file format "%s", known formats %s'
                   % (format_name, ', '.join(fw[1] for fw in file_writers)))
        raise UserError(msg)
        
    options = {}
    if chunk_shapes is not None:
        options['chunk_shapes'] = chunk_shapes
    if append:
        options['append'] = True
    if compress:
        options['compress'] = True
    if path in ('browse', 'browser'):
        from .data import select_save_path
        path, format_name = select_save_path()
    if path:
        grids = []
        for v in vlist:
          g = v.grid_data(region, step, mask_zone)
          color = v.single_color
          if color is not None:
            g.rgba = tuple(r/255 for r in color)	# Set default map color to current color
          grids.append(g)
        from .data import save_grid_data
        if is_multifile_save(path):
            for i,g in enumerate(grids):
                save_grid_data(g, path % (i + base_index), session, format_name, options)
        else:
            save_grid_data(grids, path, session, format_name, options)

# -----------------------------------------------------------------------------
# 
class VolumeUpdateManager:
  def __init__(self, session):
    self._volumes_to_update = set()
    # Only update displayed volumes.  Keep list or efficiency with time series.
    self._displayed_volumes_to_update = set()
    t = session.triggers
    t.add_handler('graphics update', self._update_drawings)
    t.add_handler('model display changed', self._display_change)
    
  def add(self, v):
    self._volumes_to_update.add(v)
    if v.display and v.parents_displayed:
      self._displayed_volumes_to_update.add(v)

  def _display_change(self, tname, m):
    if m.display and m.parents_displayed:
      vset = self._volumes_to_update
      vlist = [v for v in m.all_models() if v in vset and v.display and v.parents_displayed]
      self._displayed_volumes_to_update.update(vlist)

  def _update_drawings(self, *_):
    vdisp = self._displayed_volumes_to_update
    if vdisp:
      vset = self._volumes_to_update
      for v in tuple(vdisp):
        if v.display:
          # Remove volume from update list before update since update may re-add it
          # if surface calculation done in thread.
          vset.remove(v)
          vdisp.remove(v)
          if not v.initialized_thresholds:
            v.initialize_thresholds()
          v.update_drawings()
   
# -----------------------------------------------------------------------------
# Check if file name contains %d type format specification.
#
def is_multifile_save(path):
    try:
        path % 0
    except:
        return False
    return True

# -----------------------------------------------------------------------------
#
def register_map_file_formats(session):
    from chimerax.core import io, toolshed
    from .data.fileformats import file_types, file_writers
    fwriters = set(fw[0] for fw in file_writers)
    for d,t,nicknames,suffixes,batch in file_types:
      suf = tuple('.' + s for s in suffixes)
      save_func = save_map if d in fwriters else None
      def open_map_format(session, stream, name = None, format = t, **kw):
        return open_map(session, stream, name=name, format=format, **kw)
      io.register_format(d, toolshed.VOLUME, suf, nicknames=nicknames,
                         open_func=open_map_format, batch=True, export_func=save_func)

    # Add keywords to open command for maps
    from chimerax.core.commands import BoolArg, IntArg
    from chimerax.core.commands.cli import add_keyword_arguments
    add_keyword_arguments('open', {'vseries':BoolArg, 'channel':IntArg})

    # Add keywords to save command for maps
    from chimerax.core.commands import BoolArg, ListOf, EnumOf, IntArg
    from .mapargs import MapRegionArg, Int1or3Arg
    save_map_args = [
      ('region', MapRegionArg),
      ('step', Int1or3Arg),
      ('mask_zone', BoolArg),
      ('chunk_shapes', ListOf(EnumOf(('zyx','zxy','yxz','yzx','xzy','xyz')))),
      ('append', BoolArg),
      ('compress', BoolArg),
      ('base_index', IntArg),
    ]
    add_keyword_arguments('save', dict(save_map_args))

    # Register save map subcommand
    from chimerax.core.commands import CmdDesc, register, SaveFileNameArg, ModelsArg
    from chimerax.core.commands.save import SaveFileFormatsArg, save
    from chimerax.core import toolshed
    desc = CmdDesc(
        required=[('filename', SaveFileNameArg)],
        optional=[('models', ModelsArg)],
        keyword=[('format', SaveFileFormatsArg(toolshed.VOLUME))] + save_map_args,
        synopsis='save map'
    )
    register('save map', desc, save, logger=session.logger)
