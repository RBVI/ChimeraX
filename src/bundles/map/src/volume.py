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
# Manages surface and volume display for a region of a data set.
# Holds surface and image rendering thresholds, color, and transparency and brightness
# factors.
#
from chimerax.core.models import Model
class Volume(Model):
  '''
  A Volume is a Model that renders a 3-d image of a :class:`~.data.GridData` object.
  It includes color, display styles including surface, mesh and grayscale, contouring levels,
  brightness and transparency for grayscale rendering, region bounds for display
  a subregion including single plane display, subsampled display of every Nth data
  value along each axis, outline box display.

  Attributes
  ----------
  session : :class:`~chimerax.core.session.Session`
      The session that the Volume will belong to.
  data : :class:`~.data.GridData`
      3D data array
  region : (ijk_min, ijk_max, ijk_step)
      Initial displayed subregion of 3D array
  rendering_options : :class:`.RenderingOptions`
      Appearance settings for surface and image display.
  '''
  def __init__(self, session, data, region = None, rendering_options = None):

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
      rendering_options = ds.rendering_option_defaults()
    self.rendering_options = rendering_options

    self.message_cb = session.logger.status

    self.matrix_stats = None
    self._matrix_id = 1          # Incremented when shape or values change.

    rlist = Region_List()
    ijk_min, ijk_max = self.region[:2]
    rlist.insert_region(ijk_min, ijk_max)
    self.region_list = rlist

    self._channels = None	# MapChannels object

    self._keep_displayed_data = None

    self._style_when_shown = 'surface'

    # Surface display submodels
    self._surfaces = []				# VolumeSurface instances

    self.outline_box = OutlineBox(self)

    # Image display submodel and parameters
    self._image = None
    self.image_levels = []                      # list of (threshold, scale)
    self.image_colors = []
    self._mask_colors = None			# For coloring by segmentation
    self._segment_colors = None			# For coloring segmentations
    self.transparency_depth = 0.5               # for image rendering
    self.image_brightness_factor = 1

    self.default_rgba = data.rgba if data.rgba else (.7,.7,.7,1)

#    from chimera import triggers
#    h = triggers.addHandler('SurfacePiece', self.surface_piece_changed_cb, None)
#    self.surface_piece_change_handler = h
    self.surface_piece_change_handler = None

    self.model_panel_show_expanded = False	# Don't show submodels initially in model panel

  # ---------------------------------------------------------------------------
  #
  def show_info(self):
    self.data.show_info()

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
  def info_string(self):

    px,py,pz = self.data.step
    psize = '%.3g' % px if py == px and pz == px else '%.3g,%.3g,%.3g' % (px,py,pz)
    info = ('grid size %d,%d,%d' % tuple(self.data.size) +
            ', pixel %s' % psize +
            ', shown at ')
    if self.surface_shown:
      info += 'level %s, ' % ','.join('%.3g' % s.level for s in self.surfaces)
    sx,sy,sz = self.region[2]
    step = '%d' % sx if sy == sx and sz == sx else '%d,%d,%d' % (sx,sy,sz)
    info += 'step %s' % step
    info += ', values %s' % self.data.value_type.name
    if hasattr(self, 'fit_pdb_ids') and self.fit_pdb_ids:
      pdb_links = ', '.join(f'<a href="cxcmd: open {pdb_id}">{pdb_id}</a>'
                            for pdb_id in self.fit_pdb_ids)
      info += f', fit PDB {pdb_links}'
    return info

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
    if getattr(self, 'series', None) is None and self._channels is None:
      msg = 'Opened %s as #%s, %s' % (self.name, self.id_string, self.info_string())
      session.logger.info(msg, is_html = ('cxcmd' in msg))

    # Use full lighting for initial map display
    if len(session.models.list()) == 1:
      from chimerax.std_commands.lighting import lighting
      lighting(session, 'full')

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
    '''Supported API.  Return a list of :class:`.VolumeSurface` instances for this Volume.'''
    return self._surfaces

  # ---------------------------------------------------------------------------
  #
  def add_surface(self, level, rgba = (.7,.7,.7,1), display = True):
    '''
    Supported API.  Create and add a new :class:`.VolumeSurface` with
    specified contour level and color.
    '''
    ses = self.session
    s = VolumeSurface(self, level, rgba)
    s.display = display
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
    Supported API.  Remove a list of :class:`.VolumeSurface` instances from this Volume.
    If surfaces is None then all current surfaces are removed.
    '''
    surfs = tuple(self._surfaces if surfaces is None else surfaces)
    for surf in surfs:
      surf.delete()

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
  def set_parameters(self,
                     surface_levels = None,
                     surface_colors = None,
                     transparency = None,
                     brightness = None,
                     image_levels = None,
                     image_colors = None,
                     transparency_depth = None,
                     image_brightness_factor = None,
                     default_rgba = None,
                     **rendering_options):
    '''
    Set volume display parameters.

    Parameters
    ----------
    surface_levels : list of float
      Threshold levels for contour surfaces.
    surface_colors : list of (r,g,b) or (r,g,b,a)
      Color for each surface level, color components have 0-1 range.
    transparency : float
      Surface transparency, 0 = fully opaque, 1 = fully transparent.
    brightness : float
      Scale surface brightness by this factor.
    image_levels : list of (float, float)
      Pairs of (threshold, brightness) where threshold is a map value
      and brighness ranges from 0-1.  This defines a piecewise linear
      brightness curve for image style rendering.
    image_colors : list of (r,g,b) or (r,g,b,a)
      Color associated with each image level, color components have 0-1 range.
    transparency_depth : float
      Controls how transparent image style renderings are, range 0-1.
      Image rendering makes opacity equal to brightness, ie. full brightness (= 1)
      image levels are fully opaque, and 0 brightness levels are fully transparent.
      The thickness that produces this transparency is the displayed region size
      multiplied by the transparency depth, where the region size is the size along
      the axis (x, y, or z) having fewest grid points.
    image_brightness_factor : float
      Scale image style rendering by this factor.
    default_rgba : 4 floats
      Initial color (red, green, blue, alpha) to use for surface and
      image style renderings.  Color components in range 0-1.
    rendering_options : all additional settings
      Any RenderingOption attribute name and value can be specified
      as a keyword option.
    '''

    kw = rendering_options.copy()
    parameters = ('surface_levels',
                  'surface_colors',
                  'transparency',
                  'brightness',
                  'image_levels',
                  'image_colors',
                  'image_brightness_factor',
                  'transparency_depth',
                  'default_rgba',
                  )
    loc = locals()
    kw.update({attr:loc[attr] for attr in parameters if loc[attr] is not None})

    def rgb_to_rgba(color):
      if len(color) == 3:
        return tuple(color) + (1,)
      return color

    for attr in ('solid_levels', 'solid_colors', 'solid_brightness_factor'):
      if attr in kw:
        kw['image_' % attr[6:]] = kw[attr]	# Rename old names for these attributes.

    if 'surface_colors' in kw:
      kw['surface_colors'] = [rgb_to_rgba(c) for c in kw['surface_colors']]
    if 'image_colors' in kw:
      kw['image_colors'] = [rgb_to_rgba(c) for c in kw['image_colors']]

    if ('surface_levels' in kw and
        not 'surface_colors' in kw and
        len(kw['surface_levels']) != len(self.surfaces)):
      kw['surface_colors'] = [self.default_rgba] * len(kw['surface_levels'])
    if ('image_levels' in kw and
        not 'image_colors' in kw and
        len(kw['image_levels']) != len(self.image_colors)):
      rgba = saturate_rgba(self.default_rgba)
      kw['image_colors'] = [rgba] * len(kw['image_levels'])

    if 'default_rgba' in kw:
      self.default_rgba = kw['default_rgba'] = rgb_to_rgba(kw['default_rgba'])

    # Make copies of lists.
    for param in ('surface_levels', 'surface_colors',
                  'image_levels', 'image_colors'):
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
        elif param == 'transparency':
          for s in self.surfaces:
            s.set_transparency((1-kw['transparency'])*255)
        elif param == 'brightness':
          for s in self.surfaces:
            s.set_brightness(kw['brightness'])
        else:
          setattr(self, param, values)

    # Update rendering options.
    option_changed = False
    ro = self.rendering_options
    image_mode_changed = ('image_mode' in kw and kw['image_mode'] != ro.image_mode)
    adjust_step = (self.image_shown and image_mode_changed)
    for k,v in kw.items():
      # TODO: Only allow setting allowed RenderOptions attributes
      if k in ro.__dict__ or hasattr(ro, k):
        setattr(ro, k, v)
        option_changed = True
    if adjust_step:
      r = self.region
      self.new_region(r[0], r[1], r[2], adjust_step = True)

    if 'surface_levels' in kw or 'image_levels' in kw:
      self.call_change_callbacks('thresholds changed')

    if ('surface_colors' in kw or 'transparency' in kw or
        'image_colors' in kw or 'image_brightness_factor' in kw or
        'transparency_depth' in kw):
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
    self.image_colors = [rgba]*len(self.image_levels)
    self._drawings_need_update()
    self.call_change_callbacks('colors changed')

  # ---------------------------------------------------------------------------
  #
  def _get_model_color(self):
    from chimerax.core.colors import rgba_to_rgba8
    if self.surface_shown:
      surfs = self.surfaces
      if surfs:
        return min(surfs, key = lambda s: s.level).color
    elif self.image_shown:
      lev = self.image_levels
      if lev:
        from numpy import argmin
        i = argmin([v for v,b in lev])
        return rgba_to_rgba8(self.image_colors[i])
    drgba = self.data.rgba
    if drgba:
      return rgba_to_rgba8(drgba)
    return None
  def _set_model_color(self, color):
    from chimerax.core.colors import rgba8_to_rgba
    self.set_color(rgba8_to_rgba(color))
  model_color = property(_get_model_color, _set_model_color)

  # ---------------------------------------------------------------------------
  #
  def _get_mask_colors(self):
    return self._mask_colors
  def _set_mask_colors(self, mask_colors):
    self._mask_colors = mask_colors
    if self._image:
      self._image.mask_colors = mask_colors
  mask_colors = property(_get_mask_colors, _set_mask_colors)

  # ---------------------------------------------------------------------------
  #
  def _get_segment_colors(self):
    return self._segment_colors
  def _set_segment_colors(self, segment_colors):
    self._segment_colors = segment_colors
    if self._image:
      self._image.segment_colors = segment_colors
  segment_colors = property(_get_segment_colors, _set_segment_colors)

  # ---------------------------------------------------------------------------
  #
  def set_transparency(self, alpha):
    '''Alpha values in range 0-255. Only changes current style (surface/mesh or image).'''
    if self.surface_shown:
      for s in self.surfaces:
        s.set_transparency(alpha)
    if self.image_shown:
      a1 = alpha/255
      self.set_parameters(image_colors = [(r,g,b,a1) for r,g,b,a in self.image_colors])
    self._drawings_need_update()

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
    fpa = faces_per_axis(self.image_shown, ro.image_mode)
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
  def is_full_region(self, region = None, any_step = False):

    if region is None:
      region = self.region
    elif region == 'all':
      return True
    ijk_min, ijk_max,ijk_step = region
    dmax = tuple([s-1 for s in self.data.size])
    full = (tuple(ijk_min) == (0,0,0) and
            tuple(ijk_max) == dmax and
            (any_step or tuple(ijk_step) == (1,1,1)))
    return full

  # ---------------------------------------------------------------------------
  # Either data values or subregion has changed.
  #
  def matrix_changed(self):

    self.matrix_stats = None
    self._matrix_id += 1
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

    return len(self.surfaces) > 0 and len(self.image_levels) > 0

  # ---------------------------------------------------------------------------
  #
  def initial_surface_levels(self, mstats = None, vfrac = (0.01, 0.90), mfrac = None):
    d = self.data
    rgba = self.default_rgba
    if hasattr(d, 'initial_surface_level'):
      levels = [d.initial_surface_level]
      colors = [rgba]
      return levels, colors

    if mstats is None:
      mstats = self.matrix_value_statistics()
    if mfrac is None:
      v = mstats.rank_data_value(1-vfrac[0], estimate = 'high')
    else:
      v = mstats.mass_rank_data_value(1-mfrac[0], estimate = 'high')

    binary = getattr(self.data, 'binary', False)
    polar = getattr(self.data, 'polar_values', False)
    if polar:
      levels = [-v,v]
      neg_rgba = _negative_color(rgba)
      colors = [neg_rgba,rgba]
    elif binary:
      levels = [0.5]
      colors = [rgba]
    else:
      levels = [v]
      colors = [rgba]
    return levels, colors

  # ---------------------------------------------------------------------------
  #
  def initial_image_levels(self, mstats = None, vfrac = (0.01, 0.90), mfrac = None):
    rgba = saturate_rgba(self.default_rgba)
    d = self.data
    if hasattr(d, 'initial_image_thresholds'):
      levels = d.initial_image_thresholds
      colors = [rgba]*len(levels)
      return levels, colors
    if mstats is None:
      mstats = self.matrix_value_statistics()
    ilow, imid, imax = 0, 0.8, 1
    if mfrac is None:
      vlow = mstats.rank_data_value(1-vfrac[1])
      vmid = mstats.rank_data_value(1-vfrac[0])
    else:
      vlow = mstats.mass_rank_data_value(1-mfrac[1])
      vmid = mstats.mass_rank_data_value(1-mfrac[0])
    vmax = mstats.maximum
    binary = getattr(self.data, 'binary', False)
    polar = getattr(self.data, 'polar_values', False)
    if polar:
      levels = ((mstats.minimum,imax), (max(-vmid,mstats.minimum),imid), (0,ilow),
                (0,ilow), (vmid,imid), (vmax,imax))
      neg_rgba = tuple([1-c for c in rgba[:3]] + [rgba[3]])
      colors = (neg_rgba,neg_rgba,neg_rgba, rgba,rgba,rgba)
    elif binary:
      levels = ((0.5,ilow),(1,imax))
      colors = [rgba, rgba]
    elif getattr(self.data, 'initial_thresholds_linear', False):
      levels = ((mstats.minimum,0), (mstats.maximum,1))
      colors = [rgba, rgba]
    else:
      if vlow < vmid and vmid < vmax:
        levels = ((vlow,ilow), (vmid,imid), (vmax,imax))
      else:
        levels = ((vlow,ilow), (0.9*vlow+0.1*vmax,imid), (vmax,imax))
      colors = [rgba]*len(levels)
    return levels, colors

  # ---------------------------------------------------------------------------
  #
  def set_display_style(self, style):
    '''
    Set display style to "surface", "mesh", or "image".
    '''
    self._style_when_shown = None

    if style == 'image' and self.image_shown and not self.surface_shown:
      return
    if (style in ('surface', 'mesh') and self.surface_shown
        and self.surfaces_in_style(style) and not self.image_shown):
      return

# TODO: Seems wrong to adjust step when setting style.
#   Doing this to handle switching between orthoplanes, box faces, and volume styles.
    self.redraw_needed()  # Switch to image does not change surface until draw
    if style == 'image' or self.image_shown:
      ro = self.rendering_options
      adjust_step = (ro.image_mode in ('box faces', 'orthoplanes'))
    else:
      adjust_step = False
    if adjust_step:
      ijk_min, ijk_max = self.region[:2]
      self.new_region(ijk_min, ijk_max)

    # Show or hide surfaces
    surfshow = (style in ('surface', 'mesh'))
    mesh = (style == 'mesh')
    if surfshow and len(self.surfaces) == 0:
      self._style_when_shown = style
    else:
      for s in self.surfaces:
        s.display = surfshow
        s.show_mesh = mesh

    # Show or hide image
    im = self._image
    if im:
      im.display = (style == 'image')
    elif style == 'image':
      self._style_when_shown = 'image'

    self._drawings_need_update()

    self.call_change_callbacks('display style changed')

  # ---------------------------------------------------------------------------
  #
  @property
  def surface_shown(self):
    return len([s for s in self.surfaces if s.display]) >= 1

  # ---------------------------------------------------------------------------
  #
  @property
  def has_mesh(self):
    for s in self.surfaces:
      if s.show_mesh:
        return True
    return self._style_when_shown == 'mesh'

  # ---------------------------------------------------------------------------
  #
  @property
  def image_shown(self):
    im = self._image
    return (im and im.display)

  # ---------------------------------------------------------------------------
  #
  @property
  def image_will_show(self):
    return self._style_when_shown == 'image'

  # ---------------------------------------------------------------------------
  #
  def surfaces_in_style(self, style):
    for s in self.surfaces:
      if s.style != style:
          return False
    return True

  # ---------------------------------------------------------------------------
  #
  def _set_display(self, display):
    if display == self.display:
      return
    Model.display.fset(self, display)
    if display:
      self._drawings_need_update()	# Create model geometry if needed.
    self.call_change_callbacks('displayed')
    if not display:
      self._keep_displayed_data = None	# Allow data to be freed from cache.
  display = Model.display.setter(_set_display)

  # ---------------------------------------------------------------------------
  #
  def show(self, style = None, rendering_options = None, show = True):
    '''
    Deprecated: Use v.display = True.
    Display the volume using the current parameters.
    '''
    if style is not None:
      self.set_display_style(style)

    if rendering_options:
      self.rendering_options = rendering_options

    # Prevent cached matrix for displayed data from being freed.
    self._keep_displayed_data = self.displayed_matrices() if show else None

    # Update surface or image rendering
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
    if s.in_script and getattr(self, '_initial_style_set', False):
      # In scripts update volume drawings immediately.
      # Script commands often depend on volume surfaces being computed immediately.
      # For examnple, set surface level, then run volume dust.
      self.update_drawings()

  # ---------------------------------------------------------------------------
  #
  def update_drawings(self):

    if self._style_when_shown in ('surface', 'mesh') and len(self.surfaces) == 0:
      # No surfaces, so create one at default level.
      levels, colors = self.initial_surface_levels()
      for lev, c in zip(levels, colors):
        s = self.add_surface(lev, rgba = c)
        s.show_mesh = (self._style_when_shown == 'mesh')
    if self.surface_shown:
      self._update_surfaces()

    if self.image_shown:
      self._update_image()
    elif self._style_when_shown == 'image':
      if len(self.image_levels) == 0:
        self.image_levels, self.image_colors = self.initial_image_levels()
      self._update_image()	# Create image

    if self._style_when_shown is not None:
      self._style_when_shown = None
      self.call_change_callbacks('display style changed')

    # Prevent cached matrix for displayed data from being freed.
    self._keep_displayed_data = self.displayed_matrices()

  # ---------------------------------------------------------------------------
  #
  def _update_surfaces(self):

    ro = self.rendering_options
    try:
      for s in self.surfaces:
        s.update_surface(ro)
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
    from chimerax.map_data import surface_level_enclosing_volume
    try:
      level = surface_level_enclosing_volume(matrix, gvolume, tolerance, max_bisections)
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
    self.outline_box.show(show, rgb, linewidth)

  # ---------------------------------------------------------------------------
  #
  def _update_image(self):

    im = self._image
    if im is None or im.deleted:
      self._image = im = VolumeImage(self)
    else:
      im.update_settings()

    ro = self.rendering_options
    self.show_outline_box(ro.show_outline_box, ro.outline_box_rgb,
                          ro.outline_box_linewidth)
    return im

  # ---------------------------------------------------------------------------
  #
  def shown(self):

    surf_disp = len([s for s in self.surfaces if s.display]) > 0
    image_disp = (self._image and self._image.display)
    return self.display and (surf_disp or image_disp or self._style_when_shown is not None)

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
  def showing_image(self, image_mode = None):

    same_mode = (image_mode is None or self.rendering_options.image_mode == image_mode)
    return (self.image_shown or self._style_when_shown == 'image') and same_mode

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
      vc = [v for v in c.maps if v is not self and not v.deleted]
    return vc

  # ---------------------------------------------------------------------------
  #
  def copy(self):

    v = volume_from_grid_data(self.data, self.session, style = None, show_dialog = False)
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

    if copy_thresholds:
      # Copy thresholds
      self.set_parameters(
        surface_levels = [s.level for s in v.surfaces],
        image_levels = v.image_levels,
        )

    if copy_colors:
      # Copy colors

      color_kw = {}
      if len(self.surfaces) == len(v.surfaces):
        color_kw['surface_colors'] = [s.rgba for s in v.surfaces]
      if len(self.image_colors) == len(v.image_colors):
        color_kw['image_colors'] = v.image_colors

      self.set_parameters(
        transparency_depth = v.transparency_depth,
        image_brightness_factor = v.image_brightness_factor,
        default_rgba = v.default_rgba,
        **color_kw
        )

    if copy_rendering_options:
      # Copy rendering options
      self.set_parameters(**v.rendering_options.__dict__)

    if copy_region:
      # Copy region bounds
      ijk_min, ijk_max, ijk_step = v.region
      self.new_region(ijk_min, ijk_max, ijk_step)

    if copy_style:
      # Copy display style
      if v.surface_shown:
        self.set_display_style('mesh' if v.has_mesh else 'surface')
      if v.image_shown:
        self.set_display_style('image')
      # TODO: This doesn't handle case when both surface and image styles shown.

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
                    unshow_original = True, model_id = None, open_model = True,
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

    v = volume_from_grid_data(g, self.session, style = None, model_id = model_id,
                              show_dialog = False, open_model = open_model)
    v.copy_settings_from(self, copy_region = False, copy_colors = copy_colors)

    if unshow_original:
      self.display = False

    return v

  # ---------------------------------------------------------------------------
  #
  def region_grid(self, r, value_type = None, new_spacing = None, clamp = True):

    shape = self.matrix_size(region = r, clamp = clamp)
    if new_spacing is not None:
      d = self.data
      shape = [int(sz*(s*st/ns)) for s,ns,sz,st in zip(d.step, new_spacing, d.size, r[2])]
    shape.reverse()
    d = self.data
    if value_type is None:
      value_type = d.value_type
    from numpy import zeros
    m = zeros(shape, value_type)
    origin, step = self.region_origin_and_step(r)
    if new_spacing is not None:
      step = new_spacing
    from chimerax.map_data import ArrayGridData
    g = ArrayGridData(m, origin, step, d.cell_angles, d.rotation)
    g.rgba = d.rgba           # Copy default data color.
    return g

  # ---------------------------------------------------------------------------
  #
  def surface_bounds(self):
    '''Surface bounds in volume coordinate system.'''
    from chimerax.geometry import union_bounds
    return union_bounds([s.geometry_bounds() for s in self.surfaces])

  # ---------------------------------------------------------------------------
  # The xyz bounding box encloses the subsampled grid with half a step size
  # padding on all sides.
  #
  def xyz_bounds(self, step = None, subregion = None):

    ijk_min_edge, ijk_max_edge = self.ijk_bounds(step, subregion)

    from chimerax.map_data import box_corners, bounding_box
    ijk_corners = box_corners(ijk_min_edge, ijk_max_edge)
    data = self.data
    xyz_min, xyz_max = bounding_box([data.ijk_to_xyz(c) for c in ijk_corners])

    return (xyz_min, xyz_max)

  # ---------------------------------------------------------------------------
  #
  def center(self, step = None, subregion = None):

    ijk_min, ijk_max = self.ijk_bounds(step, subregion)
    ijk_mid = [0.5*(i0+i1) for i0,i1 in zip(ijk_min, ijk_max)]
    c = self.data.ijk_to_xyz(ijk_mid)
    return c

  # ---------------------------------------------------------------------------
  #
  def corners(self, step = None, subregion = None):
    from chimerax.map_data import box_corners
    ijk_corners = box_corners(*self.ijk_bounds())
    corners = self.data.ijk_to_xyz_transform * ijk_corners
    return corners

  # ---------------------------------------------------------------------------
  #
  def first_intercept(self, mxyz1, mxyz2, exclude = None):

    if exclude is not None and exclude(self):
      return None

    if self.image_shown:
      vxyz1, vxyz2 = self.position.inverse() * (mxyz1, mxyz2)
      from . import slice
      xyz_in, xyz_out = slice.box_line_intercepts((vxyz1, vxyz2), self.xyz_bounds())
      if xyz_in is None or xyz_out is None:
        return None
      from chimerax.geometry import norm
      f = norm(0.5*(xyz_in+xyz_out) - mxyz1) / norm(mxyz2 - mxyz1)
      ro = self.rendering_options
      if ro.image_mode == 'full region' and self.single_plane():
        # Report voxel under mouse and data value.
        ijk = tuple(int(round(i)) for i in self.data.xyz_to_ijk(0.5*(xyz_in + xyz_out)))
        detail = 'voxel %d,%d,%d' % ijk
        ijk_step = self.region[2]
        v = self.region_matrix((ijk,ijk,ijk_step))
        if v.size == 1:
          detail += ' value %.4g' % v[0,0,0]
      else:
        detail = ''
      return PickedMap(self, f, detail)
    elif self.surface_shown:
      from chimerax.graphics import Drawing
      pd = Drawing.first_intercept(self, mxyz1, mxyz2, exclude)
      if pd:
        d = pd.drawing()
        detail = d.name
        p = PickedMap(self, pd.distance, detail)
        p.triangle_pick = pd.picked_triangle if hasattr(pd, 'picked_triangle') else pd
        if d.display_style == d.Mesh or hasattr(pd, 'is_transparent') and pd.is_transparent():
          # Try picking opaque object under transparent map
          p.pick_through = True
        return p

    return None

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
    from chimerax.map_data import points_ijk_bounds
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
    v = self.scene_position.transform_vector(lv)
    from chimerax.geometry import normalize_vector
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
    from chimerax.map_data import ProgressReporter
    progress = ProgressReporter(operation, size, d.value_type.itemsize,
                                log = self.session.logger)
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
  # is specified as an axis and a matrix index.  This is used for image
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
    from chimerax.geometry import Place
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
  # Surface / image rendering is not automatically redrawn when data values
  # change.
  #
  def data_changed_cb(self, type):

    if type == 'values changed':
      self.data.clear_cache()
      self.matrix_changed()
      if self._image:
        self._image.map_values_changed()
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
  # Return the origin aligned to a multiple of step, and size and step of the region.
  # The origin[axis] is the smallest index equal or greater to region ijk_min[axis]
  # that is a multiple of the step.  The end of the region is the largest index equal
  # or less than ijk_max[axis] that is a multiple of the step, unless that index is
  # less than the origin in which case the end equals the origin.  The returned
  # size is always a multiple of step unless clamp is true and size would extend
  # beyond grid size.
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

    end = [max(s*(i//s),o) for i,s,o in zip(ijk_max, ijk_step, origin)]
    size = [e-o+s for e,o,s in zip(end, origin, ijk_step)]

    if clamp:
      origin = [max(i,0) for i in origin]
      size = [(s if o+s <= lim else max(0,lim-o))
              for o,s,lim in zip(origin, size, self.data.size)]

    return tuple(origin), tuple(size), tuple(ijk_step)

  # ---------------------------------------------------------------------------
  # Applying point_xform to points gives Chimera world coordinates.  If the
  # point_xform is None then the points are in local volume coordinates.
  # The returned values are float32.  The returned outside array contains
  # integer index value for points outside the volume.
  #
  def interpolated_values(self, points, point_xform = None,
                          out_of_bounds_list = False, subregion = 'all',
                          step = (1,1,1), method = 'linear'):

    matrix, p2m_transform = self.matrix_and_transform(point_xform,
                                                      subregion, step)
    from chimerax.map_data import interpolate_volume_data
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

    from chimerax.map_data import interpolate_volume_gradient
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
      if subregion == 'all' and self.data.voxel_count() > 2**28:
        # Load just the small subregion of self that covers the grid in case
        # the vgrid is small and map self is huge (e.g. does not fit in memory).
        subregion = self._covering_subregion(vgrid)
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
  def _covering_subregion(self, v):
    '''Return subregion of self that covers the full region of map v.'''
    vijk_min = (0,0,0)
    vijk_max = tuple(s-1 for s in v.data.size)
    from chimerax.map_data import box_corners, bounding_box
    vijk_corners = box_corners(vijk_min, vijk_max)
    vxyz_corners = v.data.ijk_to_xyz(vijk_corners)
    xyz_corners = self.scene_position.inverse() * v.scene_position * vxyz_corners
    ijk_corners = self.data.xyz_to_ijk(xyz_corners)
    ijk_min, ijk_max = bounding_box(ijk_corners)
    from math import floor, ceil
    ijk_min = tuple(max(0,int(floor(i))) for i in ijk_min)
    ijk_max = tuple(min(s-1,int(ceil(i))) for i,s in zip(ijk_max, self.data.size))
    return ijk_min, ijk_max

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
    from chimerax.map_data import grid_indices
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
      from chimerax.map_data import GridSubregion
      sg = GridSubregion(self.data, ijk_min, ijk_max, ijk_step)

    if mask_zone:
#      surf_model = self.surface_model()
#      import SurfaceZone
#      if SurfaceZone.showing_zone(surf_model):
      if False:
        points, radius = SurfaceZone.zone_points_and_distance(surf_model)
        from chimerax.map_data import zone_masked_grid_data
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
    from chimerax.map_data import zone_masked_grid_data
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
    ipv = getattr(self.data, 'ignore_pad_value', None)
    from chimerax.map_data import MatrixValueStatistics
    self.matrix_stats = ms = MatrixValueStatistics(matrices, ignore_pad_value = ipv)
    if nvox >= min_status_message_voxels:
      self.message('')

    return ms

  # ---------------------------------------------------------------------------
  #
  def displayed_matrices(self, read_matrix = True):

    matrices = []
    if self.image_shown:
      ro = self.rendering_options
      if ro.image_mode == 'box faces':
        msize = self.matrix_size()
        for axis in (0,1,2):
          matrices.append(self.matrix_plane(axis, 0, read_matrix))
          matrices.append(self.matrix_plane(axis, msize[axis]-1, read_matrix))
      elif ro.image_mode == 'orthoplanes':
        omijk = self.matrix_index(ro.orthoplane_positions)
        for axis in (0,1,2):
          if ro.orthoplanes_shown[axis]:
            matrices.append(self.matrix_plane(axis, omijk[axis], read_matrix))

    if len(matrices) == 0:
      matrices.append(self.matrix(read_matrix))

    matrices = [m for m in matrices if not m is None]
    return matrices

  # ---------------------------------------------------------------------------
  #
  def write_file(self, path, format = None, options = {}):

    from chimerax.map_data import save_grid_data
    d = self.grid_data()
    format = save_grid_data(d, path, self.session, format, options)

  # ---------------------------------------------------------------------------
  #
  def showing_transparent(self):
    if not self.display:
      return False
    if self.image_shown:
      return 'a' in self._image.color_mode
    from chimerax.graphics import Drawing
    return Drawing.showing_transparent(self)

  # ---------------------------------------------------------------------------
  #
  def models(self):

    mlist = [self]
    return mlist

  # ---------------------------------------------------------------------------
  #
  def image_model(self):

    return self._image

  # ---------------------------------------------------------------------------
  #
  def model_transform(self):

    return self.position

  # ---------------------------------------------------------------------------
  #
  def close_models(self):

    self.close_image()
    self.close_surface()

  # ---------------------------------------------------------------------------
  #
  def close_surface(self):

    self.remove_surfaces()

  # ---------------------------------------------------------------------------
  #
  def close_image(self):

    im = self._image
    if im:
      im.close_model()
      self._image = None

  # ---------------------------------------------------------------------------
  #
  def delete(self):

    d = self.data
    if d:
      d.clear_cache()
      d.remove_change_callback(self.data_changed_cb)
    self.data = None
    self._keep_displayed_data = None
    self.outline_box = None
    self.close_models()
    Model.delete(self)

  # ---------------------------------------------------------------------------
  #
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

  # ---------------------------------------------------------------------------
  #
  @staticmethod
  def restore_snapshot(session, data):
    grid_data = data['grid data state'].grid_data
    if grid_data is None:
      return None	# Map file not available.
    v = Volume(session, grid_data)
    Model.set_state_from_snapshot(v, session, data['model state'])
    v._style_when_shown = None		# Don't show surface style by default.
    from .session import set_map_state
    set_map_state(data['volume state'], v)
    v._drawings_need_update()
    show_volume_dialog(session)
    return v

# -----------------------------------------------------------------------------
#
from .image3d import Image3d
class VolumeImage(Image3d):
  '''
  Model for displaying 3d semi-transparent images.
  These models are children of a :class:`.Volume` model
  and should only be created by Volume.
  '''
  def __init__(self, volume):

    self._volume = v = volume

    ro = v.rendering_options
    from .image3d import blend_manager, Colormap
    cmap = Colormap(self._transfer_function(), v.image_brightness_factor,
                    self._transparency_thickness(),
                    extend_left = ro.colormap_extend_left, extend_right = ro.colormap_extend_right)

    Image3d.__init__(self, 'image', v.data, v.region, cmap, v.rendering_options,
                     v.session, blend_manager(v.session))
    v.add([self])

    if v.mask_colors is not None:
      self.mask_colors = v.mask_colors
    if v.segment_colors is not None:
      self.segment_colors = v.segment_colors

  # ---------------------------------------------------------------------------
  #
  def delete(self):
    self._volume._image = None
    Image3d.delete(self)

  # ---------------------------------------------------------------------------
  #
  def update_settings(self):
    v = self._volume
    ro = v.rendering_options
    self.set_options(ro)
    self.set_region(v.region)
    self._update_colormap()

  # ---------------------------------------------------------------------------
  #
  def _update_colormap(self):
    v = self._volume
    ro = v.rendering_options
    from .image3d import Colormap
    cmap = Colormap(self._transfer_function(), v.image_brightness_factor, self._transparency_thickness(),
                    extend_left = ro.colormap_extend_left, extend_right = ro.colormap_extend_right)
    self.set_colormap(cmap)

  # ---------------------------------------------------------------------------
  #
  def _get_model_color(self):
    '''Return average color.'''
    v = self._volume
    colors = v.image_colors
    from numpy import array, mean, uint8
    if len(colors) == 0:
      c = array((255,255,255,255), uint8)
    else:
      c = array([int(r*255) for r in mean(colors, axis=0)], uint8)
    return c
  def _set_model_color(self, color):
    v = self._volume
    rgba = [[r/255 for r in color]] * len(v.image_levels)
    if rgba != v.image_colors:
      v.image_colors = rgba
      self._update_colormap()
      v.call_change_callbacks('colors changed')
  model_color = property(_get_model_color, _set_model_color)

  # ---------------------------------------------------------------------------
  #
  def _transparency_thickness(self):
    v = self._volume
    ro = v.rendering_options
    if ro.image_mode == 'tilted slab':
      thickness = ro.tilted_slab_spacing * ro.tilted_slab_plane_count
    else:
      ijk_min, ijk_max = v.ijk_bounds()
      box_size = [(i1-i0)*s for i0,i1,s in zip(ijk_min, ijk_max, v.data.step)]
      thickness = min(box_size)
    return v.transparency_depth * thickness

  # ---------------------------------------------------------------------------
  # Without brightness and transparency adjustment.
  #
  def _transfer_function(self):
    v = self._volume
    tf = [tuple(ts) + tuple(c) for ts,c in zip(v.image_levels, v.image_colors)]
    tf.sort()
    return tf


  # ---------------------------------------------------------------------------
  # State save/restore in ChimeraX
  #
  def take_snapshot(self, session, flags):
    data = {
      'volume': self._volume,
      'model state': Surface.take_snapshot(self, session, flags),
      'version': 1
    }
    return data

  # ---------------------------------------------------------------------------
  #
  @staticmethod
  def restore_snapshot(session, data):
    v = data['volume']
    if v is None:
      return None	# Volume was not restored, e.g. file missing.
    im = VolumeImage(v)
    Model.set_state_from_snapshot(im, session, data['model state'])
    v._image = im
    return im

# -----------------------------------------------------------------------------
#
from chimerax.core.models import Surface
class VolumeSurface(Surface):
  '''
  Model for displaying a contour surface of a :class:`.Volume` model.
  These models are children of a Volume and should only be created
  by the :func:`.Volume.add_surface` method.
  '''

  def __init__(self, volume, level, rgba = (1,1,1,1), mesh = False):
    name = 'surface'
    Surface.__init__(self, name, volume.session)
    self.volume = volume
    self._level = level
    self._mesh = mesh
    color = [int(min(255,max(0,255*r))) for r in rgba]
    Surface.set_color(self, color)	# Don't set self.rgba since that calls color changed volume callback
    self._contour_settings = {}	         	# Settings for current surface geometry
    self._min_status_message_voxels = 2**24	# Show status messages only on big surface calculations
    self._use_thread = False			# Whether to compute next surface in thread
    self._surf_calc_thread = None
    self.clip_cap = True			# Cap surface when clipped

  def delete(self):
    try:
      self.volume._surfaces.remove(self)
    except ValueError:
      pass	# This VolumeSurface was already removed from Volume
    Surface.delete(self)

  def _get_level(self):
    return self._level
  def set_level(self, level, use_thread = False):
    self._level = level
    self._use_thread = use_thread
    self.volume.redraw_needed(shape_changed = True)
  level = property(_get_level, set_level)
  '''Threshold level for the surface. Settable.'''

  def _get_rgba(self):
    return tuple(c/255 for c in self.color)
  def _set_rgba(self, rgba):
    self.set_color([int(min(255,max(0,255*r))) for r in rgba])
  rgba = property(_get_rgba, _set_rgba)
  '''Float red,green,blue,alpha values in range 0-1'''

  def set_transparency(self, alpha):
    '''Set surface transparency, 0-255 range.'''
    if self.vertex_colors is None:
      self.rgba = tuple(self.rgba[:3]) + (alpha/255,)
    else:
      # Change per-vertex transparency leaving colors the same.
      Surface.set_transparency(self, alpha)

  def set_brightness(self, brightness):
    '''
    Scale color so maximum of red, green, blue components equals 255*brightness.
    '''
    r,g,b,a = self.color
    cb = max(r,g,b)
    b255 = max(0,min(255,brightness*255))
    if cb == 0:
      self.color = (b255, b255, b255, a)
    else:
      f = b255/cb
      self.color = (int(f*r), int(f*g), int(f*b), a)

  def get_color(self):
    return Surface.get_color(self)
  def set_color(self, color):
    if (color != self.color).any():
      Surface.set_color(self, color)
      self._set_clip_cap_color(color)
      self.volume.call_change_callbacks('colors changed')
  color = property(get_color, set_color)

  def _set_clip_cap_color(self, color):
    for c in self.child_models():
      if getattr(c, 'is_clip_cap', False):
        c.set_color(color)

  def _get_colors(self):
    return Surface.get_colors(self)
  def _set_colors(self, colors):
    Surface.set_colors(self, colors)
    self.volume.call_change_callbacks('colors changed')
  colors = property(_get_colors, _set_colors)

  def _get_show_mesh(self):
    return self._mesh
  def _set_show_mesh(self, show_mesh):
    if show_mesh != self._mesh:
      self._mesh = show_mesh
      self.display_style = self.Mesh if show_mesh else self.Solid
      self.volume._drawings_need_update()
  show_mesh = property(_get_show_mesh, _set_show_mesh)

  @property
  def style(self):
    return 'mesh' if self.show_mesh else 'surface'

  # ---------------------------------------------------------------------------
  #
  def update_surface(self, rendering_options):

    self._use_thread_result()

    if not self._geometry_changed(rendering_options):
      self._set_appearance(rendering_options, clear_vertex_colors = False)
      return

    v = self.volume
    matrix = v.matrix()
    level = self.level

    show_status = (matrix.size >= self._min_status_message_voxels)
    if show_status:
      v.message('Computing %s surface, level %.3g' % (v.data.name, level))

    if self._use_thread:
      self._use_thread = False
      self._calc_surface_in_thread(matrix, level, rendering_options)
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
    self._set_appearance(rendering_options)

  # ---------------------------------------------------------------------------
  #
  def _calc_surface_in_thread(self, matrix, level, rendering_options):
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

    sct.in_queue.put((matrix, level, rendering_options))

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

    va, na, ta, hidden_edges, matrix, level, rendering_options = result
    self._set_surface(va, na, ta, hidden_edges)
    self._set_appearance(rendering_options)

    show_status = (matrix.size >= self._min_status_message_voxels)
    if show_status:
      v = self.volume
      v.message('Calculated %s surface, level %.3g, with %d triangles'
                   % (v.data.name, level, len(ta)), blank_after = 3.0)

  # ---------------------------------------------------------------------------
  #
  def _calculate_contour_surface_threaded(self, matrix, level, rendering_options):
    va, na, ta, hidden_edges = self._calculate_contour_surface(matrix,level, rendering_options)
    return va, na, ta, hidden_edges, matrix, level, rendering_options

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

    return varray, narray, tarray, hidden_edges

  # ---------------------------------------------------------------------------
  #
  def _set_surface(self, va, na, ta, hidden_edges):

    self.set_geometry(va, na, ta, edge_mask = hidden_edges)

    # TODO: Clip cap offset for different contour levels is not related to voxel size.
    v = self.volume
    self.clip_offset = .002* len([s for s in v.surfaces if self.level < s.level])

  # ---------------------------------------------------------------------------
  #
  def _set_appearance(self, rendering_options, clear_vertex_colors = True):

    # Update color
    if self.auto_recolor_vertices is None and clear_vertex_colors:
      self.vertex_colors = None

    # Update display style
    ro = rendering_options
    # OpenGL draws nothing for degenerate triangles where two vertices are
    # identical.  For 2d contours want to see these triangles so show as mesh.
    contour_2d = self.volume.single_plane() and not ro.cap_faces
    style = self.Mesh if self.show_mesh or contour_2d else self.Solid
    self.display_style = style

    # Update lighting
    if contour_2d:  lit = False
    elif self.show_mesh: lit = ro.mesh_lighting
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
                        'matrix_id': v._matrix_id,
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
      'show_mesh': self.show_mesh,
      'model state': Surface.take_snapshot(self, session, flags),
      'version': 1
    }
    if self.vertex_colors is not None and self.auto_recolor_vertices is None:
      data['vertex_colors'] = self.vertex_colors
    return data

  @staticmethod
  def restore_snapshot(session, data):
    v = data['volume']
    if v is None:
      return None	# Volume was not restored, e.g. file missing.
    s = VolumeSurface(v, data['level'], data['rgba'], data.get('show_mesh', False))
    Model.set_state_from_snapshot(s, session, data['model state'])
    if v._style_when_shown == 'image':
      s.display = False		# Old sessions had surface shown but not computed when image style used.
    v._surfaces.append(s)
    if 'vertex_colors' in data:
      # Compute surface and set vertex colors.
      s.update_surface(v.rendering_options)
      vc = data['vertex_colors']
      if len(s.vertices) == len(vc):
        s.vertex_colors = vc
    return s

# -----------------------------------------------------------------------------
#
def maps_pickable(session, pickable):
  for m in session.models.list(type = Volume):
    m.pickable = pickable
  session._maps_pickable = pickable

# -----------------------------------------------------------------------------
#
from chimerax.graphics import Pick
class PickedMap(Pick):
  '''
  Returned by :func:`.Volume.first_intercept()` when a Volume is picked.
  '''
  def __init__(self, v, distance = None, detail = ''):
    Pick.__init__(self, distance)
    self.map = v
    self.detail = detail
  def description(self):
    return '%s %s %s' % (self.map.id_string, self.map.name, self.detail)
  def specifier(self):
    return '#%s' % self.map.id_string
  def drawing(self):
    return self.map
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
class OutlineBox:

  def __init__(self, volume):

    self._volume = volume
    self._drawing = None		# Child of volume
    self._settings = {}			# Settings from last draw.

  # ---------------------------------------------------------------------------
  # The center and planes option are for orthoplane outlines.
  #
  def show(self, show, rgb, linewidth):
    if not show or rgb is None:
      self.erase_box()
      return

    v = self._volume
    corners = v.corners()
    settings = {'rgb': rgb, 'linewidth': linewidth, 'corners': corners.tolist()}
    ortho = v.showing_image('orthoplanes')
    slab = v.showing_image('tilted slab')
    ro = v.rendering_options
    if ortho:
      settings['planes'] = ro.orthoplanes_shown
      settings['center'] = tuple(v.data.ijk_to_xyz(ro.orthoplane_positions))
      settings['crosshair_width'] = tuple(v.data.step)
    elif slab:
      settings['axis'] = tuple(ro.tilted_slab_axis)
      settings['offset'] = ro.tilted_slab_offset
      settings['spacing'] = ro.tilted_slab_spacing
      settings['plane_count'] = ro.tilted_slab_plane_count

    if settings == self._settings:
      return
    self._settings = settings

    if ortho:
      vertices, triangles, edge_mask = self._orthoplanes_outline()
    elif slab:
      vertices, triangles, edge_mask = self._tilted_slab_outline()
    else:
      vertices, triangles, edge_mask = self._box_outline()

    self._make_box(rgb, linewidth, vertices, triangles, edge_mask)

  # ---------------------------------------------------------------------------
  #
  def _box_outline(self):

    vlist = self._settings['corners']
    tlist = ((0,4,5), (5,1,0), (0,2,6), (6,4,0),
             (0,1,3), (3,2,0), (7,3,1), (1,5,7),
             (7,6,2), (2,3,7), (7,5,4), (4,6,7))
    b = 8 + 2 + 1    # Bit mask, 8 = show triangle, edges are bits 4,2,1
    edge_mask = [b]*len(tlist)		# hide box face diagonals
    return vlist, tlist, edge_mask

  # ---------------------------------------------------------------------------
  #
  def _orthoplanes_outline(self):

    s = self._settings
    corners, center, planes, crosshair_width = s['corners'], s['center'], s['planes'], s['crosshair_width']
    vlist, tlist = [], []
    self._plane_outlines(corners, center, planes, vlist, tlist)
    self._crosshairs(corners, center, planes, crosshair_width, vlist, tlist)
    b = 8 + 2 + 1    # Bit mask, 8 = show triangle, edges are bits 4,2,1
    edge_mask = [b]*len(tlist)		# hide box face diagonals
    return vlist, tlist, edge_mask

  # ---------------------------------------------------------------------------
  #
  def _tilted_slab_outline(self):

    # Slab box
    s = self._settings
    offset, spacing, plane_count = s['offset'], s['spacing'], s['plane_count']
    start = offset - 0.5*spacing
    thickness = spacing * plane_count
    from . import box_cuts
    sva, sta = box_cuts(s['corners'], s['axis'], start, thickness, num_cuts = 2)
    from chimerax.surface import boundary_edge_mask
    se = boundary_edge_mask(sta)

    # Show region box
    bva, bta, be = self._box_outline()

    # Combine box and slab arrays.
    from chimerax.surface import combine_geometry_vte
    va, ta, edge_mask = combine_geometry_vte(((bva,bta,be), (sva,sta,se)))

    return va, ta, edge_mask

  # ---------------------------------------------------------------------------
  #
  def _make_box(self, rgb, linewidth, vertices, triangles, edge_mask):

    self.erase_box()

    self._drawing = d = self._volume.new_drawing('outline box')
    d.display_style = d.Mesh
#    d.linewidth = linewidth
    d.use_lighting = False
    d.casts_shadows = False
    d.pickable = False
    # Set geometry after setting outline_box attribute to avoid undesired
    # coloring and capping of outline boxes.
    from numpy import array
    va, ta = array(vertices), array(triangles)
    d.set_geometry(va, None, ta)
    d.edge_mask = edge_mask
    rgba = tuple(rgb) + (1,)
    d.color = tuple(int(255*r) for r in rgba)

  # ---------------------------------------------------------------------------
  #
  def _plane_outlines(self, corners, center, planes, vlist, tlist):

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
  def _crosshairs(self, corners, center, planes, width, vlist, tlist):

    hw0,hw1,hw2 = [0.5*w for w in width]
    btlist = ((0,4,5), (5,1,0), (0,2,6), (6,4,0),
              (0,1,3), (3,2,0), (7,3,1), (1,5,7),
              (7,6,2), (2,3,7), (7,5,4), (4,6,7))
    from chimerax.map_data import box_corners
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

    d = self._drawing
    if d is None:
      return

    if not d.was_deleted:
      self._volume.remove_drawing(d)

    self._drawing = None
    self._settings = {}

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
class RenderingOptions:
  '''
  Rendering options for a :class:`.Volume` that specify details of how
  surface and image style depictions appear.  Some options are not implemented
  but existed in Chimera and may be implemented in the future.

  Attributes
  ----------
  show_outline_box : False
    Whether a outline box is shown for the displayed subregion.
  outline_box_rgb : (1,1,1)
    Outline box color (red, green, blue) components, range 0-1.
  limit_voxel_count : True
    Whether to auto-adjust step size so at most voxel_limit voxels are shown.
  voxel_limit : 16
    Choose step size so the region has at most this many Mvoxels.
  color_mode : 'auto8'
    Sets the pixel format for image style rendering color vs grayscale,
    transparent vs opaque, and bits per color component.
    (auto|opaque|rgba|rgb|la|l)(4|8|12|16).
  color_modes : ('auto4', 'auto8', 'auto12', 'auto16', 'opaque4', 'opaque8', 'opaque12', 'opaque16', 'rgba4', 'rgba8', 'rgba12', 'rgba16', 'rgb4', 'rgb8', 'rgb12', 'rgb16', 'la4', 'la8', 'la12', 'la16', 'l4', 'l8', 'l12', 'l16')
    The allowed color modes for image style rendering.  Read only.
  colormap_on_gpu : False
    Whether colors are computed from map values on the gpu for image style rendering.
  colormap_size : 2048
    If colormap_on_gpu is true, what is the size of the colormap for map values
    that are not  8 or 16-bit data types.
  colormap_extend_left : False
    Whether the image coloring applies to map values less than the minimum Volume image_level.
  colormap_extend_right : True
    Whether the image coloring applies to map values greater than the maximum Volume image_level.
  blend_on_gpu : False
    Whether image rendering blends images on gpu instead of cpu.
  projection_mode : 'auto'
    Determines what slices are used for image rendering.
  projection_modes : ('auto', '2d-xyz', '2d-x', '2d-y', '2d-z', '3d')
    Allowed projection modes.  Read only.
  plane_spacing : 'min'
    Spacing of slices for image style rendering. Values "min", "max", "mean" use
    the grid spacing, or specific distance value can be given.
  full_region_on_gpu : False
    For image rendering is the entire map kept on the GPU for fast cropping.
  bt_correction : False
    Image rendering axis-dependent brightness and transparency correction.  Not implemented.
  minimal_texture_memory : False
    Whether to reuse a single texture for image rendering.  Not implemented.
  maximum_intensity_projection : False
    Whether to use maximum intensity projection image rendering.  If False then
    transparent blending is used.
  linear_interpolation : True
    Whether image rendering linearly interpolates pixel colors.
  dim_transparency : True
    Whether transparent surface rendering multiplies colors
    by opacity making more transparent voxels dimmer.
    True uses (alpha, 1-alpha) blending while False uses (1, 1-alpha) blending.
  dim_transparent_voxels : True
    Whether transparent image rendering multiplies colors
    by opacity making more transparent voxels dimmer.
    True uses (alpha, 1-alpha) blending while False uses (1, 1-alpha) blending.
  line_thickness : 1
    The thickness of lines in pixels for mesh display.  Not implemented because
    OpenGL core profile does not support line thickness.
  smooth_lines : False
    Whether mesh lines are rendered with anti-aliasing giving a smoother appearance.
  mesh_lighting : True
    Whether mesh rendering uses directional lighting.
  two_sided_lighting : True
    Whether the interior of surfaces and meshes have directional lighting.
    Not implemented, always uses two-sided.
  flip_normals : False
    Whether negative map values have surface normals flipped.  Not implemented.
    This only has an effect when two sided lighting is false, and that mode is not implemented.
  subdivide_surface : False
    Whether to split every triangle into 4 smaller triangles for surfaces and meshes.
  subdivision_levels : 1
    How many levels of triangle splitting to apply if subdivide surface is True.
    A value of 1 divides triangles into 4 smaller triangles, 2 divides into 16
    smaller triangles, N divides into 4^N smaller triangles.
  surface_smoothing : False
    Whether to move surface or mesh vertices to give smoother surface appearance.
  smoothing_iterations : 2
    How many iterations of smoothing to apply if surface smoothing is enabled.
  smoothing_factor : .3
    When surface smoothing each vertex is moved a fraction of the ways towards
    the average position of the connected vertices.  This parameter is the fraction.
  square_mesh : True
    Whether mesh display hides diagonal mesh lines.  If true than only mesh lines
    intersecting the xy, yz, and xz grid planes are shown.
  cap_faces : True
    Whether surface and mesh display covers the holes on the faces of the
    volume box where the surface reaches the box boundaries.
  orthoplanes_shown : (False, False, False)
    For image style display, show 0 to 3 orthogonal planes perpendicular to x,y,z axes.
    If any of the 3 values is True then orthoplane mode is enabled.
  orthoplane_positions : (0,0,0)
    The center voxel i,j,k grid index for orthoplane image rendering.
  tilted_slab_axis : (0,0,1)
    If image_mode is "tilted slab" then this is the axis perpendicular
    to the displayed slab in volume xyz coordinates.
  tilted_slab_offset : 0
    Offset of the front face of the slab.  The front face plane is defined
    by dot((x,y,z), tilted_slab_axis) = tilted_slab_offset
  tilted_slab_spacing : 1
    Spacing of planes shown in tilted slab mode in physical units.
  tilted_slab_plane_count : 1
    Number of planes shown in tilted slab mode.
  image_mode : 'full region'
    The mode for image style rendering.  Can be 'full region', 'orthoplanes',
    'box faces', or 'tilted slab'.
  backing_color : None
    Color drawn behind transparent image rendering.  This blocks the view
    of objects and the background behind the volume and can give better
    contrast (e.g. black backing when white background color in use).
  '''
  def __init__(self):

    self.show_outline_box = False
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
    self.color_mode = 'auto8'         # image rendering pixel formats
                                      #  (auto|opaque|rgba|rgb|la|l)(4|8|12|16)
    self.colormap_on_gpu = False      # image rendering with colors computed on gpu
    self.colormap_size = 2048	      # image rendering on GPU or other than 8 or 16-bit data types
    self.colormap_extend_left = False
    self.colormap_extend_right = True
    self.blend_on_gpu = False	      # image rendering blend images on gpu instead of cpu
    self.projection_modes = ('auto', '2d-xyz', '2d-x', '2d-y', '2d-z', '3d')
    self.projection_mode = 'auto'           # auto, 2d-xyz, 2d-x, 2d-y, 2d-z, 3d
    self.plane_spacing = 'min'		    # "min", "max", "mean" or distance value
    self.full_region_on_gpu = False	    # for image rendering for fast cropping
    self.bt_correction = False              # brightness and transparency
    self.minimal_texture_memory = False
    self.maximum_intensity_projection = False
    self.linear_interpolation = True
    self.dim_transparency = True            # for surfaces
    self.dim_transparent_voxels = True      # for image rendering
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
    self.square_mesh = True
    self.cap_faces = True
    self.orthoplanes_shown = (False, False, False)
    self.orthoplane_positions = (0,0,0) # image rendering
    self.tilted_slab_axis = (0,0,1)	# volume xyz coordinates
    self.tilted_slab_offset = 0
    self.tilted_slab_spacing = 1
    self.tilted_slab_plane_count = 1
    self.image_mode = 'full region'	# 'full region', 'orthoplanes', 'box faces', or 'tilted slab'
    self.backing_color = None		# Color drawn behind transparent images

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

    ro = RenderingOptions()
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

  from chimerax.map_data import clamp_region
  r = clamp_region(region[:2], size) + tuple(region[2:])
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
# For box style image display 2 planes per axis are used.  For orthoplane
# display 1 plane per axis is shown.  Returns 0 for normal (all planes)
# display style.
#
def faces_per_axis(image_style, image_mode):

  fpa = 0
  if image_style:
    if image_mode == 'box faces':
      fpa = 2
    elif image_mode == 'orthoplanes':
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
  orthoplanes = v.showing_image('orthoplanes')
  if orthoplanes:
    if depth == 1:
      ro.show_orthoplane(axis, p)
      v._drawings_need_update()
      if not extend_axes:
        return
    else:
      orthoplanes = False
      v.set_parameters(orthoplanes = False)

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

  fpa = faces_per_axis(v.image_shown, ro.image_mode)
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

  def __init__(self, v, session, axis, pstart = None, pend = None, pstep = 1, pdepth = 1):

    axis = {'x':0, 'y':1, 'z':2}.get(axis, axis)
    if pstart is None:
      pstart = (v.region[0][axis] + v.region[1][axis])//2
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

  from chimerax.map_data import points_ijk_bounds
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
    from chimerax.geometry import distance
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
    from chimerax.map_data import ArrayGridData
    g = ArrayGridData(m, xyz_min, grid.step, grid.cell_angles, grid.rotation,
                      name = grid.name)
    return g

# -----------------------------------------------------------------------------
# Open and display a map.
#
def open_volume_file(path, session, format = None, name = None, style = 'auto',
                     open_models = True, model_id = None,
                     show_data = True, show_dialog = True, verbose = False):

  from chimerax.map_data import open_file, FileFormatError
  try:
    glist = open_file(path, format, log = session.logger, verbose = verbose)
  except FileFormatError as value:
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

  vlist = [volume_from_grid_data(g, session, style, open_models,
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
    session.volume_defaults = defaultsettings.VolumeDefaultSettings(session)
  return session.volume_defaults

# -----------------------------------------------------------------------------
#
def set_data_cache(grid_data, session):
  from chimerax.map_data import ArrayGridData
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
    from chimerax.map_data import datacache
    session._volume_data_cache = dc = datacache.Data_Cache(size = size)
  return dc

# -----------------------------------------------------------------------------
# Open and display a map using Volume Viewer.
#
def volume_from_grid_data(grid_data, session, style = 'auto',
                          open_model = True, model_id = None, show_dialog = True):
  '''
  Supported API.
  Create a new :class:`.Volume` model from a :class:`~.data.GridData` instance and set its initial
  display style and color and add it to the session open models.

  Parameters
  ----------
  grid_data : :class:`~.data.GridData`
    Use this GridData to create the Volume.
  session : :class:`~chimerax.core.session.Session`
    The session that the Volume will belong to.
  style : 'auto', 'surface', 'mesh' or 'image'
    The initial display style.
  open_model : bool
    Whether to add the Volume to the session open models.
  model_id : tuple of integers
    Model id for the newly created Volume.
    It is an error if the specifid id equals the id of an existing model.
  show_dialog : bool
    Whether to show the Volume Viewer user interface panel.

  Returns
  -------
  volume : the created :class:`.Volume`
  '''

  set_data_cache(grid_data, session)

  ds = default_settings(session)
  ro = ds.rendering_option_defaults()
  if getattr(grid_data, 'polar_values', None):
    ro.flip_normals = True
    ro.cap_faces = False
  if hasattr(grid_data, 'initial_rendering_options'):
    for oname, ovalue in grid_data.initial_rendering_options.items():
      setattr(ro, oname, ovalue)

  # Create volume model
  d = data_already_opened(grid_data.path, grid_data.grid_id, session)
  if d:
    grid_data = d

  v = Volume(session, grid_data, rendering_options = ro)

  # Set display style
  if style == 'auto':
    # Show single plane data in image style.
    single_plane = [s for s in grid_data.size if s == 1]
    style = 'image' if single_plane else 'surface'
  if style is not None:
    v._style_when_shown = style

  if grid_data.rgba is None:
    if not any_volume_open(session):
      _reset_color_sequence(session)
    set_initial_volume_color(v, session)

  if not model_id is None:
    if session.models.have_id(model_id):
      from chimerax.core.errors import UserError
      raise UserError('Tried to create model #%s which already exists'
                      % '.'.join('%d'%i for i in model_id))

    v.id = model_id

  if open_model:
    session.models.add([v])

  if show_dialog:
    show_volume_dialog(session)

  return v

# -----------------------------------------------------------------------------
#
def show_volume_dialog(session):
  if hasattr(session, 'ui') and session.ui.is_gui:
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
def _reset_color_sequence(session):
  '''When all models are closed, volume color sequence is reset.'''
  ds = default_settings(session)
  ds._next_color_index = 0

# ---------------------------------------------------------------------------
#
def set_initial_volume_color(v, session):

  ds = default_settings(session)
  if ds['use_initial_colors']:
    i = getattr(ds, '_next_color_index', 0)
    if not hasattr(v.data, 'series_index') or v.data.series_index == 0:
      ds._next_color_index = i+1
    icolors = ds['initial_colors']
    rgba = icolors[i%len(icolors)]
    v.set_parameters(default_rgba = rgba)

# ---------------------------------------------------------------------------
#
def _negative_color(rgba):
  neg_rgba = tuple([1-c for c in rgba[:3]] + [rgba[3]])
  minc = max(neg_rgba[:3])
  if minc == 0:
    neg_rgba = (1,0,0,neg_rgba[3])
  elif minc < 0.7:
    neg_rgba = tuple(c/minc for c in neg_rgba[:3]) + (neg_rgba[3],)
  return neg_rgba

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
def any_volume_open(session):
  for m in session.models:
    if isinstance(m, Volume):
      return True
  return False

# -----------------------------------------------------------------------------
#
def open_map(session, path, name = None, format = None, **kw):
    '''
    Supported API. Open a density map file having any of the known density map formats.
    File path can be a string or list of paths.

    Parameters
    ----------
    session : :class:`~chimerax.core.session.Session`
       The session that the created Volume will belong to.
    path : string
       File path on disk.
    name : string or None
       Name used when creating the Volume model.  If None,
       then the name will be the file name.
    format : string or None
       Name of the file format.  The available formats can be listed
       with ChimeraX command "open formats".  If None, then the format
       is derived from the file suffix.
    channel : int
       The channel number to assign for multi-channel data.
    vseries : bool
       Whether to treat the open data as a time series.
    show : bool
       Whether the Volume should be shown or hidden initially.

    Returns
    -------
    models : list of :class:`.Volume`
    message : description of the opened data
    '''
    if name is None:
      from os.path import basename
      name = basename(path if isinstance(path, str) else path[0])

    from chimerax.map_data import open_file
    grids = open_file(path, file_type = format, log = session.logger, **kw)

    models = []
    msg_lines = []
    sgrids = []
    for grid_group in grids:
      if isinstance(grid_group, (tuple, list)):
        # Handle multiple channels or time series
        from os.path import commonprefix
        gname = commonprefix([g.name for g in grid_group])
        if len(gname) == 0:
          gname = name
        gmodels, gmsg = open_grids(session, grid_group, gname, **kw)
        models.extend(gmodels)
        msg_lines.append(gmsg)
      else:
        sgrids.append(grid_group)

    if sgrids:
      smodels, smsg = open_grids(session, sgrids, name, **kw)
      models.extend(smodels)
      msg_lines.append(smsg)

    msg = '\n'.join(msg_lines)

    return models, msg

# -----------------------------------------------------------------------------
#
def open_grids(session, grids, name, **kw):

    level = kw.get('initial_surface_level', None)
    if level is not None:
      for g in grids:
        g.initial_surface_level = level

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
            from chimerax.core.errors import UserError
            raise UserError('Cannot make series from volumes with different sizes:\n%s' % gsizes)
          g.series_index = i
      else:
        for g in grids:
          if hasattr(g, 'series_index'):
            delattr(g, 'series_index')

    maps = []
    if 'show' in kw:
      show = kw['show']
    else:
      show = (len(grids) >= 1 and getattr(grids[0], 'show_on_open', True))
    si = [d.series_index for d in grids if hasattr(d, 'series_index')]
    is_series = (len(si) == len(grids) and len(set(si)) > 1)
    cn = [d.channel for d in grids if d.channel is not None]
    is_multichannel = (len(cn) == len(grids) and len(set(cn)) > 1)
    for d in grids:
      show_data = show
      if is_series or is_multichannel:
        show_data = False	# MapSeries or MapChannelsModel classes will decide which to show
      vkw = {'show_dialog': False}
      if hasattr(d, 'initial_style') and d.initial_style in ('surface', 'mesh', 'image'):
        vkw['style'] = d.initial_style
      v = volume_from_grid_data(d, session, open_model = False, **vkw)
      maps.append(v)
      if not show_data:
        v.display = False
      set_initial_region_and_style(v)

    show_dialog = kw.get('show_dialog', True)
    if maps and show_dialog:
      show_volume_dialog(session)

    msg = ''
    if is_series and is_multichannel:
      cmaps = {}
      for m in maps:
        cmaps.setdefault(m.data.channel,[]).append(m)
      if len(set(len(cm) for cm in cmaps.values())) > 1:
        session.logger.warning('Map channels have differing numbers of series maps: %s'
                               % ', '.join('%d (%d)' % (c,cm) for c, cm in cmaps.items()))
      from chimerax.map_series import MapSeries
      ms = [MapSeries('channel %d' % c, cm, session) for c, cm in cmaps.items()]
      mc = MultiChannelSeries(name, ms, session)
      models = [mc]
    elif is_series:
      from chimerax.map_series import MapSeries
      ms = MapSeries(name, maps, session)
      ms.display = show
      models = [ms]
    elif is_multichannel:
      mc = MapChannelsModel(name, maps, session)
      mc.display = show
      mc.show_n_channels(3)
      models = [mc]
    elif len(maps) == 0:
      msg = 'No map data opened'
      session.logger.warning(msg)
      models = maps
    else:
      models = maps

    # Create surfaces before adding to session so that initial view can use corrrect bounds.
    for v in maps:
      if v.display:
        v.update_drawings()

    return models, msg

# -----------------------------------------------------------------------------
#
def set_initial_region_and_style(v):

  ro = v.rendering_options
  data = v.data
  if getattr(data, 'polar_values', False):
    ro.flip_normals = True
    ro.cap_faces = False

  ds = default_settings(v.session)
  one_plane = (getattr(data, 'initial_plane_display', False)
               or show_one_plane(data.size, ds['show_plane'], ds['voxel_limit_for_plane']))
  if one_plane:
    v._style_when_shown = 'image'
  elif show_when_opened(v, ds['show_on_open'], ds['voxel_limit_for_open']):
    if v._style_when_shown is None:
      v._style_when_shown = 'surface'
  else:
    v.display = False
  v._initial_style_set = True

  # Determine initial region bounds and step.
  region = v.full_region()[:2]
  if one_plane:
    region[0][2] = region[1][2] = data.size[2]//2

  fpa = faces_per_axis(v.image_shown, ro.image_mode)
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

  def added_to_session(self, session):
    maps = self.maps
    msg = ('Opened multi-channel map %s as #%s, %d channels'
           % (self.name, self.id_string, len(maps)))
    if maps:
      msg += ', ' + maps[0].info_string()
    session.logger.info(msg)

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

  def added_to_session(self, session):
    ms = self.map_series
    msg = ('Opened multichannel map series %s as #%s, %d channels'
           % (self.name, self.id_string, len(ms)))
    if ms and ms[0].maps:
      maps = ms[0].maps
      msg += ', %d images per channel, %s' % (len(maps), maps[0].info_string())
    session.logger.info(msg)

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
    def restore_maps(trigger_name, session, mcs = mcs, map_ids = data['map series ids']):
      idm = {m.id : m for m in mcs.child_models()}
      map_series = [idm[id] for id in map_ids if id in idm]
      mcs.set_map_series(map_series)
      from chimerax.core.triggerset import DEREGISTER
      return DEREGISTER
    session.triggers.add_handler('end restore session', restore_maps)

    return mcs

# -----------------------------------------------------------------------------
#
def save_map(session, path, format_name, models = None, region = None, step = (1,1,1),
             mask_zone = True, subsamples = None, chunk_shapes = None, append = None,
             compress = None, compress_method = None, compress_level = None, compress_shuffle = None,
             base_index = 1, value_type = None, **kw):
    '''
    Supported API.
    Save a density map file having any of the known density map formats.

    Parameters:
        session: :class:`~chimerax.core.session.Session`
            The session containing the Volume models.
        path: string
            File path on disk.  For saving multiple volumes to multiple files
            the path can contain a C-style integer format specifier like "%d" or "%03d"
            which will have be replaced by integer values starting at parameter base_index
            for each of the volumes specified in parameter models.
        format_name: string or None
            Name of the file format.  The available formats can be listed
            with ChimeraX command "save formats".  If None, then the format
            is derived from the file suffix.
        models: list of :class:`.Volume`
            Volume models to save.  Some formats allow saving multiple volumes
            in one file and some do not.  It is an error to specify multiple
            models if the format only supports saving one volume and the path
            does not contain a "%d" style integer substitution.
        region: 6 integers or None
            Save only the subregion imin,jmin,kmin,imax,jmax,kmax.  If None
            the current volume region is saved.
        step: 3 integers
            Save only subsampled data using this step.
        mask_zone: bool
            If only a zone is shown near atoms or markers write zeros outside
            that zone in the saved file if this option is True, otherwise save
            original data values outside zone.  Default True
        base_index: int
            When saving multiple files with a C-style integer substitution like "%d"
            in the path this will be the first integer used.  Default 1.


    Parameters below only supported by Chimera Map format (\*.cmap)

    Parameters:
        subsamples: list of tuples of 3 integers or None
            For file formats that support saving multiple subsampled copies of
            the data , this lists the specific subsamples to save.
            Chimera map format will automatically determine subsamples to
            save if this is not specified.
        chunk_shapes: list of 'xyz', 'xzy', 'yxz', 'yzx', 'zxy', 'zyx'
            Axis order for laying out the data in the file.
            Can save multiple axis orders for faster performance access
            of data slices from disk.
        append: bool
            Whether to append this volume to an existing file.  Default False.
        compress: bool
            Whether to compress the data in the file.  Default False.
        compress_method: string
            Compression method to use.  Default zlib.
            Some HDF5 compression methods are 'zlib', 'lzo', 'bzip2', 'blosc', 'blosc:blosclz',
            'blosc:lz4', 'blosc:lz4hc', 'blosc:snappy', 'blosc:zlib', 'blosc:zstd'.
        compress_level: integer 1 to 9
            Level of compression.  Default 5.
            Higher compression levels take longer.  Not all compression methods use level.
        compress_shuffle: bool
            Option to blosc compression.  Default False.

    Parameters below only supported for MRC format (\*.mrc)

    Parameters:
        value_type: string
            Numeric value type to save in MRC file.  Can be int8, int16, uint16, float16, float32.
            If not specified then the closest type holding the actual map value type is used.
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
          mstring = ' (#%s)' % ','.join(m.id_string for m in models) if models else ''
          from chimerax.core.errors import UserError
          raise UserError('Specified models are not volumes' + mstring)


    from chimerax.map_data.fileformats import file_writer, file_formats
    if file_writer(path, format_name) is None:
        from chimerax.core.errors import UserError
        if format_name is None:
            suffixes = ', '.join(sum([ff.suffixes for ff in file_formats if ff.writable], []))
            msg = ('Unknown file suffix for "%s", known suffixes %s'
                   % (path, suffixes))
        else:
            fmt_names = ', '.join(ff.name for ff in file_formats if ff.writable)
            msg = ('Unknown file format "%s", known formats %s'
                   % (format_name, fmt_names))
        raise UserError(msg)

    options = {}
    if subsamples is not None:
        options['subsamples'] = subsamples
    if chunk_shapes is not None:
        options['chunk_shapes'] = chunk_shapes
    if append:
        options['append'] = True
    if compress:
        options['compress'] = True
    if compress_method:
        options['compress_method'] = compress_method
        options['compress'] = True
    if compress_level:
        options['compress_level'] = compress_level
        options['compress'] = True
    if compress_shuffle is not None:
        options['compress_shuffle'] = compress_shuffle
        options['compress'] = True
    if value_type is not None:
        options['value_type'] = value_type
    if path in ('browse', 'browser'):
        from chimerax.map_data import select_save_path
        path, format_name = select_save_path()
    if path:
        grids = []
        for v in vlist:
          g = v.grid_data(region, step, mask_zone)
          color = v.model_color
          if color is not None:
            g.rgba = tuple(r/255 for r in color)	# Set default map color to current color
          grids.append(g)
        from chimerax.map_data import save_grid_data
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
    if t.has_trigger('graphics update'):
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
        if v.deleted or v.id is None:
          vset.remove(v)
          vdisp.remove(v)
        elif v.display:
          # Remove volume from update list before update since update may re-add it
          # if surface calculation done in thread.
          vset.remove(v)
          vdisp.remove(v)
          v.update_drawings()

# -----------------------------------------------------------------------------
# Check if file name contains %d type format specification.
#
def is_multifile_save(path):
    try:
        path % 0
    except Exception:
        return False
    return True

# -----------------------------------------------------------------------------
#
def add_map_format(session, map_format):
  from chimerax.map_data import file_formats
  file_formats.append(map_format)
