# -----------------------------------------------------------------------------
# Manages surface and volume display for a region of a data set.
# Holds surface and solid thresholds, color, and transparency and brightness
# factors.
#
from ..graphics import Drawing
class Volume(Drawing):
  '''
  A Volume is a rendering of a 3-d image Grid_Data object.  It includes
  color, display styles including surface, mesh and grayscale, contouring levels,
  brightness and transparency for grayscale rendering, region bounds for display
  a subregion including single plane display, subsampled display of every Nth data
  value along each axis, outline box display.
  '''
  def __init__(self, data, session, region = None, rendering_options = None,
               model_id = None, open_model = True, message_cb = None):

    Drawing.__init__(self, data.name)

    self.session = session
    if not model_id is None:
      self.id = model_id

    self.data = data
    data.add_change_callback(self.data_changed_cb)
    self.path = data.path

    if region is None:
      region = full_region(data.size)
    self.region = clamp_region(region, data.size)

    if rendering_options is None:
      rendering_options = Rendering_Options()
    self.rendering_options = rendering_options

    self.message_cb = message_cb
    
    self.matrix_stats = None
    self.matrix_id = 1          # Incremented when shape or values change.

    rlist = Region_List()
    ijk_min, ijk_max = self.region[:2]
    rlist.insert_region(ijk_min, ijk_max)
    self.region_list = rlist

    self.representation = 'surface'

    self.solid = None
    self.keep_displayed_data = None

    self.display = True        # Display surface

    self.initialized_thresholds = False

    # Surface display parameters
    self.surface_levels = []
    self.surface_colors = []
    self.surface_drawings = []    		# Drawing graphics objects
    self.surface_brightness_factor = 1
    self.transparency_factor = 0                # for surface/mesh
    self.outline_box = Outline_Box(self)

    # Solid display parameters
    self.solid_levels = []                      # list of (threshold, scale)
    self.solid_colors = []
    self.transparency_depth = 0.5               # for solid
    self.solid_brightness_factor = 1

    self.default_rgba = data.rgba

    self.change_callbacks = []

    if open_model:
      self.open_model(model_id)

#    from chimera import addModelClosedCallback
#    addModelClosedCallback(self, self.model_closed_cb)

#    from chimera import triggers
#    h = triggers.addHandler('SurfacePiece', self.surface_piece_changed_cb, None)
#    self.surface_piece_change_handler = h
    self.surface_piece_change_handler = None

  # ---------------------------------------------------------------------------
  #
  def message(self, text):

    if self.message_cb:
      self.message_cb(text)
    
  # ---------------------------------------------------------------------------
  #
  def full_name(self):

    return self.name
    
  # ---------------------------------------------------------------------------
  #
  def name_with_id(self):

    if self.subid:
      sid = '#%d.%d' % (self.id, self.subid)
    else:
      sid = '#%d' % self.id
    return self.name + ' ' + sid

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
  def call_change_callbacks(self, change_types):

    if isinstance(change_types, str):
      change_types = [change_types]

    for cb in self.change_callbacks:
      for ct in change_types:
        cb(self, ct)
    
  # ---------------------------------------------------------------------------
  # Detect surface piece color change and display style change.
  #
  def surface_piece_changed_cb(self, trigger, unused, changes):

    change = ('color changed' in changes.reasons
              or 'display style changed' in changes.reasons)
    if not change:
      return

    if len(self.surface_drawings) != len(self.surface_colors):
      return

    pindex = {}
    for i, p in enumerate(self.surface_drawings):
      pindex[p] = i

    # Check if surface pieces for this volume have changed.
    ctypes = set()
    pchange = False
    for p in changes.modified:
      if p in pindex:
        i = pindex[p]
        vcolor = self.modulated_surface_color(self.surface_colors[i])
        from numpy import array, single as floatc
        if (p.color != tuple(int(255*r) for r in vcolor)).any():
          self.surface_colors[i] = p.color / 255.0
          ctypes.add('colors changed')
        pchange = True

    # Check if display style of all surface pieces has changed.
    if pchange and self.representation != 'solid':
      styles = set()
      for p in self.surface_drawings:
        styles.add(p.display_style)
      if len(styles) == 1:
        pstyle = {p.Solid: 'surface', p.Mesh: 'mesh'}.get(styles.pop(), None)
        if pstyle and self.representation != pstyle:
          # TODO: Eliminate special case for 2d contour rendering.
          contour_2d = (pstyle == 'mesh' and
                        not self.rendering_options.cap_faces
                        and self.single_plane())
          if not contour_2d:
            self.set_representation(pstyle)

    # Notify about changes.
    self.call_change_callbacks(ctypes)
  
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
  
    The volume display is not automatically updated.  Use v.show() when
    redisplay is desired.
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
        len(kw['surface_levels']) != len(self.surface_levels)):
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

    for param in parameters:
      if param in kw:
        values = kw[param]
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
      self.new_region(r[0], r[1], r[2], show = False, adjust_step = True,
                      save_in_region_queue = False)

    if 'surface_levels' in kw or 'solid_levels' in kw:
      self.call_change_callbacks('thresholds changed')

    if ('surface_colors' in kw or 'solid_colors' in kw or
        'surface_brightness_factor' in kw or 'transparency_factor' in kw or
        'solid_brightness_factor' in kw or 'transparency_depth' in kw):
      self.call_change_callbacks('colors changed')

    if option_changed:
      self.call_change_callbacks('rendering options changed')

  # ---------------------------------------------------------------------------
  #
  def new_region(self, ijk_min = None, ijk_max = None, ijk_step = None,
                 show = True, adjust_step = True, save_in_region_queue = True):
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
    elif tuple(ijk_step) != tuple(adjusted_ijk_step):
      # Change automatic step adjustment voxel limit.
      vc = subarray_size(ijk_min, ijk_max, ijk_step, fpa)
      ro.voxel_limit = (1.01*vc) / (2**20)  # Mvoxels rounded up for gui value
      self.call_change_callbacks('voxel limit changed')

    if save_in_region_queue:
      self.region_list.insert_region(ijk_min, ijk_max)

    region = (ijk_min, ijk_max, ijk_step)
    if self.same_region(region, self.region):
      return False

    self.region = region
    self.matrix_changed()

    self.call_change_callbacks('region changed')

    if show:
      self.show()

    return True

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
  def has_thresholds(self):

    return len(self.surface_levels) > 0 and len(self.solid_levels) > 0

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

    if replace or len(self.surface_levels) == 0:
      if mfrac is None:
        v = s.rank_data_value(1-vfrac[0])
      else:
        v = s.mass_rank_data_value(1-mfrac[0])
      rgba = self.default_rgba
      if polar:
        self.surface_levels = [-v,v]
        neg_rgba = tuple([1-c for c in rgba[:3]] + [rgba[3]])
        self.surface_colors = [neg_rgba,rgba]
      else:
        self.surface_levels = [v]
        self.surface_colors = [rgba]

    if replace or len(self.solid_levels) == 0:
      if mfrac is None:
        vlow = s.rank_data_value(1-vfrac[1])
        vmid = s.rank_data_value(1-vfrac[0])
      else:
        vlow = s.mass_rank_data_value(1-mfrac[1])
        vmid = s.mass_rank_data_value(1-mfrac[0])
      vmax = s.maximum
      rgba = saturate_rgba(self.default_rgba)
      if polar:
        self.solid_levels = ((s.minimum,1), (max(-vmid,s.minimum),0.99), (0,0),
                             (0,0), (vmid,0.99), (vmax,1))
        neg_rgba = tuple([1-c for c in rgba[:3]] + [rgba[3]])
        self.solid_colors = (neg_rgba,neg_rgba,neg_rgba, rgba,rgba,rgba)
      else:
        if vlow < vmid and vmid < vmax:
          self.solid_levels = ((vlow,0), (vmid,0.99), (vmax,1))
        else:
          self.solid_levels = ((vlow,0), (vmax,1))
        self.solid_colors = [rgba]*len(self.solid_levels)

    self.initialized_thresholds = True

    self.call_change_callbacks('thresholds changed')

    return True

  # ---------------------------------------------------------------------------
  #
  def set_representation(self, rep):
    '''
    Set display style to "surface", "mesh", or "solid".
    '''
    if rep != self.representation:
      self.redraw_needed()  # Switch to solid does not change surface until draw
      if rep == 'solid' or self.representation == 'solid':
        ro = self.rendering_options
        adjust_step = (ro.box_faces or ro.any_orthoplanes_shown())
      else:
        adjust_step = False
      self.representation = rep
      if adjust_step:
        ijk_min, ijk_max = self.region[:2]
        self.new_region(ijk_min, ijk_max, show = False)
      self.call_change_callbacks('representation changed')

  # ---------------------------------------------------------------------------
  #
  def show(self, representation = None, rendering_options = None, show = True):
    '''
    Display the volume using the current parameters.
    '''
    if representation:
      self.set_representation(representation)

    if rendering_options:
      self.rendering_options = rendering_options

    if show:
      # Prevent cached matrix for displayed data from being freed.
#       from chimera import CancelOperation
      try:
        self.keep_displayed_data = self.displayed_matrices()
      except CancelOperation:
        return
    else:
      self.keep_displayed_data = None   # Release data if not shown.
      
    if self.representation == 'surface' or self.representation == 'mesh':
      self.hide_solid()
      show_mesh = (self.representation == 'mesh')
      self.show_surface(show, show_mesh, self.rendering_options)
    elif self.representation == 'solid':
      self.hide_surface()
      self.show_solid(show, self.rendering_options)

    if show:
      self.call_change_callbacks('displayed')
      
  # ---------------------------------------------------------------------------
  #
  def show_surface(self, show, show_mesh, rendering_options):

    if show:
      self.update_surface(show_mesh, rendering_options)
      self.display = True
      for p in self.surface_drawings:
        p.display = True
    else:
      self.hide_surface()

  # ---------------------------------------------------------------------------
  #
  def update_display(self):

    if self.representation == 'solid':
      self.update_solid()
    else:
      self.update_surface()

  # ---------------------------------------------------------------------------
  #
  def update_surface(self, show_mesh = None, rendering_options = None):

    pieces = self.match_surface_pieces(self.surface_levels)
    self.surface_drawings = pieces
    if show_mesh is None:
      show_mesh = (self.representation == 'mesh')
    ro = self.rendering_options if rendering_options is None else rendering_options
#    from chimera import CancelOperation
    try:
      for k, level in enumerate(self.surface_levels):
        color = self.surface_colors[k]
        rgba = self.modulated_surface_color(color)
        p = pieces[k]
        p.name = 'level %.4g' % level
        self.update_surface_piece(level, rgba, show_mesh, ro, p)
    except CancelOperation:
      pass

    self.show_outline_box(ro.show_outline_box, ro.outline_box_rgb,
                          ro.outline_box_linewidth)

  # ---------------------------------------------------------------------------
  # Pair up surface pieces with contour levels.  Aim is to avoid
  # recalculating contours if a piece already exists for a contour
  # level.  Common cases are 1) no levels have changed, 2) one level
  # has changed, 3) one level added or deleted, 4) multiple levels
  # added or deleted.  Level order is typically preserved.
  #
  def match_surface_pieces(self, levels):

    smodel = self
    plist = [p for p in self.surface_drawings if not p.was_deleted]
    for k,level in enumerate(levels):
      if k < len(plist) and level == plist[k].contour_settings['level']:
        pass
      elif (k+1 < len(plist) and k+1 < len(levels) and
            levels[k+1] == plist[k+1].contour_settings['level']):
        pass
      elif k+1 < len(plist) and level == plist[k+1].contour_settings['level']:
        smodel.remove_drawing(plist[k])
        del plist[k]
      elif (k < len(plist) and k+1 < len(levels) and
            levels[k+1] == plist[k].contour_settings['level']):
        plist.insert(k, smodel.new_drawing())
      elif k >= len(plist):
        plist.append(smodel.new_drawing())

    while len(plist) > len(levels):
      smodel.remove_drawing(plist[-1])
      del plist[-1]
      
    return plist
  
  # ---------------------------------------------------------------------------
  #
  def update_surface_piece(self, level, rgba, show_mesh, rendering_options,
                           piece):

    ro = rendering_options
    p = piece

    contour_settings = {'level': level,
                        'matrix_id': self.matrix_id,
                        'transform': self.matrix_indices_to_xyz_transform(),
                        'surface_smoothing': ro.surface_smoothing,
                        'smoothing_factor': ro.smoothing_factor,
                        'smoothing_iterations': ro.smoothing_iterations,
                        'subdivide_surface': ro.subdivide_surface,
                        'subdivision_levels': ro.subdivision_levels,
                        'square_mesh': ro.square_mesh,
                        'cap_faces': ro.cap_faces,
                        'flip_normals': ro.flip_normals,
                        }
    if (not hasattr(p, 'contour_settings') or
        p.contour_settings != contour_settings):
      if self.calculate_contour_surface(level, rendering_options, p):
        p.contour_settings = contour_settings

    p.color = tuple(int(255*r) for r in rgba)

    # OpenGL draws nothing for degenerate triangles where two vertices are
    # identical.  For 2d contours want to see these triangles so show as mesh.
    single_plane = self.single_plane()
    contour_2d = single_plane and not ro.cap_faces

    style = p.Mesh if show_mesh or contour_2d else p.Solid
    p.display_style = style
    
    if contour_2d:  lit = False
    elif show_mesh: lit = ro.mesh_lighting
    else:           lit = True
    p.use_lighting = lit

    p.twoSidedLighting = ro.two_sided_lighting

    p.lineThickness = ro.line_thickness

    p.smoothLines = ro.smooth_lines

#     if ro.dim_transparency:
#       bmode = p.SRC_ALPHA_DST_1_MINUS_ALPHA
#     else:
#       bmode = p.SRC_1_DST_1_MINUS_ALPHA
#     p.transparencyBlendMode = bmode
      
  # ---------------------------------------------------------------------------
  #
  def calculate_contour_surface(self, level, rendering_options, piece):

    name = self.data.name
    self.message('Computing %s surface, level %.3g' % (name, level))

    matrix = self.matrix()

    # map_cpp contour code does not handle single data planes.
    # Handle these by stacking two planes on top of each other.
    plane_axis = [a for a in (0,1,2) if matrix.shape[a] == 1]
    for a in plane_axis:
      matrix = matrix.repeat(2, axis = a)

    ro = rendering_options

    from ..map_cpp import surface
    try:
      varray, tarray, narray = surface(matrix, level,
                                       cap_faces = ro.cap_faces,
                                       calculate_normals = True)
    except MemoryError:
      from chimera.replyobj import warning
      warning('Ran out of memory contouring at level %.3g.\n' % level +
              'Try a higher contour level.')
      return False

    for a in plane_axis:
      varray[:,2-a] = 0
    
    if ro.flip_normals and level < 0:
      from _surface import invert_vertex_normals
      invert_vertex_normals(narray, tarray)

    # Preserve triangle vertex traversal direction about normal.
    transform = self.matrix_indices_to_xyz_transform()
    if transform.determinant() < 0:
      from ..map_cpp import reverse_triangle_vertex_order
      reverse_triangle_vertex_order(tarray)

    if ro.subdivide_surface:
      from _surface import subdivide_triangles
      for level in range(ro.subdivision_levels):
        varray, tarray, narray = subdivide_triangles(varray, tarray, narray)

    if ro.square_mesh:
      from numpy import empty, int32
      hidden_edges = empty((len(tarray),), int32)
      from .. import map_cpp
      map_cpp.principle_plane_edges(varray, tarray, hidden_edges)

    if ro.surface_smoothing:
      sf, si = ro.smoothing_factor, ro.smoothing_iterations
      from _surface import smooth_vertex_positions
      smooth_vertex_positions(varray, tarray, sf, si)
      smooth_vertex_positions(narray, tarray, sf, si)

    # Transform vertices and normals
    transform.move(varray)
    tf = transform.inverse().transpose().zero_translation()
    tf.move(narray)
    from ..geometry import vector
    vector.normalize_vectors(narray)

    self.message('Making %s surface with %d triangles' % (name, len(tarray)))

    p = piece
    p.geometry = varray, tarray
    p.normals = narray

    if ro.square_mesh:
      p.set_edge_mask(hidden_edges)
    else:
      p.set_edge_mask(None)

    self.message('')

    return True
    
  # ---------------------------------------------------------------------------
  #
  def remove_surfaces(self):

    for p in self.surface_drawings:
      if not p.was_deleted:
        self.remove_drawing(p)
    self.surface_drawings = []
    
  # ---------------------------------------------------------------------------
  #
  def open_model(self, model_id):

    return
    from chimera import openModels
    if model_id == None:
      m_id = m_subid = openModels.Default
    elif isinstance(model_id, int):
      m_id = model_id
      m_subid = openModels.Default
    else:
      m_id, m_subid = model_id
    openModels.add([self], baseId = m_id, subid = m_subid)

  # ---------------------------------------------------------------------------
  #
  def show_outline_box(self, show, rgb, linewidth):
    '''
    Show an outline box enclosing the displayed subregion of the volume.
    '''
    if show and rgb:
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
  def show_solid(self, show, rendering_options):

    if show:
      self.update_solid(rendering_options)
      self.display = True
    else:
      self.hide_solid()

  # ---------------------------------------------------------------------------
  #
  def update_solid(self, rendering_options = None):

    s = self.solid
    if s is None:
      s = self.make_solid()
      self.solid = s

    ro = self.rendering_options if rendering_options is None else rendering_options
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
    s.set_matrix(self.matrix_size(), self.data.value_type, self.matrix_id,
                 self.matrix_plane)

    s.update_model(self)
    s.show()

    self.show_outline_box(ro.show_outline_box, ro.outline_box_rgb,
                          ro.outline_box_linewidth)


  # ---------------------------------------------------------------------------
  #
  def make_solid(self):

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

    return self.display
    
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
  def copy(self):

    v = volume_from_grid_data(self.data, self.session, self.representation,
                              show_data = False, show_dialog = False)
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
        surface_levels = v.surface_levels,
        solid_levels = v.solid_levels,
        )
    if copy_colors:
      # Copy colors
      self.set_parameters(
        surface_colors = v.surface_colors,
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
      self.new_region(ijk_min, ijk_max, ijk_step, show = False)

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
                    show = True, unshow_original = True, model_id = None,
                    subregion = None, step = (1,1,1), name = None,
                    copy_colors = True):

    r = self.subregion(step, subregion)
    if not require_copy and self.data.writable and self.is_full_region(r):
      return self

    g = self.region_grid(r)
    g.array[:,:,:] = self.region_matrix(r)

    if name:
      g.name = name
    elif self.name.endswith('copy'):
      g.name = self.name
    else:
      g.name = self.name + ' copy'

    v = volume_from_grid_data(g, self.session, self.representation, model_id = model_id,
                              show_data = False, show_dialog = False)
    v.copy_settings_from(self, copy_region = False, copy_colors = copy_colors)

    # Display copy and undisplay original
    if show:
      v.show()
    if unshow_original:
      self.unshow()

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
    from .data import Array_Grid_Data
    g = Array_Grid_Data(m, origin, step, d.cell_angles, d.rotation)
    g.rgba = d.rgba           # Copy default data color.
    return g

  # ---------------------------------------------------------------------------
  #
  def bounds(self, positions = True):

    b = Drawing.bounds(self, positions)
    if b is None:
      # TODO: Should this be only displayed bounds?
      b = self.xyz_bounds()
    return b

  # ---------------------------------------------------------------------------
  # The xyz bounding box encloses the subsampled grid with half a step size
  # padding on all sides.
  #
  def xyz_bounds(self, step = None, subregion = None):

    ijk_min_edge, ijk_max_edge = self.ijk_bounds(step, subregion)
    
    from .data import box_corners
    ijk_corners = box_corners(ijk_min_edge, ijk_max_edge)
    data = self.data
    xyz_min, xyz_max = bounding_box([data.ijk_to_xyz(c) for c in ijk_corners])
    
    return (xyz_min, xyz_max)

  # ---------------------------------------------------------------------------
  #
  def first_intercept(self, mxyz1, mxyz2, exclude = None):

    if self.representation == 'solid':
      ro = self.rendering_options
      if not ro.box_faces and ro.orthoplanes_shown == (False,False,False):
        vxyz1, vxyz2 = self.position.inverse() * (mxyz1, mxyz2)
        from . import slice
        xyz_in, xyz_out = slice.box_line_intercepts((vxyz1, vxyz2), self.xyz_bounds())
        if xyz_in is None or xyz_out is None:
          return None, None
        from ..geometry.vector import norm
        f = norm(0.5*(xyz_in+xyz_out) - mxyz1) / norm(mxyz2 - mxyz1)
        return f, None

    return Drawing.first_intercept(self, mxyz1, mxyz2, exclude)

  # ---------------------------------------------------------------------------
  # The data ijk bounds with half a step size padding on all sides.
  #
  def ijk_bounds(self, step = None, subregion = None, integer = False):

    ss_origin, ss_size, subsampling, ss_step = self.ijk_region(step, subregion)
    ijk_origin = [a*b for a,b in zip(ss_origin, subsampling)]
    ijk_step = [a*b for a,b in zip(subsampling, ss_step)]
    mat_size = [(a+b-1)//b for a,b in zip(ss_size, ss_step)]
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
  def bounding_region(self, points, padding = 0, step = None, clamp = True):

    d = self.data
    from .data import points_ijk_bounds
    ijk_min, ijk_max = points_ijk_bounds(points, padding, d)
    if clamp:
      ijk_min, ijk_max = clamp_region((ijk_min, ijk_max, None), d.size)[:2]
    r = self.integer_region(ijk_min, ijk_max, step)
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
    from ..geometry import vector
    vn = vector.normalize_vector(v)
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
    origin, size, subsampling, step = self.subsample_region(region)
    d = self.data
    operation = 'reading %s' % d.name
    from .data import Progress_Reporter
    progress = Progress_Reporter(operation, size, d.value_type.itemsize)
    from_cache_only = not read_matrix
    if subsampling == (1,1,1):
      m = d.matrix(origin, size, step, progress, from_cache_only)
    else:
      m = d.matrix(origin, size, step, progress, from_cache_only, subsampling)
    return m

  # ---------------------------------------------------------------------------
  # Size of matrix for subsampled subregion returned by matrix().
  #
  def matrix_size(self, step = None, subregion = None, region = None,
                  clamp = True):

    if region is None:
      region = self.subregion(step, subregion)
    ss_origin, ss_size, subsampling, ss_step = self.subsample_region(region,
                                                                     clamp)
    mat_size = [(a+b-1)//b for a,b in zip(ss_size, ss_step)]
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
    ss_origin, ss_size, subsampling, ss_step = self.subsample_region(region,
                                                                     clamp)
    step = [sm*st for sm,st in zip(subsampling, ss_step)]
    origin = [o*st for o,st in zip(ss_origin, subsampling)]
    size = [(a+b-1)//b for a,b in zip(ss_size,ss_step)]
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
    s = [slice(None), slice(None), slice(None)]
    s[2-axis] = 0
    m2d = m[s]
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
        self.new_region(ijk_min, ijk_max, show = False)

  # ---------------------------------------------------------------------------
  # Transform mapping matrix indices to xyz.  The matrix indices are not the
  # same as the data indices since the matrix includes only the current
  # subregion and subsampled data values.
  #
  def matrix_indices_to_xyz_transform(self, step = None, subregion = None):

    ss_origin, ss_size, subsampling, ss_step = self.ijk_region(step, subregion)
    ijk_origin = [a*b for a,b in zip(ss_origin, subsampling)]
    ijk_step = [a*b for a,b in zip(subsampling, ss_step)]

    data = self.data
    xo, yo, zo = data.ijk_to_xyz(ijk_origin)
    io, jo, ko = ijk_origin
    istep, jstep, kstep = ijk_step
    xi, yi, zi = data.ijk_to_xyz((io+istep, jo, ko))
    xj, yj, zj = data.ijk_to_xyz((io, jo+jstep, ko))
    xk, yk, zk = data.ijk_to_xyz((io, jo, ko+kstep))
    from ..geometry.place import Place
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
      self.call_change_callbacks('data values changed')
      # TODO: should this automatically update the data display?
    elif type == 'coordinates changed':
      self.call_change_callbacks('coordinates changed')
    elif type == 'path changed':
      self.name = utf8_string(self.data.name)

  # ---------------------------------------------------------------------------
  # Return the origin and size of the subsampled submatrix to be read.
  #
  def ijk_region(self, step = None, subregion = None):

    r = self.subregion(step, subregion)
    return self.subsample_region(r)

  # ---------------------------------------------------------------------------
  # Return the origin and size of the subsampled submatrix to be read.
  # Also return the subsampling factor and additional step (ie stride) that
  # must be used to get the displayed data.
  #
  def subsample_region(self, region, clamp = True):

    ijk_min, ijk_max, ijk_step = region
    
    # Samples always have indices divisible by step, so increase ijk_min if
    # needed to make it a multiple of ijk_step.
    m_ijk_min = [s*((i+s-1)//s) for i,s in zip(ijk_min, ijk_step)]

    # If region is non-empty but contains no step multiple decrease ijk_min.
    for a in range(3):
      if m_ijk_min[a] > ijk_max[a] and ijk_min[a] <= ijk_max[a]:
        m_ijk_min[a] -= ijk_step[a]

    subsampling, ss_full_size = self.choose_subsampling(ijk_step)

    ss_origin = [(i+s-1)//s for i,s in zip (m_ijk_min, subsampling)]
    if clamp:
      ss_origin = [max(i,0) for i in ss_origin]
    ss_end = [(i+s)//s for i,s in zip(ijk_max, subsampling)]
    if clamp:
      ss_end = [min(i,lim) for i,lim in zip(ss_end, ss_full_size)]
    ss_size = [e-o for e,o in zip(ss_end, ss_origin)]
    ss_step = [s//d for s,d in zip(ijk_step, subsampling)]

    return tuple(ss_origin), tuple(ss_size), tuple(subsampling), tuple(ss_step)

  # ---------------------------------------------------------------------------
  # Return the subsampling and size of subsampled matrix for the requested
  # ijk_step.
  #
  def choose_subsampling(self, ijk_step):
    
    data = self.data
    if not hasattr(data, 'available_subsamplings'):
      return (1,1,1), data.size

    compatible = []
    for step, grid in data.available_subsamplings.items():
      if (ijk_step[0] % step[0] == 0 and
          ijk_step[1] % step[1] == 0 and
          ijk_step[2] % step[2] == 0):
        e = ((ijk_step[0] // step[0]) *
             (ijk_step[1] // step[1]) *
             (ijk_step[2] // step[2]))
        compatible.append((e, step, grid.size))

    if len(compatible) == 0:
      return (1,1,1), data.size

    subsampling, size = min(compatible)[1:]
    return subsampling, size

  # ---------------------------------------------------------------------------
  # Applying point_xform to points gives Chimera world coordinates.  If the
  # point_xform is None then the points are in local volume coordinates.
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
  def add_interpolated_values(self, v, subregion = 'all', step = (1,1,1),
                              scale = 1):

    if scale == 0:
      return

    values, const_values = v.interpolate_on_grid(self, subregion, step)

    # Add scaled copy of array.
    d = self.data
    m = d.full_matrix()
    if scale == 'minrms':
      level = min(v.surface_levels) if v.surface_levels else 0
      scale = minimum_rms_scale(values, m, level)
      from chimera.replyobj import info
      info('Minimum RMS scale factor for "%s" above level %.5g\n'
           '  subtracted from "%s" is %.5g\n'
           % (v.name_with_id(), level, self.name_with_id(), -scale))
    if scale == 1:
      m[:,:,:] += values
    elif scale == -1:
      m[:,:,:] -= values
    else:
      # Avoid copying array unless needed for scaling.
      if const_values:
        values = values.copy()
      values *= scale
      m[:,:,:] += values
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
    mt.move(points)
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
      m2s_transform = scene_to_source_tf * self.position * m2s_transform
      
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
      from .data import Grid_Subregion
      sg = Grid_Subregion(self.data, ijk_min, ijk_max, ijk_step)

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
    mv = volume_from_grid_data(masked_data, self.session, show_data = False)
    mv.copy_settings_from(self, copy_region = False, copy_zone = False)
    mv.show()
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
      
    self.message('Computing histogram for %s' % self.name)
    from . import data
    self.matrix_stats = ms = data.Matrix_Value_Statistics(matrices)
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
  def modulated_surface_color(self, rgba):

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
    if self.representation == 'solid' and self.solid:
      return 'a' in self.solid.color_mode
    from ..graphics import Drawing
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
  def unshow(self):

    self.display = False
  
  # ---------------------------------------------------------------------------
  #
  def hide_surface(self):

    for p in self.surface_drawings:
      p.display = False
  
  # ---------------------------------------------------------------------------
  #
  def hide_solid(self):

    s = self.solid
    if s:
      s.hide()
    
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
      s.close_model()
      self.solid = None
      
  # ---------------------------------------------------------------------------
  #
  def close(self):

    self.close_models()
      
  # ---------------------------------------------------------------------------
  #
  def delete(self):

    self.close()
      
  # ---------------------------------------------------------------------------
  #
  def model_closed_cb(self, model):

    if self.data:
      self.data.remove_change_callback(self.data_changed_cb)
      self.data = None
      self.keep_displayed_data = None
      self.outline_box = None   # Remove reference loops
      from chimera import triggers
      triggers.deleteHandler('SurfacePiece', self.surface_piece_change_handler)
      self.surface_piece_change_handler = None

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

    if not corners is None and rgb:
      from numpy import any
      changed = (any(corners != self.corners) or
                 rgb != self.rgb or
                 linewidth != self.linewidth or
                 any(center != self.center) or
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
    p = self.model.new_drawing()
    p.display_style = p.Mesh
    p.lineThickness = linewidth
    p.use_lighting = False
    p.is_outline_box = True # Do not cap clipped outline box.
    # Set geometry after setting outline_box attribute to avoid undesired
    # coloring and capping of outline boxes.
    from numpy import array
    p.geometry = array(vlist), array(tlist)
    p.triangle_and_edge_mask = hide_diagonals
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

  same = (region1 == region2 and
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
    self.voxel_limit = 1                    # Mvoxels
    self.color_modes = (
      'auto4', 'auto8', 'auto12', 'auto16',
      'opaque4', 'opaque8', 'opaque12', 'opaque16',
      'rgba4', 'rgba8', 'rgba12', 'rgba16',
      'rgb4', 'rgb8', 'rgb12', 'rgb16',
      'la4', 'la8', 'la12', 'la16',
      'l4', 'l8', 'l12', 'l16')
    self.color_mode = 'auto8'         # solid rendering pixel formats
                                      #  (auto|opaque|rgba|rgb|la|l)(4|8|12|16)
    self.projection_modes = ('auto', '2d-xyz', '2d-x', '2d-y', '2d-z', '3d')
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
def show_planes(v, axis, plane, depth = 1, extend_axes = [], show = True,
                save_in_region_queue = True):

  p = int(plane)
  ro = v.rendering_options
  orthoplanes = v.showing_orthoplanes()
  if orthoplanes:
    if depth == 1:
      ro.show_orthoplane(axis, p)
      if not extend_axes:
        if show:
          v.show()
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

  changed = v.new_region(ijk_min, ijk_max, step, show = show,
                         save_in_region_queue = save_in_region_queue)
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
    session.main_window.view.add_new_frame_callback(self.handler)

  def next_plane_cb(self):
    
    p = self.plane
    if self.step * (self.plast - p) >= 0:
      self.plane += self.step
      show_planes(self.volume, self.axis, p, self.depth,
                  save_in_region_queue = False)
    else:
      self.session.main_window.view.remove_new_frame_callback(self.handler)
      self.handler = None

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
def is_empty_region(ijk_region):

  ijk_min, ijk_max, ijk_step = ijk_region
  ijk_size = [a - b + 1 for a,b in zip(ijk_max, ijk_min)]
  if filter(lambda size: size <= 0, ijk_size):
    return 1
  return 0

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
    from ..geometry.vector import distance
    d = max(distance(ijk_to_xyz((0,0,0)), ijk_to_xyz((imax,jmax,kmax))),
            distance(ijk_to_xyz((0,0,kmax)), ijk_to_xyz((imax,jmax,0))),
            distance(ijk_to_xyz((0,jmax,0)), ijk_to_xyz((imax,0,kmax))),
            distance(ijk_to_xyz((0,jmax,kmax)), ijk_to_xyz((imax,0,0))))
    return d

# -----------------------------------------------------------------------------
#
from .data import bounding_box

# -----------------------------------------------------------------------------
#
def transformed_points(points, tf):

  from numpy import array, single as floatc
  tf_points = array(points, floatc)
  tf.move(tf_points)
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
    v = volume_from_grid_data(g, volume.session, show_data = False)
    v.copy_settings_from(volume, copy_region = False)

    return v

# ----------------------------------------------------------------------------
#
def atom_bounds(atoms, pad, volume):

    # Get atom positions.
    from _multiscale import get_atom_coordinates
    xyz = get_atom_coordinates(atoms, transformed = True)

    # Transform atom coordinates to volume ijk indices.
    tf = volume.data.xyz_to_ijk_transform * volume.model_transform().inverse()
    tf.move(xyz)
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
    from .data import Array_Grid_Data
    g = Array_Grid_Data(m, xyz_min, grid.step, grid.cell_angles, grid.rotation,
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
  except data.File_Format_Error as value:
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
                                 model_id, show_data, show_dialog)
            for g in glist]
  return vlist

# -----------------------------------------------------------------------------
# Open and display a map using Volume Viewer.
#
def volume_from_grid_data(grid_data, session, representation = None,
                          open_model = True, model_id = None,
                          show_data = True, show_dialog = False):

#  if show_dialog:
  if False:
    import chimera
    if not chimera.nogui:
      from .volumedialog import show_volume_dialog
      show_volume_dialog()

  # Set display style
  if representation is None:
    # Show single plane data in solid style.
    single_plane = [s for s in grid_data.size if s == 1]
    representation = 'solid' if single_plane else 'surface'
  ds = session.volume_defaults
  one_plane = show_one_plane(grid_data.size, ds['show_plane'],
                             ds['voxel_limit_for_plane'])
  if one_plane:
    representation = 'solid'

  # Determine initial region bounds and step.
  region = full_region(grid_data.size)[:2]
  if one_plane:
    region[0][2] = region[1][2] = grid_data.size[2]//2
  ro = ds.rendering_option_defaults()
  if hasattr(grid_data, 'polar_values') and grid_data.polar_values:
    ro.flip_normals = True
    ro.cap_faces = False
  fpa = faces_per_axis(representation, ro.box_faces,
                       ro.any_orthoplanes_shown())
  ijk_step = ijk_step_for_voxel_limit(region[0], region[1], (1,1,1), fpa,
                                      ro.limit_voxel_count, ro.voxel_limit)
  region = tuple(region) + (ijk_step,)

  # Create volume model
  d = data_already_opened(grid_data.path, grid_data.grid_id, session)
  if d:
    grid_data = d
  v = Volume(grid_data, session, region, ro, model_id, open_model)
  v.set_representation(representation)
  set_initial_volume_color(v, session)

  # Show data
  if show_data:
    if show_when_opened(v, ds['show_on_open'], ds['voxel_limit_for_open']):
      v.initialize_thresholds()
      v.show()
    else:
      v.message('%s not shown' % v.name)

  if open_model:
    session.add_model(v)

  return v

# -----------------------------------------------------------------------------
#
class CancelOperation(BaseException):
  pass

# -----------------------------------------------------------------------------
# Decide whether a data region is small enough to show when opened.
#
def show_when_opened(data_region, show_on_open, max_voxels):

  if not show_on_open:
    return False
  
  if max_voxels == None:
    return False
  
  voxel_limit = int(max_voxels * (2 ** 20))
  ss_origin, ss_size, subsampling, ss_step = data_region.ijk_region()
  voxels = float(ss_size[0]) * float(ss_size[1]) * float(ss_size[2])

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

  ds = session.volume_defaults
  if ds['use_initial_colors']:
    vlist = volume_list(session)
    n = len(vlist)
    if v in vlist:
      n -= 1
    icolors = ds['initial_colors']
    rgba = icolors[n%len(icolors)]
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
  return session.maps()
