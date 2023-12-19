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
# Save and restore volume viewer state.
#
def map_states(session):
  d2v = {}
  for v in session.maps():
    d2v.setdefault(v.data,[]).append(v)
  s = state_from_maps(d2v, session_path = session.session_file_path)
  return s

# -----------------------------------------------------------------------------
#
def restore_maps(dms, session, file_paths = None, attributes_only = False):
  if attributes_only:
    restore_map_attributes(dms, session)
  else:
    create_maps_from_state(dms, session, file_paths)
  return True

# -----------------------------------------------------------------------------
# Restore map attributes for a scene.
#
def restore_map_attributes(dms, session):
  set_maps_attributes(dms, session)

# ---------------------------------------------------------------------------
#
def state_from_maps(data_maps, include_unsaved_volumes = True, session_path = None):

  dvlist = []
  unsaved_data = []
  for data, volumes in data_maps.items():
    if data.path == '' and not include_unsaved_volumes:
      unsaved_data.append(data)
      continue                # Do not save data sets with no path
    dvlist.append((state_from_grid_data(data, session_path),
                   [state_from_map(v) for v in volumes]))

  if unsaved_data:
    names = ', '.join([d.name for d in unsaved_data])
    from chimera.replyobj import warning
    warning('Volume data sets\n\n' +
            '\t%s\n\n' % names +
            'were not saved in the session.  To have them included\n' +
            'in the session they must be saved in separate volume\n' +
            'files before the session is saved.  The session file only\n' +
            'records file system paths to the volume data.')

  return dvlist

# ---------------------------------------------------------------------------
#
def create_maps_from_state(dms, session, file_paths):

  # Cache of GridData objects improves load speed when multiple regions
  # are using same data file.  Especially important for files that contain
  # many data arrays.
  gdcache = {}        # (path, grid_id) -> GridData object
  for ds, vslist in dms:
    data = grid_data_from_state(ds, gdcache, session, file_paths)
    if data:        # Can be None if user does not replace missing file.
      for vs in vslist:
        v = create_map_from_state(vs, data, session)
        session.add_model(v)

# ---------------------------------------------------------------------------
# Used for scene restore using already existing volume models.
#
def set_maps_attributes(dms, session):

  for ds, vslist in dms:
    volumes = [find_volume_by_session_id(vs['session_volume_id'], session)
               for vs in vslist]
    dset = set(v.data for v in volumes if not v is None)
    for data in dset:
      set_grid_data_attributes(ds, data)
    for vs, volume in zip(vslist, volumes):
      if volume:
        set_map_state(vs, volume)

# -----------------------------------------------------------------------------
#
def session_volume_id(v):

    # Generate a unique volume id as a random string of characters.
    if not hasattr(v, 'session_volume_id'):
      import random, string
      sid = ''.join(random.choice(string.printable) for i in range(32))
      v.session_volume_id = sid
    return v.session_volume_id

# -----------------------------------------------------------------------------
#
def find_volume_by_session_id(id, session):

  for v in session.maps():
    if hasattr(v, 'session_volume_id') and v.session_volume_id == id:
      return v
  return None

# -----------------------------------------------------------------------------
#
def find_volumes_by_session_id(ids, session):

  idv = dict((id,None) for id in ids)
  for v in session.maps():
    if hasattr(v, 'session_volume_id') and v.session_volume_id in idv:
      idv[v.session_volume_id] = v
  return [idv[id] for id in ids]

# -----------------------------------------------------------------------------
# Path can be a tuple of paths.
#
def absolute_path(path, file_paths, ask = False, base_path = None):

  from os.path import abspath
  if isinstance(path, (tuple, list)):
    fpath = [full_path(p, base_path) for p in path]
    apath = file_paths.find_multiple(fpath,ask)
    apath = tuple(abspath(p) for p in apath if p)
  elif path == '':
    return path
  else:
    fpath = full_path(path, base_path)
    apath = file_paths.find(fpath,ask)
    if not apath is None:
      apath = abspath(apath)
  return apath

# -----------------------------------------------------------------------------
# If path is relative use specified directory to produce absolute path.
#
def full_path(path, base_path):
  from os.path import isabs, join, dirname
  if isabs(path) or base_path is None:
    return path
  return join(dirname(base_path), path)

# -----------------------------------------------------------------------------
# Path can be a tuple of paths.
#
def relative_path(path, base_path):

  if base_path is None:
    return path
  
  if isinstance(path, (tuple, list)):
    return tuple([relative_path(p, base_path) for p in path])


  from os.path import dirname, join, abspath
  bpath = abspath(base_path)
  d = join(dirname(bpath), '')       # Make directory end with "/".
  if not path.startswith(d):
    return path

  rpath = path[len(d):]
  return rpath

# ---------------------------------------------------------------------------
# Get ChimeraX unique GridDataState object for a grid.
#
def grid_data_state(grid_data, session, include_maps = False):
  gs = getattr(session, '_volume_grid_data_session_states', None)
  if gs is None:
    session._volume_grid_data_session_states = gs = {}

  if len(gs) == 0:
    # Clear dictionary at the end of session restore.
    session.triggers.add_handler("end save session", lambda t,td,gs=gs: gs.clear())

  gds = gs.get(grid_data, None)
  if gds is None:
    gs[grid_data] = gds = GridDataState(grid_data, include_maps = include_maps)
  return gds

# ---------------------------------------------------------------------------
# Encapsulate GridData state for session saving in ChimeraX.
#
from chimerax.core.state import State
class GridDataState(State):

  def __init__(self, grid_data, include_maps = False):
    self.grid_data = grid_data
    self._include_maps = include_maps

  # State save/restore in ChimeraX
  def take_snapshot(self, session, flags):
    try:
      data = state_from_grid_data(self.grid_data, session_path = session.session_file_path,
                                  include_maps = self._include_maps)
    except MemoryError as mem_error:
      from .volume import Volume
      saved_maps = [v for v in session.models.list(type = Volume)
                    if self._include_maps or not v.data.path]
      map_sizes = [v.data.voxel_count() * v.data.value_type.itemsize for v in saved_maps]
      map_gbytes = sum(map_sizes) / (1024 ** 3)
      map_names = '\n'.join('#%s   "%s"  (%d MB)' % (v.id_string, v.name, size / (1024*1024))
                            for v,size in zip(saved_maps, map_sizes))
      msg = 'Ran out of memory trying to save a session including %d maps (%.1f Gbytes).\n\nTo save the session you will either have to save the maps to separate files (then only the path to the file is included in the session) or close some of the maps.\n\n%s' % (len(saved_maps), map_gbytes, map_names)
      from chimerax.core.errors import UserError
      raise UserError(msg) from mem_error
    return data

  @staticmethod
  def restore_snapshot(session, data):
    # Grid cache used for opening HDF5 files that can contain hundreds of maps in a series.
    if not hasattr(session, '_grid_restore_cache'):
      session._grid_restore_cache = {}
      # Remove cache when session restore ends
      def remove_grid_cache(trigger_name, session):
        delattr(session, '_grid_restore_cache')
        from chimerax.core.triggerset import DEREGISTER
        return DEREGISTER
      session.triggers.add_handler('end restore session', remove_grid_cache)
    gdcache = session._grid_restore_cache        # (path, grid_id) -> GridData object
    rfp = getattr(session, '_map_replacement_file_paths', None)
    if rfp is None:
      session._map_replacement_file_paths = rfp = ReplacementFilePaths(session.ui)
    grids = grid_data_from_state(data, gdcache, session, rfp)

    return GridDataState(grids[0] if grids else None)

# ---------------------------------------------------------------------------
#
class ReplacementFilePaths:
  def __init__(self, ui):
    self._ui = ui
    self._replaced_paths = {}
    self._replace_dirs = {}
  def find(self, path, ask = False, replace_dir = True):
    replacements = self._replaced_paths
    from os.path import isfile
    if isfile(path):
      return path
    elif path in replacements:
      return replacements[path]
    elif replace_dir:
      for pf, pt in self._replace_dirs.items():
        if path.startswith(pf):
          p = pt + path[len(pf):]
          if isfile(p):
            return p
    if not ask:
      return None
    elif self._ui.is_gui:
      # If path doesn't exist show file dialog to let user enter new path to file.
      from chimerax.ui.open_save import OpenDialogWithMessage
      d = OpenDialogWithMessage(self._ui.main_window,
                                message = 'Replace missing file %s' % path,
                                caption = 'Replace missing file',
                                starting_directory = existing_directory(path))
      p = d.get_path()
      if p is None:
        return None
      replacements[path] = p
      if replace_dir:
        from os.path import basename, dirname
        # Remove the right part of path that stays the same.
        # This find directory replacements that work for data trees, like DICOM files or Tiff stacks.
        dorig, dnew = path, p
        while dnew and basename(dnew) == basename(dorig):
          dorig, dnew = dirname(dorig), dirname(dnew)
        self._replace_dirs[dorig] = dnew
      return p
    else:
      return path
  def find_multiple(self, paths, ask = False, replace_dir = True):
    # If user does not replace a path then don't ask about more paths.
    # This is to handle image stacks where a map uses hundreds of 2d files.
    npaths = []
    for p in paths:
      np = self.find(p, ask=ask, replace_dir=replace_dir)
      if np is None:
        return ()
      npaths.append(np)
    return tuple(npaths)

# ---------------------------------------------------------------------------
#
def existing_directory(path):
  from os.path import dirname, isdir
  d = path
  while d:
    if isdir(d):
      return d
    parent = dirname(d)
    if parent == d:
      break
    d = parent

  from os import getcwd
  return getcwd()

# ---------------------------------------------------------------------------
#
def state_from_grid_data(data, session_path = None, include_maps = False):
    
  dt = data
  relpath = relative_path(dt.path, session_path)
  s = {'path': relpath,
       'file_type': dt.file_type,
       'name': dt.name,
       'version': 1,
     }

  if not dt.path or include_maps:
    s['size'] = dt.size
    s['value_type'] = str(dt.value_type)
    compress_maps = False  # No advantage since session is compressed.  Ticket #4002
    bytes = dt.matrix().tobytes()

    MAX_MSGPACK_OBJECT_SIZE = 2**32-1
    if len(bytes) > MAX_MSGPACK_OBJECT_SIZE:
      from chimerax.core.errors import UserError
      raise UserError('ChimeraX session files cannot include maps over 4 Gbytes in size.\n\n' +
                      'You tried to save map "%s"' % dt.name +
                      ', size %d,%d,%d' % tuple(dt.size) +
                      ', type %s.  ' % str(dt.value_type) +
                      'Instead save the map in a map file (e.g. *.mrc, *.cmap) ' +
                      'then save the session file and it will reference the ' +
                      'map file instead of including the map data in the session file.')
      
    if compress_maps:
      from gzip import compress
      s['array_compression'] = 'gzip'
      s['array'] = compress(bytes)
    else:
      s['array_compression'] = 'none'
      s['array'] = bytes
    save_position = True
  else:
    save_position = False

  if hasattr(dt, 'database_fetch'):
    s['database_fetch'] = dt.database_fetch
  if dt.grid_id != '':
    s['grid_id'] = dt.grid_id
  if dt.step != dt.original_step or save_position:
    s['xyz_step'] = dt.step   # Use step values from data file
  if dt.origin != dt.original_origin or save_position:
    s['xyz_origin'] = dt.origin
  if dt.cell_angles != (90,90,90) or save_position:
    s['cell_angles'] = dt.cell_angles
  if dt.rotation != ((1,0,0),(0,1,0),(0,0,1)) or save_position:
    s['rotation'] = dt.rotation
  if dt.symmetries is not None and len(dt.symmetries) > 0:
    s['symmetries'] = dt.symmetries
  if hasattr(dt, 'series_index'):
    s['series_index'] = dt.series_index
  if hasattr(dt, 'channel') and dt.channel is not None:
    s['channel'] = dt.channel
  if hasattr(dt, 'time') and dt.time is not None:
    s['time'] = dt.time

  from chimerax.map_data import SubsampledGrid
  if isinstance(dt, SubsampledGrid):
    s['available_subsamplings'] = ass = {}
    for csize, ssdata in dt.available_subsamplings.items():
      if ssdata.path != dt.path:
        ass[csize] = state_from_grid_data(ssdata, session_path)

  return s

# ---------------------------------------------------------------------------
#
def grid_data_from_state(s, gdcache, session, file_paths):

  if 'array' in s:
    compression = s.get('array_compression')
    if compression == 'none':
      bytes = s['array']
    else:
      from gzip import decompress
      if compression == 'gzip':
        bytes = decompress(s['array'])
      else:
        # Older sessions without array_compression attribute used gzip and base64 encoding.
        from base64 import b64decode
        bytes = decompress(b64decode(s['array']))
    from numpy import frombuffer, dtype
    a = frombuffer(bytes, dtype = dtype(s['value_type']))
    if not a.flags.writeable:
      a = a.copy()
    array = a.reshape(s['size'][::-1])
    from chimerax.map_data import ArrayGridData
    dlist = [ArrayGridData(array)]
  else:
    dbfetch = s.get('database_fetch')
    ask = (dbfetch is None)
    if s.get('series_index',0) >= 1 or s.get('time',0) >= 1:
      ask = False
    path = absolute_path(s['path'], file_paths, ask = ask,
                         base_path = session.session_file_path)
    empty_path = (path is None or path == '' or path == () or path == [])
    if empty_path and dbfetch is None:
      return None
    else:
      gid = s.get('grid_id','')
      file_type = s['file_type']
      dlist = open_data(path, gid, file_type, dbfetch, gdcache, session)

  for data in dlist:
    data.name = s['name']

  if 'xyz_step' in s:
    for data in dlist:
      data.set_step(s['xyz_step'])

  if 'xyz_origin' in s:
    for data in dlist:
      data.set_origin(s['xyz_origin'])

  if 'cell_angles' in s:
    for data in dlist:
      data.set_cell_angles(s['cell_angles'])

  if 'rotation' in s:
    for data in dlist:
      data.set_rotation(s['rotation'])

  if 'symmetries' in s:
    for data in dlist:
      data.symmetries = s['symmetries']

  if 'series_index' in s:
    for data in dlist:
      data.series_index = s['series_index']

  if 'channel' in s:
    for data in dlist:
      data.channel = s['channel']

  if 'time' in s:
    for data in dlist:
      data.time = s['time']

  if 'available_subsamplings' in s:
    # Subsamples may be from separate files or the same file.
    from chimerax.map_data import SubsampledGrid
    dslist = []
    for data in dlist:
      if not isinstance(data, SubsampledGrid):
        data = SubsampledGrid(data)
      dslist.append(data)
    dlist = dslist
    for cell_size, dstate in s['available_subsamplings'].items():
      dpath = absolute_path(dstate['path'], file_paths, base_path = session.session_file_path)
      if dpath != path:
        ssdata = grid_data_from_state(dstate, gdcache, session, file_paths)
        if len(ssdata) == 1:
          for data in dlist:
            data.add_subsamples(ssdata[0], cell_size)
        elif len(ssdata) > 1:
          raise ValueError('session restore of volume subsamples returned more than one subsample data set: %s' % dstate)

  return dlist

# ---------------------------------------------------------------------------
#
def open_data(path, gid, file_type, dbfetch, gdcache, session):

  if (path, gid) in gdcache:
    # Caution: If data objects for the same file array can have different
    #          coordinates then cannot use this cached object.
    dlist = [gdcache[(path, gid)]]
    return dlist

  if dbfetch is None:
    from chimerax.map_data import opendialog
    paths_and_types = [(path, file_type)]
    grids, error_message = opendialog.open_grid_files(paths_and_types,
                                                      stack_images = False,
                                                      log = session.logger)
    grids = _flatten_nested_lists(grids)
    if error_message:
      print ('Error opening map "%s": %s' % (path, error_message))
      msg = error_message
# TODO: Show file dialog to locate map file.
#      from chimera import tkgui
#      grids = opendialog.select_grids(tkgui.app, 'Replace File', msg)
#      if grids is None:
#        grids = []
#      for data in grids:
#        gdcache[(path, gid)] = data # Cache using old path.

    from .volume import set_data_cache
    for data in grids:
      set_data_cache(data, session)
      gdcache[(data.path, data.grid_id)] = data
  else:
    dbid, dbn = dbfetch
    from chimerax.core.files import fetch
    mlist, status = fetch.fetch_from_database(dbid, dbn, session)
    grids = [m.data for m in mlist]
    for m in mlist:
      m.delete()        # Only use grid data from fetch
    for data in grids:
      gdcache[(dbid, dbn, data.grid_id)] = data
      gdcache[(path, data.grid_id)] = data
      data.database_fetch = dbfetch

  if (path,gid) not in gdcache:
      return []
  data = gdcache[(path, gid)]
  del gdcache[(path,gid)]	# Only use grid for one Volume
  dlist = [data]

  return dlist

def _flatten_nested_lists(l):
  fl = []
  for e in l:
    if isinstance(e, (list, tuple)):
      fl.extend(_flatten_nested_lists(e))
    else:
      fl.append(e)
  return fl

# ---------------------------------------------------------------------------
# Used for scene restore on existing grid data object.
#
def set_grid_data_attributes(s, data):

  data.set_step(s['xyz_step'] if 'xyz_step' in s else data.original_step)
  data.set_origin(s['xyz_origin'] if 'xyz_origin' in s else data.original_origin)
  if 'cell_angles' in s:
    data.set_cell_angles(s['cell_angles'])
  if 'rotation' in s:
    data.set_rotation(s['rotation'])

# -----------------------------------------------------------------------------
#
map_attributes = (
  'id',
  'place',
  'region',
  'rendering_options',
  'region_list',
  'image_levels',
  'image_colors',
  'image_brightness_factor',
  'transparency_depth',
  'default_rgba',
  'session_volume_id',
  'version',
)

basic_map_attributes = (
  'id', 'display', 'region',
  'image_levels', 'image_colors', 'image_brightness_factor',
  'transparency_depth', 'default_rgba')

renamed_attributes = (('solid_levels', 'image_levels'),
                      ('solid_colors', 'image_colors'),
                      ('solid_brightness_factor', 'image_brightness_factor'))

# ---------------------------------------------------------------------------
#
def state_from_map(volume):

  v = volume
  s = dict((attr, getattr(v, attr)) for attr in basic_map_attributes)

  s['place'] = v.position.matrix
  s['rendering_options'] = state_from_rendering_options(v.rendering_options)
  s['region_list'] = state_from_region_list(v.region_list)
  s['session_volume_id'] = session_volume_id(v)
  s['version'] = 2
  from chimerax.map_series import MapSeries
  if isinstance(v.parent, MapSeries):
    s['in_map_series'] = True
  return s

# ---------------------------------------------------------------------------
#
def create_map_from_state(s, data, session):

  ro = rendering_options_from_state(s['rendering_options'])
  from .volume import Volume
  v = Volume(session, data[0], s['region'], ro)
  v.session_volume_id = s['session_volume_id']

  if isinstance(v.data.path, str):
    v.openedAs = (v.data.path, v.data.file_type, None, False)

  set_map_state(s, v, notify = False)

  return v

# ---------------------------------------------------------------------------
# Used for scene restore on existing volumes.
#
def set_map_state(s, volume, notify = True):

  v = volume

  v.rendering_options = rendering_options_from_state(s['rendering_options'])

  if not 'display' in s:
     # Fix old session files
    s['display'] = s['displayed'] if 'displayed' in s else True
    
  for attr in basic_map_attributes:
    if attr in s:
      setattr(v, attr, s[attr])

  for old_attr, new_attr in renamed_attributes:
    if old_attr in s:
      setattr(v, new_attr, s[old_attr])

  if 'representation' in s:
    # Handle old session files that had representation attribute.
    style = s['representation']
    if style == 'solid':
      style = 'image'
    v.set_display_style(style)
      
  from chimerax.geometry import Place
  v.position = Place(s['place'])

  v.new_region(*s['region'], adjust_step = False)

  if 'region_list' in s:
    region_list_from_state(s['region_list'], v.region_list)

  if s['version'] == 1:
    for lev, color in zip(s['surface_levels'], s['surface_colors']):
      v.add_surface(lev, rgba = color)
      
#  dsize = [a*b for a,b in zip(v.data.step, v.data.size)]
#  v.transparency_depth /= min(dsize)

  if notify:
    v.call_change_callbacks(('display style changed',
                             'region changed',
                             'thresholds changed',
                             'displayed',
                             'colors changed',
                             'rendering options changed',
                             'coordinates changed'))

# -----------------------------------------------------------------------------
#
region_list_attributes = (
  'region_list',
  'current_index',
  'named_regions',
)

# ---------------------------------------------------------------------------
#
def state_from_region_list(region_list):

  s = dict((attr, getattr(region_list, attr))
           for attr in region_list_attributes)
  s['version'] = 1
  return s

# ---------------------------------------------------------------------------
#
def region_list_from_state(s, region_list):

  for attr in region_list_attributes:
    if attr in s and attr != 'version':
      setattr(region_list, attr, s[attr])

# -----------------------------------------------------------------------------
#
rendering_options_attributes = (
  'show_outline_box',
  'outline_box_rgb',
  'outline_box_linewidth',
  'limit_voxel_count',
  'voxel_limit',
  'color_mode',
  'colormap_on_gpu',
  'colormap_size',
  'colormap_extend_left',
  'colormap_extend_right',
  'blend_on_gpu',
  'projection_mode',
  'plane_spacing',
  'full_region_on_gpu',
  'dim_transparent_voxels',
  'bt_correction',
  'minimal_texture_memory',
  'maximum_intensity_projection',
  'linear_interpolation',
  'dim_transparency',
  'line_thickness',
  'smooth_lines',
  'mesh_lighting',
  'two_sided_lighting',
  'flip_normals',
  'subdivide_surface',
  'subdivision_levels',
  'surface_smoothing',
  'smoothing_factor',
  'smoothing_iterations',
  'square_mesh',
  'cap_faces',
  'orthoplanes_shown',
  'orthoplane_positions',
  'tilted_slab_axis',
  'tilted_slab_offset',
  'tilted_slab_spacing',
  'tilted_slab_plane_count',
  'image_mode',
  'backing_color',
)

# ---------------------------------------------------------------------------
#
def state_from_rendering_options(rendering_options):

  s = dict((attr,getattr(rendering_options, attr))
           for attr in rendering_options_attributes)
  s['version'] = 2
  return s

# ---------------------------------------------------------------------------
#
def rendering_options_from_state(s):

  from .volume import RenderingOptions
  ro = RenderingOptions()
  for attr in rendering_options_attributes:
    if attr in s and attr != 'version':
      setattr(ro, attr, s[attr])

  # Handle old session file box_faces attribute.
  if s['version'] == 1:
    if s.get('box_faces', False):
      ro.image_mode = 'box faces'
    elif s['orthoplanes_shown'] != (False, False, False):
      ro.image_mode = 'orthoplanes'
    
  return ro
