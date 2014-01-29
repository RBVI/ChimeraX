# -----------------------------------------------------------------------------
# Save and restore volume viewer state.
#
def map_states(session, rel_path = None):
  d2v = {}
  for v in session.maps():
    d2v.setdefault(v.data,[]).append(v)
  s = state_from_maps(d2v)
  if rel_path:
      use_relative_paths(s, rel_path)
  return s

# -----------------------------------------------------------------------------
#
def restore_maps(dms, session, attributes_only = False):
  if attributes_only:
    restore_map_attributes(dms, session)
  else:
    create_maps_from_state(dms, session)
  return True

# -----------------------------------------------------------------------------
# Restore map attributes for a scene.
#
def restore_map_attributes(dms, session):
  set_maps_attributes(dms, session)
  for v in session.maps():
    v.update_display()

# ---------------------------------------------------------------------------
#
def state_from_maps(data_maps, include_unsaved_volumes = False):

  dvlist = []
  unsaved_data = []
  for data, volumes in data_maps.items():
    if data.path == '' and not include_unsaved_volumes:
      unsaved_data.append(data)
      continue                # Do not save data sets with no path
    dvlist.append((state_from_grid_data(data),
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
def create_maps_from_state(dms, session):

  # Cache of Grid_Data objects improves load speed when multiple regions
  # are using same data file.  Especially important for files that contain
  # many data arrays.
  gdcache = {}        # (path, grid_id) -> Grid_Data object
  for ds, vslist in dms:
    data = grid_data_from_state(ds, gdcache, session)
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

# ---------------------------------------------------------------------------
#
def use_relative_paths(dms, directory):

  for ds, vslist in dms:
    p = ds['path']
    if p:
      ds['path'] = relative_path(p, directory)

# -----------------------------------------------------------------------------
# Path can be a tuple of paths.
#
def relative_path(path, dir):

  if isinstance(path, (tuple, list)):
    return tuple([relative_path(p, dir) for p in path])

  if not isinstance(path, str):
    return path

  from os.path import join
  d = join(dir, '')       # Make directory end with "/".
  if not path.startswith(d):
    return path

  rpath = path[len(d):]
  return rpath

# -----------------------------------------------------------------------------
# Path can be a tuple of paths.
#
def absolute_path(path):

  from os.path import abspath
  if isinstance(path, (tuple, list)):
    apath = tuple([abspath(p) for p in path])
  elif isinstance(path, str):
    apath = abspath(path)
  else:
    apath = path
  return apath

# ---------------------------------------------------------------------------
#
def state_from_grid_data(data):
    
  dt = data
  s = {'path': dt.path,
       'file_type': dt.file_type,
       'name': dt.name,
     }

  if hasattr(dt, 'database_fetch'):
    s['database_fetch'] = dt.database_fetch
  if dt.grid_id != '':
    s['grid_id'] = dt.grid_id
  if dt.step != dt.original_step:
    s['xyz_step'] = dt.step   # Use step values from data file
  if dt.origin != dt.original_origin:
    s['xyz_origin'] = dt.origin
  if dt.cell_angles != (90,90,90):
    s['cell_angles'] = dt.cell_angles
  if dt.rotation != ((1,0,0),(0,1,0),(0,0,1)):
    s['rotation'] = dt.rotation
  if len(dt.symmetries) > 0:
    s['symmetries'] = dt.symmetries

  from .data import Subsampled_Grid
  if isinstance(dt, Subsampled_Grid):
    s['available_subsamplings'] = ass = {}
    for csize, ssdata in dt.available_subsamplings.items():
      if ssdata.path != dt.path:
        ass[csize] = state_from_grid_data(ssdata)

  return s

# ---------------------------------------------------------------------------
#
def grid_data_from_state(s, gdcache, session):

  path = absolute_path(s['path'])
  gid = s.get('grid_id','')
  file_type = s['file_type']
  dbfetch = s.get('database_fetch')
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
    
  if 'available_subsamplings' in s:
    # Subsamples may be from separate files or the same file.
    from .data import Subsampled_Grid
    dslist = []
    for data in dlist:
      if not isinstance(data, Subsampled_Grid):
        data = Subsampled_Grid(data)
      dslist.append(data)
    dlist = dslist
    for cell_size, dstate in s['available_subsamplings'].items():
      if absolute_path(dstate.path) != path:
        ssdlist = dstate.create_object(gdcache)
        for i,ssdata in enumerate(ssdlist):
          dlist[i].add_subsamples(ssdata, cell_size)

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
    from .data import opendialog
    paths_and_types = [(path, file_type)]
    grids, error_message = opendialog.open_grid_files(paths_and_types,
                                                      stack_images = False)
    if error_message:
      print ('Error opening map', error_message)
      msg = error_message + '\nPlease select replacement file.'
# TODO: Show file dialog to locate map file.
#      from chimera import tkgui
#      grids = opendialog.select_grids(tkgui.app, 'Replace File', msg)
#      if grids is None:
#        grids = []
#      for data in grids:
#        gdcache[(path, gid)] = data # Cache using old path.

    for data in grids:
      gdcache[(data.path, data.grid_id)] = data
  else:
    dbid, dbn = dbfetch
    from ..file_io import fetch
    mlist = fetch.fetch_from_database(dbid, dbn, session)
    grids = [m.data for m in mlist]
    for m in mlist:
      m.delete()        # Only use grid data from fetch
    for data in grids:
      gdcache[(dbid, dbn, data.grid_id)] = data
      gdcache[(path, data.grid_id)] = data
      data.database_fetch = dbfetch

  data = gdcache.get((path, gid))
  if data is None:
      return []
  dlist = [data]

  return dlist

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
  'representation',
  'rendering_options',
  'region_list',
  'surface_levels',
  'surface_colors',
  'surface_brightness_factor',
  'transparency_factor',
  'solid_levels',
  'solid_colors',
  'solid_brightness_factor',
  'transparency_depth',
  'default_rgba',
  'session_volume_id',
  'version',
)
basic_map_attributes = (
  'id', 'displayed', 'region', 'representation',
  'surface_levels', 'surface_colors',
  'surface_brightness_factor', 'transparency_factor',
  'solid_levels', 'solid_colors', 'solid_brightness_factor',
  'transparency_depth', 'default_rgba')

# ---------------------------------------------------------------------------
#
def state_from_map(volume):

  v = volume
  s = dict((attr, getattr(v, attr)) for attr in basic_map_attributes)

  s['place'] = v.place.matrix
  s['rendering_options'] = state_from_rendering_options(v.rendering_options)
  s['region_list'] = state_from_region_list(v.region_list)
  s['session_volume_id'] = session_volume_id(v)
  s['version'] = 1
  return s

# ---------------------------------------------------------------------------
#
def create_map_from_state(s, data, session):

  ro = rendering_options_from_state(s['rendering_options'])
  from .volume import Volume
  v = Volume(data[0], session, s['region'], ro)
  v.session_volume_id = s['session_volume_id']

  if isinstance(v.data.path, str):
    v.openedAs = (v.data.path, v.data.file_type, None, False)

  set_map_state(s, v, notify = False)

  if v.displayed:
    v.show()

  return v

# ---------------------------------------------------------------------------
# Used for scene restore on existing volumes.
#
def set_map_state(s, volume, notify = True):

  v = volume

  v.rendering_options = rendering_options_from_state(s['rendering_options'])

  for attr in basic_map_attributes:
    if attr in s:
      setattr(v, attr, s[attr])

  from ..geometry.place import Place
  v.place = Place(s['place'])

  v.new_region(*s['region'], show = False, adjust_step = False)

  if 'region_list' in s:
    region_list_from_state(s['region_list'], v.region_list)

#  dsize = [a*b for a,b in zip(v.data.step, v.data.size)]
#  v.transparency_depth /= min(dsize)

  if notify:
    v.call_change_callbacks(('representation changed',
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
  'projection_mode',
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
  'box_faces',
  'orthoplanes_shown',
  'orthoplane_positions',
)

# ---------------------------------------------------------------------------
#
def state_from_rendering_options(rendering_options):

  s = dict((attr,getattr(rendering_options, attr))
           for attr in rendering_options_attributes)
  s['version'] = 1
  return s

# ---------------------------------------------------------------------------
#
def rendering_options_from_state(s):

  from .volume import Rendering_Options
  ro = Rendering_Options()
  for attr in rendering_options_attributes:
    if attr in s and attr != 'version':
      setattr(ro, attr, s[attr])
  return ro
