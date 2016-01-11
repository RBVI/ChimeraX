# -----------------------------------------------------------------------------
# Save and restore volume viewer state.
#
def map_states(session):
  d2v = {}
  for v in session.maps():
    d2v.setdefault(v.data,[]).append(v)
  s = state_from_maps(d2v)
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
def create_maps_from_state(dms, session, file_paths):

  # Cache of Grid_Data objects improves load speed when multiple regions
  # are using same data file.  Especially important for files that contain
  # many data arrays.
  gdcache = {}        # (path, grid_id) -> Grid_Data object
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
def absolute_path(path, file_paths, ask = False):

  from os.path import abspath
  if isinstance(path, (tuple, list)):
    apath = tuple(file_paths.find(p,ask) for p in path)
    apath = tuple(abspath(p) for p in apath if p)
  else:
    apath = file_paths.find(path,ask)
    if not apath is None:
      apath = abspath(apath)
  return apath

# ---------------------------------------------------------------------------
# Get ChimeraX unique GridDataState object for a grid.
#
def grid_data_state(grid_data, session):
  gs = getattr(session, '_volume_grid_data_session_states', None)
  if gs is None:
    session._volume_grid_data_session_states = gs = {}

  if len(gs) == 0:
    # Clear dictionary at the end of session restore.
    session.triggers.add_handler("end save session", lambda t,td,gs=gs: gs.clear())

  gds = gs.get(grid_data, None)
  if gds is None:
    gs[grid_data] = gds = GridDataState(grid_data)
  return gds

# ---------------------------------------------------------------------------
# Encapsulate Grid_Data state for session saving in ChimeraX.
#
from ..state import State
class GridDataState(State):

  def __init__(self, grid_data):
    self.grid_data = grid_data

  # State save/restore in ChimeraX
  def take_snapshot(self, session, flags):
    from ..state import CORE_STATE_VERSION
    data = state_from_grid_data(self.grid_data)
    return CORE_STATE_VERSION, data

  def restore_snapshot_init(self, session, tool_info, version, data):
    gdcache = {}        # (path, grid_id) -> Grid_Data object
    class FilePaths:
      def find(self, path, ask = False):
        # TODO: If path doesn't exist show file dialog to let user enter new path to file.
        return path
    grids = grid_data_from_state(data, gdcache, session, FilePaths())
    GridDataState.__init__(self, grids[0])

  def reset_state(self, session):
    pass

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
def grid_data_from_state(s, gdcache, session, file_paths):

  dbfetch = s.get('database_fetch')
  path = absolute_path(s['path'], file_paths, ask = (dbfetch is None))
  if path is None and dbfetch is None:
    return None

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
      if absolute_path(dstate.path, file_paths) != path:
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
    from ..files import fetch
    mlist, status = fetch.fetch_from_database(dbid, dbn, session)
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
  'id', 'display', 'region', 'representation',
  'surface_levels', 'surface_colors',
  'surface_brightness_factor', 'transparency_factor',
  'solid_levels', 'solid_colors', 'solid_brightness_factor',
  'transparency_depth', 'default_rgba')

# ---------------------------------------------------------------------------
#
def state_from_map(volume):

  v = volume
  s = dict((attr, getattr(v, attr)) for attr in basic_map_attributes)

  s['place'] = v.position.matrix
  s['rendering_options'] = state_from_rendering_options(v.rendering_options)
  s['region_list'] = state_from_region_list(v.region_list)
  s['session_volume_id'] = session_volume_id(v)
  s['version'] = 1
  if hasattr(v, 'parent'):
    from .series import Map_Series
    if isinstance(v.parent, Map_Series):
      s['in_map_series'] = True
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

  from ..geometry import Place
  v.position = Place(s['place'])

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

  d = v.display
  if d:
    v.show()
  else:
    if not s.get('in_map_series',False):
      v.show()      # Compute surface even if not displayed so that turning on display
                    # for example with model panel that only sets display to true shows surface.
    v.display = False

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
