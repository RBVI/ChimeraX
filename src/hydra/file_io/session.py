# -----------------------------------------------------------------------------
#
def save_session(path, viewer):
  '''
  Save a session.
  '''
  s = session_state(viewer, path)
  f = open(path, 'w')
  from ..file_io.SessionUtil import objecttree
  objecttree.write_basic_tree(s, f)
  f.close()

  from .history import history
  history.add_entry(path, viewer, replace_image = True)

# -----------------------------------------------------------------------------
#
def restore_session(path, viewer):
  '''
  Restore a session.
  '''
  f = open(path, 'r')
  s = f.read()
  f.close()
  import ast
  d = ast.literal_eval(s)
  viewer.close_all_models()

  set_session_state(d, viewer)

  from .history import history
  history.add_entry(path, viewer, wait_for_render = True)

# -----------------------------------------------------------------------------
#
def session_state(viewer, rel_path = None, attributes_only = False):
  '''
  Create a dictionary representing a session including molecule models,
  surface models, density map models and scenes.  This dictionary is written
  to a Python file as a session.  It contains only basic Python types:
  numbers, strings, booleans, tuples, lists, dictionaries.
  '''
  s = {'version': 2,
       'view': view_state(viewer),
  }

  from ..map import session
  vs = session.map_states(rel_path)
  if vs:
    s['volumes'] = vs

  mlist = viewer.molecules()
  if mlist:
    from ..molecule import mol_session
    s['molecules'] = tuple(mol_session.molecule_state(m) for m in mlist)

  from .read_stl import STL_Surface
  slist = tuple(m.session_state() for m in viewer.models
                if isinstance(m, STL_Surface))
  if slist:
    s['stl surfaces'] = slist

  if not attributes_only:
    from .. import scenes
    ss = scenes.scene_state()
    if ss:
      s['scenes'] = ss

  return s

# -----------------------------------------------------------------------------
#
def set_session_state(s, viewer, attributes_only = False):

  if 'view' in s:
    restore_view(s['view'], viewer)

  if 'volumes' in s:
    from ..map import session
    session.restore_maps(s['volumes'], viewer, attributes_only)

  if 'molecules' in s:
    from ..molecule import mol_session
    mol_session.restore_molecules(s['molecules'], viewer, attributes_only)

  if 'stl surfaces' in s:
    from . import read_stl
    read_stl.restore_stl_surfaces(s['stl surfaces'], viewer, attributes_only)

  if not attributes_only:
    scene_states = s.get('scenes', [])
    from .. import scenes
    scenes.restore_scenes(scene_states, viewer)

# -----------------------------------------------------------------------------
#
def scene_state(viewer):
  return session_state(viewer, attributes_only = True)
def restore_scene(s, viewer):
  set_session_state(s, viewer, attributes_only = True)

# -----------------------------------------------------------------------------
#
view_parameters = (
  'camera_view',
  'field_of_view',
  'center_of_rotation',
  'near_far_clip',
  'window_size',
  'background_color',
  'key_light_position',
  'key_light_diffuse_color',
  'key_light_specular_color',
  'key_light_specular_exponent',
  'fill_light_position',
  'fill_light_diffuse_color',
  'ambient_light_color',
  )

# -----------------------------------------------------------------------------
#
def view_state(viewer):
  v = dict((name,getattr(viewer,name)) for name in view_parameters)
  v['camera_view'] = viewer.camera_view.matrix
  return v

# -----------------------------------------------------------------------------
#
def restore_view(vs, viewer):
  exclude = set(('window_size', 'camera_view'))
  for name in view_parameters:
    if name in vs and not name in exclude:
      setattr(viewer, name, vs[name])
  from ..geometry.place import Place
  cv = Place(vs['camera_view'])
  viewer.set_camera_view(cv)    # Set cached inverse matrix

  return True
