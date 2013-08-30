def save_session(path, viewer):
  f = open(path, 'w')
  f.write('{\n')
  f.write("'version':2,\n")
  save_view(f, viewer)
  from ..VolumeViewer import session
  session.save_maps(f, viewer)
  from ..molecule import mol_session
  mol_session.save_molecules(f, viewer.molecules())
  from . import read_stl
  read_stl.save_stl_surfaces(f, viewer)
  f.write('\n}\n')
  f.close()

  from . import history
  history.save_history(path, viewer)

def restore_session(path, viewer):
  f = open(path, 'r')
  s = f.read()
  f.close()
  import ast
  d = ast.literal_eval(s)
  viewer.close_all_models()
  restore_view(d, viewer)
  from ..VolumeViewer import session
  session.restore_maps(d, viewer)
  from ..molecule import mol_session
  mol_session.restore_molecules(d, viewer)
  from . import read_stl
  read_stl.restore_stl_surfaces(d, viewer)
  
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

def save_view(file, viewer):
  v = dict((name,getattr(viewer,name)) for name in view_parameters)
  v['camera_view'] = viewer.camera_view.matrix
  file.write("'view':\n")
  from .SessionUtil import objecttree
  objecttree.write_basic_tree(v, file, indent = ' ')
  file.write(',\n')

def restore_view(d, viewer):
  vars = d.get('view')
  if vars is None:
    return False
  exclude = set(('window_size', 'camera_view'))
  for name in view_parameters:
    if name in vars and not name in exclude:
      setattr(viewer, name, vars[name])
  from ..geometry.place import Place
  cv = Place(vars['camera_view'])
  viewer.set_camera_view(cv)    # Set cached inverse matrix

  return True
