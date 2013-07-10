def save_session(path, viewer):
  f = open(path, 'w')
  f.write('{\n')
  f.write("'version':2,\n")
  save_view(f, viewer)
  save_maps(f, viewer)
  save_molecules(f, viewer)
  from . import readstl
  readstl.save_stl_surfaces(f, viewer)
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
  restore_maps(d, viewer)
  restore_molecules(d, viewer)
  from . import readstl
  readstl.restore_stl_surfaces(d, viewer)
  
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
  file.write("'view':\n")
  from .SessionUtil import objecttree
  objecttree.write_basic_tree(v, file, indent = ' ')
  file.write(',\n')

def restore_view(d, viewer):
  vars = d.get('view')
  if vars is None:
    return False
  for name in view_parameters:
    if name in vars and name != 'window_size':
      setattr(viewer, name, vars[name])
  viewer.set_camera_view(viewer.camera_view)    # Set cached inverse matrix

  return True

def save_maps(f, viewer):
  from .VolumeViewer import session
  s = session.Volume_Manager_State()
  from .VolumeViewer.volume import volume_manager
  s.state_from_manager(volume_manager)
  from os.path import dirname
  directory = dirname(f.name)
  if directory:
      s.use_relative_paths(directory)
  from .SessionUtil import objecttree
  t = objecttree.instance_tree_to_basic_tree(s)
  f.write("'volume_data_state':\n")
  objecttree.write_basic_tree(t, f, indent = ' ')
  f.write(',\n')

def restore_maps(d, viewer):
  vds = d.get('volume_data_state')
  if vds is None:
    return False
  from .VolumeViewer import session
  session.restore_volume_data_state(vds)
  from .VolumeViewer.volume import volume_manager
  for m in volume_manager.data_regions:
    viewer.add_model(m)
  return True

def save_molecules(f, viewer):
  mstate = tuple({'path':m.path, 'place':m.place, 'copies':m.copies}
                 for m in viewer.molecules())
  f.write("'molecules':(\n")
  from .SessionUtil import objecttree
  for ms in mstate:
    objecttree.write_basic_tree(ms, f, indent = ' ')
    f.write(',\n')
  f.write('),\n')

def restore_molecules(d, viewer):
  mstate = d.get('molecules')
  if mstate is None:
    return False
  from .pdb import open_pdb_file, open_mmcif_file
  for ms in mstate:
    p = ms['path']
    if p.endswith('.cif'):
      m = open_mmcif_file(p)
    else:
      m = open_pdb_file(p)
    m.place = ms['place']
    m.copies = ms.get('copies', [])
    viewer.add_model(m)
  return True
