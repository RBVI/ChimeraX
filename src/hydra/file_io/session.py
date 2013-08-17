def save_session(path, viewer):
  f = open(path, 'w')
  f.write('{\n')
  f.write("'version':2,\n")
  save_view(f, viewer)
  save_maps(f, viewer)
  save_molecules(f, viewer)
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
  restore_maps(d, viewer)
  restore_molecules(d, viewer)
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

def save_maps(f, viewer):
  from ..VolumeViewer import session
  s = session.Volume_Manager_State()
  from ..VolumeViewer.volume import volume_manager
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
  from ..VolumeViewer import session
  session.restore_volume_data_state(vds)
  from ..VolumeViewer.volume import volume_manager
  for m in volume_manager.data_regions:
    viewer.add_model(m)
  return True

mol_attrs = ('path', 'show_atoms', 'atom_style',
             'color_mode', 'show_ribbons', 'ribbon_radius',
             'ball_scale')
def save_molecules(f, viewer):
  mstate = []
  for m in viewer.molecules():
    
    ms = {'place':m.place.matrix}
    for attr in mol_attrs:
      ms[attr] = getattr(m,attr)
    if m.copies:
      ms['copies'] = tuple(c.matrix for c in m.copies)
    if not m.bonds is None:
      ms['has_bonds'] = True
    mstate.append(ms)
                 
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
  from .opensave import open_files
  for ms in mstate:
    p = ms['path']
    mlist = open_files([p], set_camera = False)
    if len(mlist) != 1:
      from ..ui.gui import show_info
      show_info('File %s unexpectedly contained %d models' % (len(mlist),))
      continue
    m = mlist[0]
    from ..geometry.place import Place
    m.place = Place(ms['place'])
    m.copies = [Place(c) for c in ms.get('copies', [])]
    for attr in mol_attrs:
      if attr in ms:
        setattr(m, attr, ms[attr])
    if 'has_bonds' in ms and ms['has_bonds'] and m.bonds is None:
      from ..molecule import connect
      connect.create_molecule_bonds(m)
    viewer.add_model(m)
  return True
