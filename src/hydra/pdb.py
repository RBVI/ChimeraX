def open_pdb_file(path):
  f = open(path, 'r')
  text = f.read()
  f.close()
  from . import _image3d
  xyz, element_nums, chain_ids, res_nums, res_names, atom_names = \
      _image3d.parse_pdb_file(text)
#  cids = chain_ids.tostring().decode('utf-8')
  cids = chain_ids
  from .molecule import Molecule
  m = Molecule(path, xyz, element_nums, cids, res_nums, res_names, atom_names)
  m.pdb_text = text
  return m

def open_pdb_file_python(path):
  # The following was replaced by C++ for speed.
  f = open(path, 'r')
  lines = f.readlines()
  f.close()
  points = []
  elements = []
  chain_ids = []
  for line in lines:
    if line.startswith('ATOM') or line.startswith('HETATM'):
      p = float(line[30:38]),float(line[38:46]),float(line[46:54])
      points.append(p)
      el = line[76:78].strip()
      elements.append(el)
      chain_id = line[21]
      chain_ids.append(chain_id)
  from .molecule import Molecule
  m = Molecule(path, points, elements, chain_ids)
  m.pdb_text = text
  return m
