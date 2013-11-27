def open_pdb_file(path):
  '''
  Open a PDB file.
  '''
#  mols = open_pdb_file_with_pdbio(path)
  mols = open_pdb_file_with_image3d(path)
  return mols

def open_pdb_file_with_image3d(path):
  from time import time
  t0 = time()
  f = open(path, 'rb')
  text = f.read()
  f.close()
  ft1 = time()
  from .. import _image3d
  xyz, element_nums, chain_ids, res_nums, res_names, atom_names = \
      _image3d.parse_pdb_file(text)
  t1 = time()
  from ..molecule import Molecule
  m = Molecule(path, xyz, element_nums, chain_ids, res_nums, res_names, atom_names)
  m.pdb_text = text
  from ..molecule import connect
  t2 = time()
  bonds, missing = connect.molecule_bonds(m)
  m.bonds = bonds
  t3 = time()
  from os.path import basename
  print ('image3d', basename(path), 'read time', '%.3f' % (ft1-t0), 'atoms', len(xyz), 'atoms/sec', int(len(xyz)/(ft1-t0)))
  t = (t1-t0)+(t3-t2)
  print('image3d', basename(path), 'read+parse time', '%.3f' % t, 'atoms', len(xyz), 'atoms/sec', int(len(xyz)/t))
#  from ..molecule import connect
#  t2 = time()
#  connect.create_molecule_bonds(m)
#  t3 = time()
#  print ('bonds', len(m.bonds), '%.3f' % (t3-t2))

#  print('xyz', xyz.shape, xyz.dtype, xyz[:5])
#  print('enums', element_nums.shape, element_nums.dtype, element_nums[:5])
#  print('cids', chain_ids.shape, chain_ids.dtype, chain_ids[:5])
#  print('rnums', res_nums.shape, res_nums.dtype, res_nums[:5])
#  print('rnames', res_names.shape, res_names.dtype, res_names[:5])
#  print('anames', atom_names.shape, atom_names.dtype, atom_names[:5])
  return m

def open_pdb_file_with_pdbio(path):
  from time import time
  t0 = time()
  f = open(path, 'r')
  from .. import pdbio, access
  molcaps = pdbio.read_pdb_file(f)
  f.close()
  from ..molecule import Molecule
  mols = []
  mclist = access.molecules(molcaps)
  for mc in mclist:
    atoms = access.atoms(mc)
    xyz = access.coords(atoms)
    element_nums = access.element_numbers(atoms)
    res = access.atom_residues(atoms)
    chain_ids = access.residue_chain_ids(res, numpy = True)
    res_nums = access.residue_numbers(res)
    res_names = access.residue_names(res, numpy = True)
    atom_names = access.atom_names(atoms, numpy = True)
    t1 = time()
    from os.path import basename
    print('pdbio', basename(path), 'read+parse time', '%.3f' % (t1-t0), 'atoms', len(xyz), 'atoms/sec', int(len(xyz)/(t1-t0)))
#    print('xyz', xyz.shape, xyz.dtype, xyz[:5])
#    print('enums', element_nums.shape, element_nums.dtype, element_nums[:5])
#    print('cids', chain_ids.shape, chain_ids.dtype, chain_ids[:5])
#    print('rnums', res_nums.shape, res_nums.dtype, res_nums[:5])
#   print('rnames', res_names.shape, res_names.dtype, res_names[:5])
#    print('anames', atom_names.shape, atom_names.dtype, atom_names[:5])
    from numpy import float32, array, int32
    xyz = xyz.astype(float32)
    m = Molecule(path, xyz, element_nums, chain_ids, res_nums, res_names, atom_names)
    atoms, bonds = access.atoms_bonds(mc)
#    print ('bonds', len(bonds))
    m.bonds = array(bonds, int32)
    mols.append(m)
  return mols

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
  from ..molecule import Molecule
  m = Molecule(path, points, elements, chain_ids)
  m.pdb_text = text
  return m

def open_mmcif_file(path):
  '''
  Open an mmCIF file.
  '''
  from os.path import basename
  from time import time
  ft0 = time()
  f = open(path, 'r')
  text = f.read()
  f.close()
  ft1 = time()
  from .. import _image3d
  t0 = time()
  xyz, element_nums, chain_ids, res_nums, res_names, atom_names = \
      _image3d.parse_mmcif_file(text)
  t1 = time()
  print ('Read', basename(path), len(xyz), 'atoms at', int(len(xyz)/(ft1-ft0)), 'per second')
  print ('Parsed', basename(path), len(xyz), 'atoms at', int(len(xyz)/(t1-t0)), 'per second')
  print ('Read+Parsed', basename(path), len(xyz), 'atoms at', int(len(xyz)/((t1-t0)+(ft1-ft0))), 'per second')
  from ..molecule import Molecule
  m = Molecule(path, xyz, element_nums, chain_ids, res_nums, res_names, atom_names)
  return m
