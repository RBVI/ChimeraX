def open_pdb_file(path, session):
  '''
  Open a PDB file.
  '''
#  mols = open_pdb_file_with_pdbio(path)
  mols = open_pdb_file_with_image3d(path, session)
  return mols

def open_pdb_file_with_image3d(path, session):
  from time import time
  t0 = time()
  f = open(path, 'rb')
  text = f.read()
  f.close()
  ft1 = time()
  from .. import _image3d
  a = _image3d.parse_pdb_file(text)
  t1 = time()
  atoms = atom_array(a)
  from ..molecule import Molecule
  m = Molecule(path, atoms)
  m.pdb_text = text
  m.color_by_chain()
  from ..molecule import connect
  t2 = time()
  bonds, missing = connect.molecule_bonds(m, session)
  m.bonds = bonds
  t3 = time()
  from os.path import basename
#  print ('image3d', basename(path), 'read time', '%.3f' % (ft1-t0), 'atoms', len(xyz), 'atoms/sec', int(len(xyz)/(ft1-t0)))
  t = (t1-t0)+(t3-t2)
#  print('image3d', basename(path), 'read+parse time', '%.3f' % t, 'atoms', len(xyz), 'atoms/sec', int(len(xyz)/t))
  print(basename(path), len(atoms), 'atoms')
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

# Convert numpy byte array of C Atom structure to a numpy structured array.
def atom_array(a):
  dtype = [('atom_name', 'a4'),
           ('element_number', 'i4'),
           ('xyz', 'f4', (3,)),
           ('radius', 'f4'),
           ('residue_name', 'a4'),
           ('residue_number', 'i4'),
           ('chain_id', 'a4'),
           ('atom_color', 'u1', (4,)),
           ('ribbon_color', 'u1', (4,)),
           ('atom_shown', 'u1'),
           ('ribbon_shown', 'u1'),
           ('pad', 'u2'),               # C struct size is multiple of 4 bytes
          ]
  atoms = a.view(dtype).reshape((len(a),))
  satoms = atoms.view('S%d'%atoms.itemsize)     # Need string array for C++ sort routine.
  from .. import _image3d
  _image3d.sort_atoms_by_chain(satoms)
  atoms['atom_shown'] = True
  enums = atoms['element_number']
  atoms['radius'][:] = _image3d.element_radii(enums)

  return atoms

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

def load_pdb_local(id, session, pdb_dir = '/usr/local/pdb'):
    '''Load a PDB file given its id from a local copy of the "divided" database.'''
    from os.path import join, exists
    p = join(pdb_dir, id[1:3].lower(), 'pdb%s.ent' % id.lower())
    if not exists(p):
      return None
    m = open_pdb_file(p, session)
    return m
