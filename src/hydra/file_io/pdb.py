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
  from ..molecule import atom_dtype
  atoms = a.view(atom_dtype).reshape((len(a),))
  init_atoms(atoms)
  return atoms

def init_atoms(atoms):
  satoms = atoms.view('S%d'%atoms.itemsize)     # Need string array for C++ sort routine.
  from .. import _image3d
  _image3d.sort_atoms_by_chain(satoms)
  atoms['atom_shown'] = True
  enums = atoms['element_number']
  atoms['radius'][:] = _image3d.element_radii(enums)

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
    t1 = time()
    atoms = access.atoms(mc)
    n = len(atoms)
    from ..molecule import atom_dtype, Molecule
    atoms = zeros((n,), atom_dtype)
    atoms['atom_name'] = access.atom_names(atoms, numpy = True)
    atoms['element_number'] = access.element_numbers(atoms)
    atoms['xyz'] = access.coords(atoms)
    atoms['radius'] = xyzra[:,3]
    res = access.atom_residues(atoms)
    atoms['residue_name'] = access.residue_names(res, numpy = True)
    atoms['residue_number'] = access.residue_numbers(res)
    atoms['chain_id'] = access.residue_chain_ids(res, numpy = True)
    atoms['atom_color'] = (178,178,178,255)
    atoms['ribbon_color'] = (178,178,178,255)
    atoms['atom_shown'] = 1
    atoms['ribbon_shown'] = 0
    from os.path import basename
    # TODO: Not tested since several months ago.
    m = Molecule(basename(path), atoms)
    print('pdbio', basename(path), 'read+parse time', '%.3f' % (t1-t0), 'atoms', len(xyz), 'atoms/sec', int(len(xyz)/(t1-t0)))
#    print('xyz', xyz.shape, xyz.dtype, xyz[:5])
#    print('enums', element_nums.shape, element_nums.dtype, element_nums[:5])
#    print('cids', chain_ids.shape, chain_ids.dtype, chain_ids[:5])
#    print('rnums', res_nums.shape, res_nums.dtype, res_nums[:5])
#   print('rnames', res_names.shape, res_names.dtype, res_names[:5])
#    print('anames', atom_names.shape, atom_names.dtype, atom_names[:5])
    atoms, bonds = access.atoms_bonds(mc)
#    print ('bonds', len(bonds))
    m.bonds = array(bonds, int32)
    mols.append(m)
  return mols

def load_pdb_local(id, session, pdb_dir = '/usr/local/pdb'):
    '''Load a PDB file given its id from a local copy of the "divided" database.'''
    from os.path import join, exists
    p = join(pdb_dir, id[1:3].lower(), 'pdb%s.ent' % id.lower())
    if not exists(p):
      return None
    m = open_pdb_file(p, session)
    return m
