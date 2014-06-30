#use_pdbio = True
use_pdbio = False

def open_pdb_file(path, session):
  '''
  Open a PDB file.
  '''
  if use_pdbio:
    mols = open_pdb_file_with_pdbio(path)
  else:
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
  matoms = _image3d.parse_pdb_file(text)
  t1 = time()
  mols = []
  from ..molecule import Molecule, connect
  for a in matoms:
    atoms = atom_array(a)
    m = Molecule(path, atoms)
    m.pdb_text = text
    m.color_by_chain()
    bonds, missing = connect.molecule_bonds(m, session)
    m.bonds = bonds
    mols.append(m)
  t3 = time()
#  from os.path import basename
#  print ('image3d', basename(path), 'read time', '%.3f' % (ft1-t0), 'atoms', len(xyz), 'atoms/sec', int(len(xyz)/(ft1-t0)))
#  t = (t1-t0)+(t3-t2)
#  print('image3d', basename(path), 'read+parse time', '%.3f' % t, 'atoms', len(xyz), 'atoms/sec', int(len(xyz)/t))
#  print(basename(path), len(atoms), 'atoms')

  return mols

# Convert numpy byte array of C Atom structure to a numpy structured array.
def atom_array(a):
  from ..molecule import atom_dtype
  atoms = a.view(atom_dtype).reshape((len(a),))
  atoms['atom_shown'] = True
  set_atom_radii(atoms)
  sort_atoms(atoms)
  return atoms

def sort_atoms(atoms, bonds = None):
  '''Sort numpy structured array of atoms by chain id and residue number. Update bonds to use new order.'''
  satoms = atoms.view('S%d'%atoms.itemsize)     # Need string array for C++ sort routine.
  from .. import _image3d
  order = _image3d.atom_sort_order(satoms)
  satoms[:] = satoms[order]
  if not bonds is None:
    from numpy import empty, int32, arange
    n = len(order)
    amap = empty((n,),int32)
    amap[order] = arange(n)
    bonds[:,0] = amap[bonds[:,0]]
    bonds[:,1] = amap[bonds[:,1]]

def set_atom_radii(atoms):
  from .. import _image3d
  atoms['radius'][:] = _image3d.element_radii(atoms['element_number'])

def open_pdb_file_with_pdbio(path):
  from time import time
  t0 = time()
  f = open(path, 'r')
  from .. import pdbio
  sblob = pdbio.read_pdb_file(f)
  f.close()
  t1 = time()
#  print('pdbio', path, 'read+parse time', '%.3f' % (t1-t0), 'atoms', len(atoms), 'atoms/sec', int(len(atoms)/(t1-t0)))

  mols = []
  from ..molecule import Molecule
  for sb in sblob.structures:
    atoms, bonds = structblob_atoms_and_bonds(sb)
    m = Molecule(path, atoms)
    m.bonds = bonds
    m.color_by_chain()
    mols.append(m)

  return mols

def structblob_atoms_and_bonds(sblob):

  sblobs = sblob.structures
  if len(sblobs) > 1:
    sblob = sblobs[0]   # Take only the first structure in NMR ensembles

  ablob = sblob.atoms
  xyz = ablob.coords
  n = len(xyz)
  enums = ablob.element_numbers
  anames = ablob.names
  rblob = ablob.residues
  cids = rblob.chain_ids
  rnames = rblob.names
  rnums = rblob.numbers
  from numpy import int32, empty, zeros
  bonds = sblob.atoms_bonds[1].astype(int32)    # Getting uint64

  from ..molecule import atom_dtype
  atoms = zeros((n,), atom_dtype)
  atoms['atom_name'][:] = anames
  atoms['element_number'][:] = enums
  atoms['xyz'][:] = xyz
  atoms['chain_id'][:] = cids
  atoms['residue_number'][:] = rnums
  atoms['residue_name'][:] = rnames
  atoms['atom_shown'] = True
  set_atom_radii(atoms)

  sort_atoms(atoms, bonds)

  return atoms, bonds

def load_pdb_local(id, session, pdb_dir = '/usr/local/pdb'):
    '''Load a PDB file given its id from a local copy of the "divided" database.'''
    from os.path import join, exists
    p = join(pdb_dir, id[1:3].lower(), 'pdb%s.ent' % id.lower())
    if not exists(p):
      return None
    mols = open_pdb_file(p, session)
    return mols
