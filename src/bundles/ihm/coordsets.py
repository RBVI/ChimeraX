# === UCSF ChimeraX Copyright ===
# Copyright 2016 Regents of the University of California.
# All rights reserved.  This software provided pursuant to a
# license agreement containing restrictions on its disclosure,
# duplication and use.  For details see:
# http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
# This notice must be embedded in or attached to all copies,
# including partial copies, of the software or any revisions
# or derivations thereof.
# === UCSF ChimeraX Copyright ===

# Read atomic structure coordinate sets from a binary file that is 32-bit float xyz data.
def read_coordinate_sets(path, mol):
    f = open(path, 'rb')
    na = mol.num_atoms
    cs_ids = set(mol.coordset_ids)
    id = 0	# Replaces the initial coordinate set.
    while True:
        coords = f.read(12 * na)
        if not coords:
            break
        if len(coords) < 12*na:
            raise ValueError('File %s ended with a partial coordinate set, %d bytes'
                             % (path, len(coords)))
        from numpy import frombuffer, float32, float64
        xyz = frombuffer(coords, float32).reshape((na,3))
        if id in cs_ids:
            replace_coordset(mol, id, xyz)
        else:
            mol.add_coordset(id, xyz.astype(float64))
        id += 1
    f.close()

def replace_coordset(mol, id, xyz):
    cid = mol.active_coordset_id
    mol.active_coordset_id = id
    from numpy import float64
    mol.atoms.coords = xyz.astype(float64)
    mol.active_coordset_id = cid

def write_coordinate_sets(path, mols):
    f = open(path, 'wb')
    for mol in mols:
        cs = mol.active_coordset_id
        atoms = mol.atoms
        from numpy import float32
        for i in mol.coordset_ids:
            mol.active_coordset_id = i
            xyz = atoms.coords.astype(float32)
            f.write(xyz)
        mol.active_coordset_id = cs
    f.close()
