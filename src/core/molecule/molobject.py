from numpy import uint8, int32, float64, float32, bool as npy_bool
from .molc import string, cptr, pyobject, c_property, set_c_pointer

# -------------------------------------------------------------------------------
# These routines convert C++ pointers to Python objects and are used for defining
# the object properties.
#
def _atomic_structure(p):
    return object_map(p, AtomicStructure)
def _residue(p):
    return object_map(p, Residue)
def _atom_pair(p):
    return (object_map(p[0],Atom), object_map(p[1],Atom))
def _pseudobond_group_map(pbgc_map):
    from .molarray import PseudoBonds
    pbg_map = dict((name, PseudoBonds(pbg)) for name, pbg in pbgc_map.items())
    return pbg_map

from .molarray import _atoms, _bonds, _residues, _chains

# -----------------------------------------------------------------------------
#
class Atom:

    def __init__(self, atom_pointer):
        set_c_pointer(self, atom_pointer)

    bfactor = c_property('atom_bfactor', float32)
    color = c_property('atom_color', uint8, 4)
    coord = c_property('atom_coord', float64, 3)
    display = c_property('atom_display', npy_bool)
    draw_mode = c_property('atom_draw_mode', int32)
    element_name = c_property('atom_element_name', string, read_only = True)
    element_number = c_property('atom_element_number', int32, read_only = True)
    molecule = c_property('atom_molecule', cptr, astype = _atomic_structure, read_only = True)
    name = c_property('atom_name', string, read_only = True)
    radius = c_property('atom_radius', float32)
    residue = c_property('atom_residue', cptr, astype = _residue, read_only = True)

# -----------------------------------------------------------------------------
#
class Bond:

    def __init__(self, bond_pointer):
        set_c_pointer(self, bond_pointer)

    atoms = c_property('bond_atoms', cptr, 2, astype = _atom_pair, read_only = True)
    color = c_property('bond_color', uint8, 4)
    display = c_property('bond_display', int32)
    halfbond = c_property('bond_halfbond', npy_bool)
    radius = c_property('bond_radius', float32)

# -----------------------------------------------------------------------------
#
class PseudoBond:

    def __init__(self, pbond_pointer):
        set_c_pointer(self, pbond_pointer)

    atoms = c_property('pseudobond_atoms', cptr, 2, astype = _atom_pair, read_only = True)
    color = c_property('pseudobond_color', uint8, 4)
    display = c_property('pseudobond_display', int32)
    halfbond = c_property('pseudobond_halfbond', npy_bool)
    radius = c_property('pseudobond_radius', float32)

# -----------------------------------------------------------------------------
#
class Residue:

    def __init__(self, residue_pointer):
        set_c_pointer(self, residue_pointer)

    atoms = c_property('residue_atoms', cptr, 'num_atoms', astype = _atoms, read_only = True)
    chain_id = c_property('residue_chain_id', string, read_only = True)
    molecule = c_property('residue_molecule', cptr, astype = _atomic_structure, read_only = True)
    name = c_property('residue_name', string, read_only = True)
    num_atoms = c_property('residue_num_atoms', int32, read_only = True)
    number = c_property('residue_number', int32, read_only = True)
    str = c_property('residue_str', string, read_only = True)
    unique_id = c_property('residue_unique_id', int32, read_only = True)
    # TODO: Currently no C++ method to get Chain

# -----------------------------------------------------------------------------
#
class Chain:

    def __init__(self, chain_pointer):
        set_c_pointer(self, chain_pointer)

    chain_id = c_property('chain_chain_id', string, read_only = True)
    molecule = c_property('chain_molecule', cptr, astype = _atomic_structure, read_only = True)
    residues = c_property('chain_residues', cptr, 'num_residues', astype = _residues, read_only = True)
    num_residues = c_property('chain_num_residues', int32, read_only = True)

# -----------------------------------------------------------------------------
#
class AtomicStructure:

    def __init__(self, mol_pointer):
        set_c_pointer(self, mol_pointer)

    atoms = c_property('molecule_atoms', cptr, 'num_atoms', astype = _atoms, read_only = True)
    bonds = c_property('molecule_bonds', cptr, 'num_bonds', astype = _bonds, read_only = True)
    chains = c_property('molecule_chains', cptr, 'num_chains', astype = _chains, read_only = True)
    name = c_property('molecule_name', string)
    num_atoms = c_property('molecule_num_atoms', int32, read_only = True)
    num_bonds = c_property('molecule_num_bonds', int32, read_only = True)
    num_chains = c_property('molecule_num_chains', int32, read_only = True)
    num_residues = c_property('molecule_num_residues', int32, read_only = True)
    residues = c_property('molecule_residues', cptr, 'num_residues', astype = _residues, read_only = True)
    pbg_map = c_property('molecule_pbg_map', pyobject, astype = _pseudobond_group_map, read_only = True)

# -----------------------------------------------------------------------------
# Return an AtomicStructure for a C++ StructBlob.
#
def atomic_structure_from_blob(struct_blob):
    return object_map(struct_blob._struct_pointers[0], AtomicStructure)

# -----------------------------------------------------------------------------
#
_object_map = {}	# Map C++ pointer to Python object
def object_map(p, object_type):
    global _object_map
    o = _object_map.get(p, None)
    if o is None:
        _object_map[p] = o = object_type(p)
    return o
