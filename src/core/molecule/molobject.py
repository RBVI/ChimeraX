from numpy import uint8, int32, float64, float32, bool as npy_bool
from .molc import string, cptr, pyobject, c_property, set_c_pointer, c_function
import ctypes

# -------------------------------------------------------------------------------
# These routines convert C++ pointers to Python objects and are used for defining
# the object properties.
#
def _atoms(a):
    from .molarray import Atoms
    return Atoms(a)
def _atom_pair(p):
    return (object_map(p[0],Atom), object_map(p[1],Atom))
def _bonds(b):
    from .molarray import Bonds
    return Bonds(b)
def _pseudobonds(b):
    from .molarray import PseudoBonds
    return PseudoBonds(b)
def _residue(p):
    return object_map(p, Residue)
def _residues(r):
    from .molarray import Residues
    return Residues(r)
def _chains(c):
    from .molarray import Chains
    return Chains(c)
def _atomic_structure(p):
    return object_map(p, CAtomicStructure)
def _pseudobond_group_map(pbgc_map):
    from .molarray import PseudoBonds
    pbg_map = dict((name, PseudoBonds(pbg)) for name, pbg in pbgc_map.items())
    return pbg_map

# -----------------------------------------------------------------------------
#
class Atom:

    def __init__(self, atom_pointer):
        set_c_pointer(self, atom_pointer)

    bfactor = c_property('atom_bfactor', float32)
    bonds = c_property('atom_bonds', cptr, 'num_bonds', astype = _bonds, read_only = True)
    bonded_atoms = c_property('atom_bonded_atoms', cptr, 'num_bonds', astype = _atoms, read_only = True)
    color = c_property('atom_color', uint8, 4)
    coord = c_property('atom_coord', float64, 3)
    display = c_property('atom_display', npy_bool)
    draw_mode = c_property('atom_draw_mode', int32)
    element_name = c_property('atom_element_name', string, read_only = True)
    element_number = c_property('atom_element_number', int32, read_only = True)
    molecule = c_property('atom_molecule', cptr, astype = _atomic_structure, read_only = True)
    name = c_property('atom_name', string, read_only = True)
    num_bonds = c_property('atom_num_bonds', int32, read_only = True)
    radius = c_property('atom_radius', float32)
    residue = c_property('atom_residue', cptr, astype = _residue, read_only = True)

    def connects_to(self, atom):
        f = c_function('atom_connects_to',
                       args = (ctypes.c_void_p, ctypes.c_void_p),
                       ret = ctypes.c_int)
        c = f(self._c_pointer, atom._c_pointer)
        return c

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
class CPseudoBondGroup:

    def __init__(self, name):
        f = c_function('pseudobond_group_get', args = [ctypes.c_char_p], ret = ctypes.c_void_p)
        pbg_pointer = f(name.encode('utf-8'))
        set_c_pointer(self, pbg_pointer)
        add_to_object_map(self)

    num_pseudobonds = c_property('pseudobond_group_num_pseudobonds', int32, read_only = True)
    pseudobonds = c_property('pseudobond_group_pseudobonds', cptr, 'num_pseudobonds',
                             astype = _pseudobonds, read_only = True)

    def new_pseudobond(self, atom1, atom2):
        f = c_function('pseudobond_group_new_pseudobond',
                       args = (ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p),
                       ret = ctypes.c_void_p)
        pb = f(self._c_pointer, atom1._c_pointer, atom2._c_pointer)
        return object_map(pb, PseudoBond)

    def delete(self):
        c_function('pseudobond_group_delete', args = [ctypes.c_void_p])(self._c_pointer)

# -----------------------------------------------------------------------------
#
class Residue:

    def __init__(self, residue_pointer):
        set_c_pointer(self, residue_pointer)

    atoms = c_property('residue_atoms', cptr, 'num_atoms', astype = _atoms, read_only = True)
    chain_id = c_property('residue_chain_id', string, read_only = True)
    is_helix = c_property('residue_is_helix', npy_bool)
    is_sheet = c_property('residue_is_sheet', npy_bool)
    ss_id = c_property('residue_ss_id', int32)
    ribbon_display = c_property('residue_ribbon_display', npy_bool)
    ribbon_color = c_property('residue_ribbon_color', uint8, 4)
    name = c_property('residue_name', string, read_only = True)
    num_atoms = c_property('residue_num_atoms', int32, read_only = True)
    number = c_property('residue_number', int32, read_only = True)
    str = c_property('residue_str', string, read_only = True)
    unique_id = c_property('residue_unique_id', int32, read_only = True)
    # TODO: Currently no C++ method to get Chain

    def add_atom(self, atom):
        f = c_function('residue_add_atom', args = (ctypes.c_void_p, ctypes.c_void_p))
        f(self._c_pointer, atom._c_pointer)

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
class CAtomicStructure:

    def __init__(self, mol_pointer = None):
        if mol_pointer is None:
            # Create a new molecule
            mol_pointer = c_function('molecule_new', args = (), ret = ctypes.c_void_p)()
        set_c_pointer(self, mol_pointer)

    def delete(self):
        c_function('molecule_delete', args = (ctypes.c_void_p,))(self._c_pointer)

    atoms = c_property('molecule_atoms', cptr, 'num_atoms', astype = _atoms, read_only = True)
    bonds = c_property('molecule_bonds', cptr, 'num_bonds', astype = _bonds, read_only = True)
    chains = c_property('molecule_chains', cptr, 'num_chains', astype = _chains, read_only = True)
    name = c_property('molecule_name', string)
    num_atoms = c_property('molecule_num_atoms', int32, read_only = True)
    num_bonds = c_property('molecule_num_bonds', int32, read_only = True)
    num_coord_sets = c_property('molecule_num_coord_sets', int32, read_only = True)
    num_chains = c_property('molecule_num_chains', int32, read_only = True)
    num_residues = c_property('molecule_num_residues', int32, read_only = True)
    residues = c_property('molecule_residues', cptr, 'num_residues', astype = _residues, read_only = True)
    pbg_map = c_property('molecule_pbg_map', pyobject, astype = _pseudobond_group_map, read_only = True)

    def new_atom(self, atom_name, element_name):
        f = c_function('molecule_new_atom',
                       args = (ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p),
                       ret = ctypes.c_void_p)
        ap = f(self._c_pointer, atom_name.encode('utf-8'), element_name.encode('utf-8'))
        return object_map(ap, Atom)

    def new_bond(self, atom1, atom2):
        f = c_function('molecule_new_bond',
                       args = (ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p),
                       ret = ctypes.c_void_p)
        bp = f(self._c_pointer, atom1._c_pointer, atom2._c_pointer)
        return object_map(bp, Bond)

    def new_residue(self, residue_name, chain_id, pos):
        f = c_function('molecule_new_residue',
                       args = (ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int),
                       ret = ctypes.c_void_p)
        rp = f(self._c_pointer, residue_name.encode('utf-8'), chain_id.encode('utf-8'), pos)
        return object_map(rp, Residue)

    def polymers(self, consider_missing_structure = True, consider_chains_ids = True):
        f = c_function('molecule_polymers',
                       args = (ctypes.c_void_p, ctypes.c_int, ctypes.c_int),
                       ret = ctypes.py_object)
        resarrays = f(self._c_pointer, consider_missing_structure, consider_chains_ids)
        from .molarray import Residues
        return tuple(Residues(ra) for ra in resarrays)

# -----------------------------------------------------------------------------
#
_object_map = {}	# Map C++ pointer to Python object
def object_map(p, object_type):
    global _object_map
    o = _object_map.get(p, None)
    if o is None:
        _object_map[p] = o = object_type(p)
    return o

def add_to_object_map(object):
    _object_map[object._c_pointer.value] = object

def register_object_map_deletion_handler(omap):
    '''
    When a C++ object such as an Atom is deleted the pointer is removed
    from the object map if it exists and the Python object has its _c_pointer
    attribute deleted.
    '''
    f = c_function('object_map_deletion_handler', args = [ctypes.c_void_p], ret = ctypes.c_void_p)
    p = ctypes.c_void_p(id(omap))
    global _omd_handler
    _omd_handler = Object_Map_Deletion_Handler(f(p))

_omd_handler = None
class Object_Map_Deletion_Handler:
    def __init__(self, h):
        self.h = h
        self.delete_handler = c_function('delete_object_map_deletion_handler', args = [ctypes.c_void_p])
    def __del__(self):
        # Make sure object map deletion handler is removed before Python exits
        # so later C++ deletes don't cause segfault on exit.
        self.delete_handler(self.h)

register_object_map_deletion_handler(_object_map)
