# vi: set expandtab shiftwidth=4 softtabstop=4:
from numpy import uint8, int32, float64, float32, bool as npy_bool
from .molc import string, cptr, pyobject, c_property, set_c_pointer, c_function, ctype_type_to_numpy
import ctypes
size_t = ctype_type_to_numpy[ctypes.c_size_t]   # numpy dtype for size_t

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
    from .molarray import Pseudobonds
    return Pseudobonds(b)
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
    from ..pbgroup import PseudobondGroup
    pbg_map = dict((name, object_map(pbg,PseudobondGroup)) for name, pbg in pbgc_map.items())
    return pbg_map

# -----------------------------------------------------------------------------
#
class Atom:

    def __init__(self, atom_pointer):
        set_c_pointer(self, atom_pointer)

    bfactor = c_property('atom_bfactor', float32)
    bonds = c_property('atom_bonds', cptr, 'num_bonds', astype = _bonds, read_only = True)
    bonded_atoms = c_property('atom_bonded_atoms', cptr, 'num_bonds', astype = _atoms, read_only = True)
    chain_id = c_property('atom_chain_id', string, read_only = True)
    color = c_property('atom_color', uint8, 4)
    coord = c_property('atom_coord', float64, 3)
    display = c_property('atom_display', npy_bool)
    draw_mode = c_property('atom_draw_mode', int32)
    element_name = c_property('atom_element_name', string, read_only = True)
    element_number = c_property('atom_element_number', uint8, read_only = True)
    in_chain = c_property('atom_in_chain', npy_bool, read_only = True)
    is_backbone = c_property('atom_is_backbone', npy_bool)
    structure = c_property('atom_structure', cptr, astype = _atomic_structure, read_only = True)
    name = c_property('atom_name', string, read_only = True)
    num_bonds = c_property('atom_num_bonds', size_t, read_only = True)
    radius = c_property('atom_radius', float32)
    residue = c_property('atom_residue', cptr, astype = _residue, read_only = True)
    selected = c_property('atom_selected', npy_bool)

    def connects_to(self, atom):
        f = c_function('atom_connects_to',
                       args = (ctypes.c_void_p, ctypes.c_void_p),
                       ret = ctypes.c_bool)
        c = f(self._c_pointer, atom._c_pointer)
        return c

    @property
    def scene_coord(self):
        return self.structure.scene_position * self.coord

# -----------------------------------------------------------------------------
#
class Bond:

    def __init__(self, bond_pointer):
        set_c_pointer(self, bond_pointer)

    atoms = c_property('bond_atoms', cptr, 2, astype = _atom_pair, read_only = True)
    color = c_property('bond_color', uint8, 4)
    display = c_property('bond_display', uint8)
    halfbond = c_property('bond_halfbond', npy_bool)
    radius = c_property('bond_radius', float32)

    def other_atom(self, atom):
        a1,a2 = self.atoms
        return a2 if atom is a1 else a1

# -----------------------------------------------------------------------------
#
class Pseudobond:

    def __init__(self, pbond_pointer):
        set_c_pointer(self, pbond_pointer)

    atoms = c_property('pseudobond_atoms', cptr, 2, astype = _atom_pair, read_only = True)
    color = c_property('pseudobond_color', uint8, 4)
    display = c_property('pseudobond_display', uint8)
    halfbond = c_property('pseudobond_halfbond', npy_bool)
    radius = c_property('pseudobond_radius', float32)

    @property
    def length(self):
        a1, a2 = self.atoms
        v = a1.scene_coord - a2.scene_coord
        from math import sqrt
        return sqrt((v*v).sum())

# -----------------------------------------------------------------------------
#
class CPseudobondGroup:
    '''Pseudobond group.'''

    def __init__(self, pbg_pointer):
        set_c_pointer(self, pbg_pointer)

    category = c_property('pseudobond_group_category', string, read_only = True)
    gc_color = c_property('pseudobond_group_gc_color', npy_bool)
    gc_select = c_property('pseudobond_group_gc_select', npy_bool)
    gc_shape = c_property('pseudobond_group_gc_shape', npy_bool)
    num_pseudobonds = c_property('pseudobond_group_num_pseudobonds', size_t, read_only = True)
    pseudobonds = c_property('pseudobond_group_pseudobonds', cptr, 'num_pseudobonds',
                             astype = _pseudobonds, read_only = True)

    def new_pseudobond(self, atom1, atom2):
        f = c_function('pseudobond_group_new_pseudobond',
                       args = (ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p),
                       ret = ctypes.c_void_p)
        pb = f(self._c_pointer, atom1._c_pointer, atom2._c_pointer)
        return object_map(pb, Pseudobond)


# -----------------------------------------------------------------------------
#
class PseudobondManager:
    '''Per-session singleton pseudobond manager'''

    def __init__(self):
        f = c_function('pseudobond_create_global_manager', args = (), ret = ctypes.c_void_p)
        set_c_pointer(self, f())

    def get_group(self, name, create = True):
        f = c_function('pseudobond_global_manager_get_group',
                       args = (ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int),
                       ret = ctypes.c_void_p)
        pbg = f(self._c_pointer, name.encode('utf-8'), create)
        if not pbg:
            return None
        from ..pbgroup import PseudobondGroup
        return object_map(pbg, PseudobondGroup)


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
    num_atoms = c_property('residue_num_atoms', size_t, read_only = True)
    number = c_property('residue_number', int32, read_only = True)
    str = c_property('residue_str', string, read_only = True)
    unique_id = c_property('residue_unique_id', int32, read_only = True)
    structure = c_property('residue_structure', cptr, astype = _atomic_structure, read_only = True)
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
    structure = c_property('chain_structure', cptr, astype = _atomic_structure, read_only = True)
    residues = c_property('chain_residues', cptr, 'num_residues', astype = _residues, read_only = True)
    num_residues = c_property('chain_num_residues', size_t, read_only = True)

# -----------------------------------------------------------------------------
#
class CAtomicStructure:

    def __init__(self, mol_pointer = None):
        if mol_pointer is None:
            # Create a new atomic structure
            mol_pointer = c_function('structure_new', args = (), ret = ctypes.c_void_p)()
        set_c_pointer(self, mol_pointer)

    def delete(self):
        c_function('structure_delete', args = (ctypes.c_void_p,))(self._c_pointer)

    atoms = c_property('structure_atoms', cptr, 'num_atoms', astype = _atoms, read_only = True)
    bonds = c_property('structure_bonds', cptr, 'num_bonds', astype = _bonds, read_only = True)
    chains = c_property('structure_chains', cptr, 'num_chains', astype = _chains, read_only = True)
    gc_color = c_property('structure_gc_color', npy_bool)
    gc_select = c_property('structure_gc_select', npy_bool)
    gc_shape = c_property('structure_gc_shape', npy_bool)
    name = c_property('structure_name', string)
    num_atoms = c_property('structure_num_atoms', size_t, read_only = True)
    num_bonds = c_property('structure_num_bonds', size_t, read_only = True)
    num_coord_sets = c_property('structure_num_coord_sets', size_t, read_only = True)
    num_chains = c_property('structure_num_chains', size_t, read_only = True)
    num_residues = c_property('structure_num_residues', size_t, read_only = True)
    residues = c_property('structure_residues', cptr, 'num_residues', astype = _residues, read_only = True)
    pbg_map = c_property('structure_pbg_map', pyobject, astype = _pseudobond_group_map, read_only = True)

    def new_atom(self, atom_name, element_name):
        f = c_function('structure_new_atom',
                       args = (ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p),
                       ret = ctypes.c_void_p)
        ap = f(self._c_pointer, atom_name.encode('utf-8'), element_name.encode('utf-8'))
        return object_map(ap, Atom)

    def new_bond(self, atom1, atom2):
        f = c_function('structure_new_bond',
                       args = (ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p),
                       ret = ctypes.c_void_p)
        bp = f(self._c_pointer, atom1._c_pointer, atom2._c_pointer)
        return object_map(bp, Bond)

    def new_residue(self, residue_name, chain_id, pos):
        f = c_function('structure_new_residue',
                       args = (ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int),
                       ret = ctypes.c_void_p)
        rp = f(self._c_pointer, residue_name.encode('utf-8'), chain_id.encode('utf-8'), pos)
        return object_map(rp, Residue)

    def polymers(self, consider_missing_structure = True, consider_chains_ids = True):
        f = c_function('structure_polymers',
                       args = (ctypes.c_void_p, ctypes.c_int, ctypes.c_int),
                       ret = ctypes.py_object)
        resarrays = f(self._c_pointer, consider_missing_structure, consider_chains_ids)
        from .molarray import Residues
        return tuple(Residues(ra) for ra in resarrays)

    def pseudobond_group(self, name, create_type = "normal"):
        if create_type is None:
            create_arg = 0
        elif create_type == "normal":
            create_arg = 1
        else:  # per-coordset
            create_arg = 2
        f = c_function('structure_pseudobond_group',
                       args = (ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int),
                       ret = ctypes.c_void_p)
        pbg = f(self._c_pointer, name.encode('utf-8'), create_arg)
        return object_map(pbg, PseudobondGroup)

# -----------------------------------------------------------------------------
#
class Element:

    def __init__(self, e_pointer = None, name = None, number = 6):
        if e_pointer is None:
            # Create a new element
            if name:
                f = c_function('element_new_name', args = (ctypes.c_char_p,), ret = ctypes.c_void_p)
                e_pointer = f(name)
            else:
                f = c_function('element_new_number', args = (ctypes.c_int,), ret = ctypes.c_void_p)
                e_pointer = f(number)
        set_c_pointer(self, e_pointer)

    name = c_property('element_name', string, read_only = True)
    number = c_property('element_number', uint8, read_only = True)
    mass = c_property('element_mass', float32, read_only = True)
    is_metal = c_property('element_is_metal', npy_bool, read_only = True)

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
