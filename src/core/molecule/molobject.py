from numpy import uint8, int32, float64, float32, bool as npy_bool
from .molc import get_value, set_value, string, cptr, pyobject


# -----------------------------------------------------------------------------
#
class Atom:

    def __init__(self, atom_pointer):
        self._atom = atom_pointer

    def get_bfactor(self):
        return get_value('atom_bfactor', self._atom, float32)
    def set_bfactor(self, b):
        set_value('set_atom_bfactor', self._atom, b, float32)
    bfactor = property(get_bfactor, set_bfactor)

    def get_color(self):
        return get_value('atom_color', self._atom, uint8, 4)
    def set_color(self, rgba):
        set_value('set_atom_color', self._atom, rgba, uint8, 4)
    color = property(get_color, set_color)

    def get_coord(self):
        return get_value('atom_coord', self._atom, float64, 3)
    def set_coord(self, xyz):
        set_value('set_atom_coord', self._atom, xyz, float64, 3)
    coord = property(get_coord, set_coord)

    def get_display(self):
        return get_value('atom_display', self._atom, npy_bool)
    def set_display(self, d):
        set_value('set_atom_display', self._atom, d, npy_bool)
    display = property(get_display, set_display)

    def get_draw_mode(self):
        return get_value('atom_draw_mode', self._atom, int32)
    def set_draw_mode(self, mode):
        set_value('set_atom_draw_mode', self._atom, mode, int32)
    draw_mode = property(get_draw_mode, set_draw_mode)

    def get_element_name(self):
        return get_value('atom_element_name', self._atom, string)
    element_name = property(get_element_name, None)

    def get_element_number(self):
        return get_value('atom_element_number', self._atom, int32)
    element_number = property(get_element_number, None)

    def get_name(self):
        return get_value('atom_name', self._atom, string)
    name = property(get_name, None)

    def get_radius(self):
        return get_value('atom_radius', self._atom, float32)
    def set_radius(self, r):
        set_value('set_atom_radius', self._atom, r, float32)
    radius = property(get_radius, set_radius)

    def get_residue(self):
        rp = get_value('atom_residue', self._atom, cptr)
        return object_map(rp, Residue)
    residue = property(get_residue, None)


# -----------------------------------------------------------------------------
#
class Bond:

    def __init__(self, bond_pointer):
        self._bond = bond_pointer

    def get_atoms(self):
        a = get_value('bond_atoms', self._bond, cptr, 2)
        return (object_map(a[0],Atom), object_map(a[1],Atom))
    atoms = property(get_atoms, None)

    def get_color(self):
        "numpy array of uint8 RGBA values"
        return get_value('bond_color', self._bond, uint8, 4)
    def set_color(self, rgba):
        set_value('set_bond_color', self._bond, rgba, uint8, 4)
    color = property(get_color, set_color)

    def get_display(self):
        return get_value('bond_display', self._bond, int32)
    def set_display(self, d):
        set_value('set_bond_display', self._bond, d, int32)
    display = property(get_display, set_display)

    def get_halfbond(self):
        return get_value('bond_halfbond', self._bond, npy_bool)
    def set_halfbond(self, d):
        set_value('set_bond_halfbond', self._bond, d, npy_bool)
    halfbond = property(get_halfbond, set_halfbond)

    def get_radius(self):
        return get_value('bond_radius', self._bond, float32)
    def set_radius(self, r):
        set_value('set_bond_radius', self._bond, r, float32)
    radius = property(get_radius, set_radius)

# -----------------------------------------------------------------------------
#
class PseudoBond:

    def __init__(self, pbond_pointer):
        self._pbond = pbond_pointer

    def get_atoms(self):
        a = get_value('pseudobond_atoms', self._pbond, cptr, 2)
        return (object_map(a[0],Atom), object_map(a[1],Atom))
    atoms = property(get_atoms, None)

    def get_color(self):
        "numpy array of uint8 RGBA values"
        return get_value('pseudobond_color', self._pbond, uint8, 4)
    def set_color(self, rgba):
        set_value('set_pseudobond_color', self._pbond, rgba, uint8, 4)
    color = property(get_color, set_color)

    def get_display(self):
        return get_value('pseudobond_display', self._pbond, int32)
    def set_display(self, d):
        set_value('set_pseudobond_display', self._pbond, d, int32)
    display = property(get_display, set_display)

    def get_halfbond(self):
        return get_value('pseudobond_halfbond', self._pbond, npy_bool)
    def set_halfbond(self, d):
        set_value('set_pseudobond_halfbond', self._pbond, d, npy_bool)
    halfbond = property(get_halfbond, set_halfbond)

    def get_radius(self):
        return get_value('pseudobond_radius', self._pbond, float32)
    def set_radius(self, r):
        set_value('set_pseudobond_radius', self._pbond, r, float32)
    radius = property(get_radius, set_radius)


# -----------------------------------------------------------------------------
#
class Residue:

    def __init__(self, residue_cpp):
        self._res = residue_cpp     # C++ pointer

    def get_atoms(self):
        a = get_value('residue_atoms', self._res, cptr, self.num_atoms)
        from .molarray import Atoms
        return Atoms(a)
    atoms = property(get_atoms, None)

    def get_num_atoms(self):
        return get_value('residue_num_atoms', self._res, int32)
    num_atoms = property(get_num_atoms, None)

    def get_chain_id(self):
        return get_value('residue_chain_id', self._res, string)
    chain_id = property(get_chain_id, None)

    def get_name(self):
        "residue name"
        return get_value('residue_name', self._res, string)
    name = property(get_name, None)

    def get_number(self):
        "residue sequence number"
        return get_value('residue_number', self._res, int32)
    number = property(get_number, None)

    def get_str(self):
        "human-friendly residue identifier strings"
        return get_value('residue_str', self._res, string)
    str = property(get_str, None)

    def get_unique_id(self):
        "integer id unique for each chain and residue number"
        return get_value('residue_unique_id', self._res, int32)
    unique_id = property(get_unique_id, None)


# -----------------------------------------------------------------------------
#
class Chain:

    def __init__(self, chain_cpp):
        self._chain = chain_cpp     # C++ pointer

    def get_chain_id(self):
        return get_value('chain_chain_id', self._chain, string)
    chain_id = property(get_chain_id, None)

    def get_residues(self):
        r = get_value('chain_residues', self._chain, cptr, self.num_residues)
        from .molarray import Residues
        return Residues(r)
    residues = property(get_residues, None)

    def get_num_residues(self):
        return get_value('chain_num_residues', self._chain, int32)
    num_residues = property(get_num_residues, None)

# -----------------------------------------------------------------------------
#
class AtomicStructure:

    def __init__(self, mol_cpp):
        self._mol = mol_cpp     # C++ pointer

    def get_name(self):
        return get_value('molecule_name', self._mol, string)
    def set_name(self):
        set_value('set_molecule_name', self._mol, string)
    name = property(get_name, set_name)

    def get_atoms(self):
        a = get_value('molecule_atoms', self._mol, cptr, self.num_atoms)
        from .molarray import Atoms
        return Atoms(a)
    atoms = property(get_atoms, None)

    def get_num_atoms(self):
        return get_value('molecule_num_atoms', self._mol, int32)
    num_atoms = property(get_num_atoms, None)

    def get_bonds(self):
        b = get_value('molecule_bonds', self._mol, cptr, self.num_bonds)
        from .molarray import Bonds
        return Bonds(b)
    bonds = property(get_bonds, None)

    def get_num_bonds(self):
        return get_value('molecule_num_bonds', self._mol, int32)
    num_bonds = property(get_num_bonds, None)

    def get_residues(self):
        r = get_value('molecule_residues', self._mol, cptr, self.num_residues)
        from .molarray import Residues
        return Residues(r)
    residues = property(get_residues, None)

    def get_num_residues(self):
        return get_value('molecule_num_residues', self._mol, int32)
    num_residues = property(get_num_residues, None)

    def get_chains(self):
        c = get_value('molecule_chains', self._mol, cptr, self.num_chains)
        from .molarray import Residues
        return Chains(c)
    chains = property(get_chains, None)

    def get_num_chains(self):
        return get_value('molecule_num_chains', self._mol, int32)
    num_chains = property(get_num_chains, None)

    def get_pbg_map(self):
        "map from name to PseudoBonds for each group"
        pbgc_map = get_value('molecule_pbg_map', self._mol, pyobject)
        from .molarray import PseudoBonds
        pbg_map = dict((name, PseudoBonds(pbg)) for name, pbg in pbgc_map.items())
        return pbg_map
    pbg_map = property(get_pbg_map, None)

# -----------------------------------------------------------------------------
#
_object_map = {}	# Map C++ pointer to Python object
def object_map(p, object_type):
    global _object_map
    o = _object_map.get(p, None)
    if o is None:
        _object_map[p] = o = object_type(p)
    return o

# -----------------------------------------------------------------------------
# Return an AtomicStructure for a C++ StructBlob.
#
def atomic_structure_from_blob(struct_blob):
    return object_map(struct_blob._struct_pointers[0], AtomicStructure)
