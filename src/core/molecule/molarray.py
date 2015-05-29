from numpy import uint8, int32, float64, float32, bool as npy_bool
from .molc import string, cptr, pyobject, cvec_property, set_cvec_pointer
from . import molobject

def _atoms(a):
    return Atoms(a)
def _bonds(a):
    return Bonds(a)
def _residues(a):
    return Residues(a)
def _chains(a):
    return Chains(a)
def _atomic_structures(p):
    return CAtomicStructures(p)
def _unique_atomic_structures(p):
    import numpy
    return CAtomicStructures(numpy.unique(p))
def _residues(p):
    return Residues(p)
def _atoms_pair(p):
    return (Atoms(p[:,0].copy()), Atoms(p[:,1].copy()))
def _pseudobond_group_map(a):
    from . import molobject
    return [molobject._pseudobond_group_map(p) for p in a]

# -----------------------------------------------------------------------------
#
class PointerArray:
    def __init__(self, pointers, object_class, objects_class):
        if pointers is None:
            # Empty Atoms
            import numpy
            pointers = numpy.empty((0,), cptr)
        self._pointers = pointers
        self._object_class = object_class
        self._objects_class = objects_class
        set_cvec_pointer(self, pointers)

    def __len__(self):
        return len(self._pointers)
    def __iter__(self):
        if not hasattr(self, '_object_list'):
            from .molobject import Atom, object_map
            c = self._object_class
            self._object_list = [object_map(p,c) for p in self._pointers]
        return iter(self._object_list)
    def __or__(self, objects):
        return self.merge(objects)
    def __and__(self, objects):
        return self.intersect(objects)
    def __sub__(self, objects):
        return self.subtract(objects)

    def intersect(self, objects):
        import numpy
        return self._objects_class(numpy.intersect1d(self._pointers, objects._pointers))
    def filter(self, mask):
        return self._objects_class(self._pointers[mask])
    def merge(self, objects):
        import numpy
        return self._objects_class(numpy.union1d(self._pointers, objects._pointers))
    def subtract(self, objects):
        import numpy
        return self._objects_class(numpy.setdiff1d(self._pointers, objects._pointers))

# -----------------------------------------------------------------------------
#
class Atoms(PointerArray):

    def __init__(self, atom_pointers = None):
        PointerArray.__init__(self, atom_pointers, molobject.Atom, Atoms)

    bfactors = cvec_property('atom_bfactor', float32)
    colors = cvec_property('atom_color', uint8, 4)
    coords = cvec_property('atom_coord', float64, 3)
    displays = cvec_property('atom_display', npy_bool)
    draw_modes = cvec_property('atom_draw_mode', int32)
    element_names = cvec_property('atom_element_name', string, read_only = True)
    element_numbers = cvec_property('atom_element_number', int32, read_only = True)
    molecules = cvec_property('atom_molecule', cptr, astype = _atomic_structures, read_only = True)
    unique_molecules = cvec_property('atom_molecule', cptr, astype = _unique_atomic_structures, read_only = True)
    names = cvec_property('atom_name', string, read_only = True)
    radii = cvec_property('atom_radius', float32)
    residues = cvec_property('atom_residue', cptr, astype = _residues, read_only = True)

# -----------------------------------------------------------------------------
#
class Bonds(PointerArray):

    def __init__(self, bond_pointers):
        PointerArray.__init__(self, bond_pointers, molobject.Bond, Bonds)

    atoms = cvec_property('bond_atoms', cptr, 2, astype = _atoms_pair, read_only = True)
    colors = cvec_property('bond_color', uint8, 4)
    displays = cvec_property('bond_display', int32)
    halfbonds = cvec_property('bond_halfbond', npy_bool)
    radii = cvec_property('bond_radius', float32)

# -----------------------------------------------------------------------------
#
class PseudoBonds(PointerArray):

    def __init__(self, pbond_pointers):
        PointerArray.__init__(self, pbond_pointers, molobject.PseudoBond, PseudoBonds)

    atoms = cvec_property('pseudobond_atoms', cptr, 2, astype = _atoms_pair, read_only = True)
    colors = cvec_property('pseudobond_color', uint8, 4)
    displays = cvec_property('pseudobond_display', int32)
    halfbonds = cvec_property('pseudobond_halfbond', npy_bool)
    radii = cvec_property('pseudobond_radius', float32)

# -----------------------------------------------------------------------------
#
class Residues(PointerArray):

    def __init__(self, residue_pointers):
        PointerArray.__init__(self, residue_pointers, molobject.Residue, Residues)

    atoms = cvec_property('residue_atoms', cptr, 'num_atoms', astype = _atoms, read_only = True, per_object = False)
    chain_ids = cvec_property('residue_chain_id', string, read_only = True)
    molecules = cvec_property('residue_molecule', cptr, astype = _atomic_structures, read_only = True)
    names = cvec_property('residue_name', string, read_only = True)
    num_atoms = cvec_property('residue_num_atoms', int32, read_only = True)
    numbers = cvec_property('residue_number', int32, read_only = True)
    strs = cvec_property('residue_str', string, read_only = True)
    unique_ids = cvec_property('residue_unique_id', int32, read_only = True)

# -----------------------------------------------------------------------------
#
class Chains(PointerArray):

    def __init__(self, chain_pointers):
        PointerArray.__init__(self, chain_pointers, molobject.Chain, Chains)

    chain_ids = cvec_property('chain_chain_id', string, read_only = True)
    molecules = cvec_property('chain_molecule', cptr, astype = _atomic_structures, read_only = True)
    residues = cvec_property('chain_residues', cptr, 'num_residues', astype = _residues,
                             read_only = True, per_object = False)
    num_residues = cvec_property('chain_num_residues', int32, read_only = True)

# -----------------------------------------------------------------------------
#
class CAtomicStructures(PointerArray):

    def __init__(self, mol_pointers):
        PointerArray.__init__(self, mol_pointers, molobject.CAtomicStructure, CAtomicStructures)

    atoms = cvec_property('molecule_atoms', cptr, 'num_atoms', astype = _atoms,
                          read_only = True, per_object = False)
    bonds = cvec_property('molecule_bonds', cptr, 'num_bonds', astype = _bonds,
                          read_only = True, per_object = False)
    chains = cvec_property('molecule_chains', cptr, 'num_chains', astype = _chains,
                           read_only = True, per_object = False)
    names = cvec_property('molecule_name', string)
    num_atoms = cvec_property('molecule_num_atoms', int32, read_only = True)
    num_bonds = cvec_property('molecule_num_bonds', int32, read_only = True)
    num_chains = cvec_property('molecule_num_chains', int32, read_only = True)
    num_residues = cvec_property('molecule_num_residues', int32, read_only = True)
    residues = cvec_property('molecule_residues', cptr, 'num_residues', astype = _residues,
                             read_only = True, per_object = False)
    pbg_maps = cvec_property('molecule_pbg_map', pyobject, astype = _pseudobond_group_map, read_only = True)
