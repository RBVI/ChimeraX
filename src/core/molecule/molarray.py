from numpy import uint8, int32, float64, float32, bool as npy_bool
from .molc import get_value, set_value, string, cptr

# -----------------------------------------------------------------------------
#
class Atoms:

    def __init__(self, atoms_cpp):
        self._atoms = atoms_cpp           # Numpy array of C++ pointers

    def __len__(self):
        return len(self._atoms)

    def __iter__(self):
        if not hasattr(self, '_atom_list'):
            from .molobject import Atom, object_map
            self._atom_list = [object_map(a,Atom) for a in self._atoms]
        return iter(self._atom_list)

    def get_bfactor(self):
        return get_value('atom_bfactor', self._atoms, float32)
    def set_bfactor(self, b):
        set_value('set_atom_bfactor', self._atoms, b, float32)
    bfactors = property(get_bfactor, set_bfactor)
            
    def get_color(self):
        "numpy Nx4 array of (unsigned char) RGBA values"
        return get_value('atom_color', self._atoms, uint8, 4)
    def set_color(self, rgba):
        set_value('set_atom_color', self._atoms, rgba, uint8, 4)
    colors = property(get_color, set_color)

    def get_coord(self):
        "numpy Nx3 array of atom coordinates"
        return get_value('atom_coord', self._atoms, float64, 3)
    def set_coord(self, xyz):
        set_value('set_atom_coord', self._atoms, xyz, float64, 3)
    coords = property(get_coord, set_coord)

    def get_display(self):
        "numpy array of (bool) displays"
        return get_value('atom_display', self._atoms, npy_bool)
    def set_display(self, d):
        set_value('set_atom_display', self._atoms, d, npy_bool)
    displays = property(get_display, set_display)

    def get_draw_mode(self):
        "numpy array of (int) draw modes"
        return get_value('atom_draw_mode', self._atoms, int32)
    def set_draw_mode(self, modes):
        set_value('set_atom_draw_mode', self._atoms, modes, int32)
    draw_modes = property(get_draw_mode, set_draw_mode)

    def get_element_name(self):
        "numpy array of element names"
        return get_value('atom_element_name', self._atoms, string)
    element_names = property(get_element_name, None)

    def get_element_number(self):
        "numpy array of element numbers"
        return get_value('atom_element_number', self._atoms, int32)
    element_numbers = property(get_element_number, None)

    def get_molecule(self):
        "numpy array of molecule for each atom"
        mp = get_value('atom_molecule', self._atoms, cptr)
        return AtomicStructures(mp)
    molecules = property(get_molecule, None)

    def get_unique_molecules(self):
        mp = get_value('atom_molecule', self._atoms, cptr)
        import numpy
        return AtomicStructures(numpy.unique(mp))
    unique_molecules = property(get_unique_molecules, None)

    def get_name(self):
        "numpy array of atom names"
        return get_value('atom_name', self._atoms, string)
    names = property(get_name, None)

    def get_radius(self):
        "numpy array of (float) atomic radii"
        return get_value('atom_radius', self._atoms, float32)
    def set_radius(self, r):
        set_value('set_atom_radius', self._atoms, r, float32)
    radii = property(get_radius, set_radius)

    def get_residue(self):
        "numpy array of residue for each atom"
        rp = get_value('atom_residue', self._atoms, cptr)
        return Residues(rp)
    residues = property(get_residue, None)

    def intersect(self, atoms):
        import numpy
        return Atoms(numpy.intersect1d(self._atoms, atoms._atoms))
    def filter(self, mask):
        return Atoms(self._atoms[mask])
    def merge(self, atoms):
        import numpy
        return Atoms(numpy.union1d(self._atoms, atoms._atoms))
    def subtract(self, atoms):
        import numpy
        return Atoms(numpy.setdiff1d(self._atoms, atoms._atoms))
    def __or__(self, atoms):
        return self.merge(atoms)
    def __and__(self, atoms):
        return self.intersect(atoms)
    def __sub__(self, atoms):
        return self.subtract(atoms)

# -----------------------------------------------------------------------------
#
class Bonds:

    def __init__(self, bonds_cpp):
        self._bonds = bonds_cpp           # Numpy array of C++ pointers

    def __len__(self):
        return len(self._bonds)

    def __iter__(self):
        if not hasattr(self, '_bond_list'):
            from .molobject import Bond, object_map
            self._bond_list = [object_map(b,Bond) for b in self._bonds]
        return iter(self._bond_list)

    def get_atoms(self):
        "returns two Atoms"
        a = get_value('bond_atoms', self._bonds, cptr, 2)
        return (Atoms(a[:,0].copy()), Atoms(a[:,1].copy()))
    atoms = property(get_atoms, None)

    def get_color(self):
        "numpy array of uint8 RGBA values"
        return get_value('bond_color', self._bonds, uint8, 4)
    def set_color(self, rgba):
        set_value('set_bond_color', self._bonds, rgba, uint8, 4)
    colors = property(get_color, set_color)

    def get_display(self):
        return get_value('bond_display', self._bonds, int32)
    def set_display(self, d):
        set_value('set_bond_display', self._bonds, d, int32)
    displays = property(get_display, set_display)

    def get_halfbond(self):
        return get_value('bond_halfbond', self._bonds, npy_bool)
    def set_halfbond(self, d):
        set_value('set_bond_halfbond', self._bonds, d, npy_bool)
    halfbonds = property(get_halfbond, set_halfbond)

    def get_radius(self):
        return get_value('bond_radius', self._bonds, float32)
    def set_radius(self, r):
        set_value('set_bond_radius', self._bonds, r, float32)
    radii = property(get_radius, set_radius)

# -----------------------------------------------------------------------------
#
class PseudoBonds:

    def __init__(self, pbonds_cpp):
        self._pbonds = pbonds_cpp           # Numpy array of C++ pointers

    def __len__(self):
        return len(self._pbonds)

    def __iter__(self):
        if not hasattr(self, '_pbond_list'):
            from .molobject import PseudoBond, object_map
            self._pbond_list = [object_map(pb,PseudoBond) for pb in self._pbonds]
        return iter(self._pbond_list)

    def get_atoms(self):
        "returns two Atoms"
        a = get_value('pseudobond_atoms', self._pbonds, cptr, 2)
        return (Atoms(a[:,0].copy()), Atoms(a[:,1].copy()))
    atoms = property(get_atoms, None)

    def get_color(self):
        "numpy array of uint8 RGBA values"
        return get_value('pseudobond_color', self._pbonds, uint8, 4)
    def set_color(self, rgba):
        set_value('set_pseudobond_color', self._pbonds, rgba, uint8, 4)
    colors = property(get_color, set_color)

    def get_display(self):
        return get_value('pseudobond_display', self._pbonds, int32)
    def set_display(self, d):
        set_value('set_pseudobond_display', self._pbonds, d, int32)
    displays = property(get_display, set_display)

    def get_halfbond(self):
        return get_value('pseudobond_halfbond', self._pbonds, npy_bool)
    def set_halfbond(self, d):
        set_value('set_pseudobond_halfbond', self._pbonds, d, npy_bool)
    halfbonds = property(get_halfbond, set_halfbond)

    def get_radius(self):
        return get_value('pseudobond_radius', self._pbonds, float32)
    def set_radius(self, r):
        set_value('set_pseudobond_radius', self._pbonds, r, float32)
    radii = property(get_radius, set_radius)

# -----------------------------------------------------------------------------
#
class Residues:

    def __init__(self, residues_cpp):
        self._res = residues_cpp     # Numpy array of C++ pointers

    def __len__(self):
        return len(self._res)

    def __iter__(self):
        if not hasattr(self, '_res_list'):
            from .molobject import Residue, object_map
            self._res_list = [object_map(r,Residue) for r in self._res]
        return iter(self._res_list)

    def get_atoms(self):
        "atoms for all residues in one array"
        a = get_value('residue_atoms', self._res, cptr, self.num_atoms.sum(),
                      per_object = False)
        return Atoms(a)
    atoms = property(get_atoms, None)

    def get_num_atoms(self):
        "numpy array of atom count for each residue"
        return get_value('residue_num_atoms', self._res, int32)
    num_atoms = property(get_num_atoms, None)

    def get_chain_id(self):
        "numpy array of chain IDs"
        return get_value('residue_chain_id', self._res, string)
    chain_ids = property(get_chain_id, None)

    def get_molecule(self):
        "numpy array of molecule for each residue"
        mp = get_value('residue_molecule', self._res, cptr)
        return AtomicStructures(mp)
    molecules = property(get_molecule, None)

    def get_name(self):
        "numpy array of residue names"
        return get_value('residue_name', self._res, string)
    names = property(get_name, None)

    def get_number(self):
        "numpy array of residue sequence numbers"
        return get_value('residue_number', self._res, int32)
    numbers = property(get_number, None)

    def get_str(self):
        "numpy array of human-friendly residue identifier strings"
        return get_value('residue_str', self._res, string)
    strs = property(get_str, None)

    def get_unique_id(self):
        "numpy array of integer ids unique for each chain and residue number"
        return get_value('residue_unique_id', self._res, int32)
    unique_ids = property(get_unique_id, None)

# -----------------------------------------------------------------------------
#
class Chains:

    def __init__(self, chains_cpp):
        self._chains = chains_cpp     # Numpy array of C++ pointers

    def __len__(self):
        return len(self._chains)

    def __iter__(self):
        if not hasattr(self, '_chains_list'):
            from .molobject import Chain, object_map
            self._chains_list = [object_map(r,Chain) for r in self._chains]
        return iter(self._chains_list)

    def get_chain_id(self):
        return get_value('chain_chain_id', self._chains, string)
    chain_ids = property(get_chain_id, None)

    def get_molecule(self):
        mp = get_value('chain_molecule', self._chains, cptr)
        return AtomicStructures(mp)
    molecule = property(get_molecule, None)

    def get_residues(self):
        "residues for all chains in one array"
        r = get_value('chain_residues', self._chains, cptr, self.num_residues.sum(),
                      per_object = False)
        from .molarray import Residues
        return Residues(r)
    residues = property(get_residues, None)

    def get_num_residues(self):
        return get_value('chain_num_residues', self._chains, int32)
    num_residues = property(get_num_residues, None)

# -----------------------------------------------------------------------------
#
class AtomicStructures:

    def __init__(self, mols_cpp):
        self._mols = mols_cpp     # Numpy array of C++ pointers

    def __len__(self):
        return len(self._mols)

    def __iter__(self):
        if not hasattr(self, '_mols_list'):
            from .molobject import Chain, object_map
            self._mols_list = [object_map(r,Chain) for r in self._mols]
        return iter(self._mols_list)

    def get_name(self):
        return get_value('molecule_name', self._mols, string)
    def set_name(self):
        set_value('set_molecule_name', self._mols, string)
    names = property(get_name, set_name)

    def get_atoms(self):
        "atoms for all molecules in one array"
        a = get_value('molecule_atoms', self._mols, cptr, self.num_atoms.sum(),
                      per_object = False)
        from .molarray import Atoms
        return Atoms(a)
    atoms = property(get_atoms, None)

    def get_num_atoms(self):
        return get_value('molecule_num_atoms', self._mols, int32)
    num_atoms = property(get_num_atoms, None)

    def get_bonds(self):
        "bonds for all molecules in one array"
        b = get_value('molecule_bonds', self._mols, cptr, self.num_bonds.sum(),
                      per_object = False)
        from .molarray import Bonds
        return Bonds(b)
    bonds = property(get_bonds, None)

    def get_num_bonds(self):
        return get_value('molecule_num_bonds', self._mols, int32)
    num_bonds = property(get_num_bonds, None)

    def get_residues(self):
        "residues for all molecules in one array"
        r = get_value('molecule_residues', self._mols, cptr, self.num_residues.sum(),
                      per_object = False)
        from .molarray import Residues
        return Residues(r)
    residues = property(get_residues, None)

    def get_num_residues(self):
        return get_value('molecule_num_residues', self._mols, int32)
    num_residues = property(get_num_residues, None)

    def get_chains(self):
        "chains for all molecules in one array"
        c = get_value('molecule_chains', self._mols, cptr, self.num_chains,
                      per_object = False)
        from .molarray import Residues
        return Chains(c)
    chains = property(get_chains, None)

    def get_num_chains(self):
        return get_value('molecule_num_chains', self._mols, int32)
    num_chains = property(get_num_chains, None)
