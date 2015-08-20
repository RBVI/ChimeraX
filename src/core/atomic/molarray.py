'''
molarray: Collections of molecular objects
==========================================

These classes Atoms, Bonds, Residues... provide access to collections of C++
molecular data objects.  One typically gets these from an :py:class:`.AtomicStructure`
object which is produced by reading a PDB file.

Data items in a collections are ordered and the same object may be repeated.

Collections have attributes such as Atoms.coords that return a numpy array of
values for each object in the collection.  This offers better performance
than using a Python list of atoms since it directly accesses the C++ atomic data.
When using a list of atoms, a Python :py:class:`Atom` object is created for each
atom which requires much more memory and is slower to use in computation.
Working with lists is still often desirable when computations are not easily
done using arrays of attributes.  To get a list of atoms use list(x) where
x is an Atoms collection. Collections behave as Python iterators so constructs
such as a "for" loop over an Atoms collection is valid: "for a in atoms: ...".

There are collections Atoms, Bonds, Pseudobonds, Residues, Chains, AtomicStructureDatas.

Some attributes return collections instead of numpy arrays.  For example,
atoms.residues returns a Residues collection that has one residue for each atom
in the collection atoms.  If only a collection unique residues are desired,
use atoms.unique_residues.

Collections have base class PointerArray which provides many standard methods
such as length, iteration, indexing with square brackets, index of an element,
intersections, unions, subtraction, filtering....

Collections are immutable so can be hashed.  The only case in which their contents
can be altered is if C++ objects they hold are deleted in which case those objects
are automatically removed from the collection.
'''
from numpy import uint8, int32, float64, float32, bool as npy_bool, integer, empty, unique, array
from .molc import string, cptr, pyobject, cvec_property, set_cvec_pointer, c_function, pointer, ctype_type_to_numpy
from . import molobject
import ctypes
size_t = ctype_type_to_numpy[ctypes.c_size_t]   # numpy dtype for size_t

def _atoms(a):
    return Atoms(a)
def _bonds(a):
    return Bonds(a)
def _residues(a):
    return Residues(a)
def _chains(a):
    return Chains(a)
def _atomic_structures(p):
    return AtomicStructureDatas(p)
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
    '''
    Base class of all molecular data collections that provides common
    methods such as length, iteration, indexing with square brackets,
    intersection, union, subtracting, and filtering.
    '''
    def __init__(self, pointers, object_class, objects_class):
        if pointers is None:
            # Empty Atoms
            import numpy
            pointers = numpy.empty((0,), cptr)
        self._pointers = pointers
        self._object_class = object_class
        self._objects_class = objects_class
        set_cvec_pointer(self, pointers)
        remove_deleted_pointers(pointers)

    def __eq__(self, atoms):
        return (atoms._pointers == self._pointers).all()
    def hash(self):
        from hashlib import sha1
        return sha1(self._pointers.view(uint8)).digest()
    def __len__(self):
        return len(self._pointers)
    def __iter__(self):
        if not hasattr(self, '_object_list'):
            from .molobject import object_map
            c = self._object_class
            self._object_list = [object_map(p,c) for p in self._pointers]
        return iter(self._object_list)
    def __getitem__(self, i):
        if not isinstance(i,(int,integer)):
            raise IndexError('Only integer indices allowed for Atoms, got %s' % str(type(i)))
        from .molobject import object_map
        return object_map(self._pointers[i], self._object_class)
    def index(self, object):
        '''Find the position of the first occurence of an object in a collection.'''
        f = c_function('pointer_index',
                       args = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p],
                       ret = ctypes.c_ssize_t)
        i = f(self._c_pointers, len(self), object._c_pointer)
        return i

    def __or__(self, objects):
        '''The or operator | takes the union of two collections removing duplicates.'''
        return self.merge(objects)
    def __and__(self, objects):
        '''The and operator & takes the intersection of two collections removing duplicates.'''
        return self.intersect(objects)
    def __sub__(self, objects):
        '''The subtract operator "-" subtracts one collection from another as sets,
        eliminating all duplicates.'''
        return self.subtract(objects)

    def intersect(self, objects):
        import numpy
        return self._objects_class(numpy.intersect1d(self._pointers, objects._pointers))
    def intersects(self, objects):
        f = c_function('pointer_intersects',
                       args = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t],
                       ret = ctypes.c_bool)
        return f(self._c_pointers, len(self), objects._c_pointers, len(objects))
    def intersects_each(self, objects_list):
        '''Check if each of serveral pointer arrays intersects this array.
        Return a boolean array of length equal to the length of objects_list.
        '''
        f = c_function('pointer_intersects_each',
                       args = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p,
                               ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p])
        sizes = array(tuple(len(a) for a in objects_list), size_t)
        arrays = array(tuple(a._pointers.ctypes.data_as(ctypes.c_void_p).value for a in objects_list), cptr)
        n = len(objects_list)
        iarray = empty((n,), npy_bool)
        f(pointer(arrays), n, pointer(sizes), self._c_pointers, len(self), pointer(iarray))
        return iarray
    def filter(self, mask):
        '''Return a subset of the collection as a new collection.

        Parameters
        ----------
        mask : numpy bool array
          Array length must match the length of the collection.
        '''
        return self._objects_class(self._pointers[mask])
    def mask(self, objects):
        '''Return bool array indicating for each object in current set whether that
        object appears in the argument objects.'''
        f = c_function('pointer_mask', args = [ctypes.c_void_p, ctypes.c_size_t,
                                               ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p])
        mask = empty((len(self),), npy_bool)
        f(self._c_pointers, len(self), objects._c_pointers, len(objects), pointer(mask))
        return mask
    def merge(self, objects):
        import numpy
        return self._objects_class(numpy.union1d(self._pointers, objects._pointers))
    def subtract(self, objects):
        import numpy
        return self._objects_class(numpy.setdiff1d(self._pointers, objects._pointers))

def concatenate(pointer_arrays):
    '''Concatenate any number of collections returning a new collection.
    All collections must have the same type.
    
    Parameters
    ----------
    pointer_arrays : sequence of PointerArray objects
    '''
    cl = pointer_arrays[0]._objects_class
    from numpy import concatenate as concat
    c = cl(concat([a._pointers for a in pointer_arrays]))
    return c

# -----------------------------------------------------------------------------
#
class Atoms(PointerArray):
    '''
    An ordered collection of atom objects. This offers better performance
    than using a list of atoms.  It provides methods to access atom attributes such
    as coordinates as numpy arrays. Atoms directly accesses the C++ atomic data
    without creating Python :py:class:`Atom` objects which require much more memory
    and are slower to use in computation.
    '''
    def __init__(self, atom_pointers = None):
        PointerArray.__init__(self, atom_pointers, molobject.Atom, Atoms)

    bfactors = cvec_property('atom_bfactor', float32)
    chain_ids = cvec_property('atom_chain_id', string, read_only = True)
    colors = cvec_property('atom_color', uint8, 4)
    '''
    Returns a :mod:`numpy` Nx4 array of uint8 RGBA values. Can be
    set with such an array (or equivalent sequence), or with a single
    RGBA value.
    '''
    coords = cvec_property('atom_coord', float64, 3)
    '''Returns a :mod:`numpy` Nx3 array of XYZ values. Can be set.'''
    displays = cvec_property('atom_display', npy_bool)
    '''
    Controls whether the Atoms should be displayed.
    Returns a :mod:`numpy` array of boolean values.  Can be
    set with such an array (or equivalent sequence), or with a\n"
    single boolean value.
    '''
    draw_modes = cvec_property('atom_draw_mode', int32)
    '''
    Controls how the Atoms should be depicted, *e.g.* sphere,
    ball, *etc.*  The values are integers, SPHERE_STYLE, BALL_STYLE
    or STICK_STYLE as documented in the :class:`.Atom` class.
    Returns a :mod:`numpy` array of integers.  Can be
    set with such an array (or equivalent sequence), or with a
    single integer value.
    '''
    element_names = cvec_property('atom_element_name', string, read_only = True)
    '''Returns a numpy array of chemical element names. Read only.'''
    element_numbers = cvec_property('atom_element_number', uint8, read_only = True)
    '''Returns a :mod:`numpy` array of atomic numbers (integers). Read only.'''
    in_chains = cvec_property('atom_in_chain', npy_bool, read_only = True)
    structures = cvec_property('atom_structure', cptr, astype = _atomic_structures, read_only = True)
    names = cvec_property('atom_name', string, read_only = True)
    '''Returns a numpy array of atom names. Read only.'''
    radii = cvec_property('atom_radius', float32)
    '''
    Returns a :mod:`numpy` array of atomic radii.  Can be
    set with such an array (or equivalent sequence), or with a single
    floating-point number.
    '''
    residues = cvec_property('atom_residue', cptr, astype = _residues, read_only = True)
    '''
    Returns a :class:`Residues` whose data items
    correspond in a 1-to-1 fashion with the items in the Atoms.
    Read only. 
    '''
    selected = cvec_property('atom_selected', npy_bool)

    @property
    def num_selected(self):
        f = c_function('atom_num_selected', args = [ctypes.c_void_p, ctypes.c_size_t], ret = ctypes.c_size_t)
        return f(self._c_pointers, len(self))

    @property
    def unique_structures(self):
        return AtomicStructureDatas(unique(self.structures._pointers))

    @property
    def unique_residues(self):
        '''The unique :class:`Residues` for these atoms.'''
        return Residues(unique(self.residues._pointers))

    @property
    def by_structure(self):
        '''Return list of pairs of structure and Atoms for that structure.'''
        amol = self.structures._pointers
        return [(m, self.filter(amol==m._c_pointer.value)) for m in self.unique_structures]

    @property
    def by_chain(self):
        '''Return list of triples of structure, chain id, and Atoms for each chain.'''
        chains = []
        for m, atoms in self.by_structure:
            r = atoms.residues
            cids = r.chain_ids
            for cid in r.unique_chain_ids:
                chains.append((m, cid, atoms.filter(cids == cid)))
        return chains

    @property
    def scene_coords(self):
        n = len(self)
        from numpy import array, empty, float64
        xyz = empty((n,3), float64)
        if n == 0:
            return xyz
        mols = self.unique_structures
        mtable = array(tuple(m.scene_position.matrix for m in mols), float64)
        from .molc import pointer
        f = c_function('atom_scene_coords', args = [ctypes.c_void_p, ctypes.c_size_t,
                                                    ctypes.c_void_p, ctypes.c_size_t,
                                                    ctypes.c_void_p, ctypes.c_void_p])
        f(self._c_pointers, n, mols._c_pointers, len(mols), pointer(mtable), pointer(xyz))
        return xyz

    def delete(self):
        '''Delete the C++ Atom objects'''
        mols = self.unique_structures
        c_function('atom_delete', args = [ctypes.c_void_p, ctypes.c_size_t])(self._c_pointers, len(self))

# -----------------------------------------------------------------------------
#
class Bonds(PointerArray):
    '''Collection of C++ bonds.'''
    def __init__(self, bond_pointers):
        PointerArray.__init__(self, bond_pointers, molobject.Bond, Bonds)

    atoms = cvec_property('bond_atoms', cptr, 2, astype = _atoms_pair, read_only = True)
    '''
    Returns a two-tuple of :class:`Atoms` objects.
    For each bond, its endpoint atoms are in the matching
    position in the two :class:`Atoms` collections. Read only.  
    '''
    colors = cvec_property('bond_color', uint8, 4)
    '''
    Returns a :mod:`numpy` Nx4 array of uint8 RGBA values.  Can be
    set with such an array (or equivalent sequence), or with a single
    RGBA value.
    '''
    displays = cvec_property('bond_display', int32)
    '''
    Controls whether the Bonds should be displayed.
    The values are integers defined in the :class:`.Bond` class.
    TODO: No values are defined and value not used for rendering.
    Returns a :mod:`numpy` array of integers.  Can be
    set with such an array (or equivalent sequence), or with a
    single integer value.
    '''
    halfbonds = cvec_property('bond_halfbond', npy_bool)
    '''
    Controls whether the Bonds should be colored in "halfbond"
    mode, *i.e.* each half colored the same as its endpoint atom.
    Returns a :mod:`numpy` array of boolean values.  Can be
    set with such an array (or equivalent sequence), or with a
    single boolean value.
    '''
    radii = cvec_property('bond_radius', float32)
    '''
    Returns a :mod:`numpy` array of bond radii (half thicknesses).
    Can be set with such an array (or equivalent sequence), or with a
    single floating-point number.
    '''

# -----------------------------------------------------------------------------
#
class Pseudobonds(PointerArray):
    '''
    Holds a collection of C++ PBonds (pseudobonds) and provides access to some of
    their attributes. It has the same attributes as the
    :class:`Bonds` class and works in an analogous fashion.
    '''
    def __init__(self, pbond_pointers):
        PointerArray.__init__(self, pbond_pointers, molobject.Pseudobond, Pseudobonds)

    atoms = cvec_property('pseudobond_atoms', cptr, 2, astype = _atoms_pair, read_only = True)
    colors = cvec_property('pseudobond_color', uint8, 4)
    displays = cvec_property('pseudobond_display', int32)
    halfbonds = cvec_property('pseudobond_halfbond', npy_bool)
    radii = cvec_property('pseudobond_radius', float32)

# -----------------------------------------------------------------------------
#
class Residues(PointerArray):
    '''Collection of C++ residues.'''
    def __init__(self, residue_pointers):
        PointerArray.__init__(self, residue_pointers, molobject.Residue, Residues)

    atoms = cvec_property('residue_atoms', cptr, 'num_atoms', astype = _atoms, read_only = True, per_object = False)
    chain_ids = cvec_property('residue_chain_id', string, read_only = True)
    '''Returns a list of chain IDs. Read only.'''
    is_helix = cvec_property('residue_is_helix', npy_bool)
    is_sheet = cvec_property('residue_is_sheet', npy_bool)
    structures = cvec_property('residue_structure', cptr, astype = _atomic_structures, read_only = True)
    names = cvec_property('residue_name', string, read_only = True)
    '''Returns a list of residue names. Read only.'''
    num_atoms = cvec_property('residue_num_atoms', size_t, read_only = True)
    numbers = cvec_property('residue_number', int32, read_only = True)
    '''
    Returns a :mod:`numpy` array of residue sequence numbers, as provided by
    whatever data source the structure came from, so not necessarily consecutive,
    or starting from 1, *etc.* Read only.
    '''
    ss_id = cvec_property('residue_ss_id', int32)
    strs = cvec_property('residue_str', string, read_only = True)
    '''
    Returns a list of strings that encapsulates each
    residue's name, sequence position, and chain ID in a readable
    form. Read only.
    '''
    unique_ids = cvec_property('residue_unique_id', int32, read_only = True)
    '''
    A :mod:`numpy` array of integers. Multiple copies of the same residue
    in the collection will have the same integer value in the returned array.
    Read only.
    '''
    ribbon_displays = cvec_property('residue_ribbon_display', npy_bool)
    ribbon_colors = cvec_property('residue_ribbon_color', uint8, 4)

    @property
    def unique_structures(self):
        return AtomicStructureDatas(unique(self.structures._pointers))

    @property
    def unique_chain_ids(self):
        return unique(self.chain_ids)

# -----------------------------------------------------------------------------
#
class Chains(PointerArray):

    def __init__(self, chain_pointers):
        PointerArray.__init__(self, chain_pointers, molobject.Chain, Chains)

    chain_ids = cvec_property('chain_chain_id', string, read_only = True)
    structures = cvec_property('chain_structure', cptr, astype = _atomic_structures, read_only = True)
    residues = cvec_property('chain_residues', cptr, 'num_residues', astype = _residues,
                             read_only = True, per_object = False)
    num_residues = cvec_property('chain_num_residues', size_t, read_only = True)

# -----------------------------------------------------------------------------
#
class AtomicStructureDatas(PointerArray):
    '''
    Collection of C++ atomic structures.
    '''
    def __init__(self, mol_pointers):
        PointerArray.__init__(self, mol_pointers, molobject.AtomicStructureData, AtomicStructureDatas)

    atoms = cvec_property('structure_atoms', cptr, 'num_atoms', astype = _atoms,
                          read_only = True, per_object = False)
    '''Returns one :class:`Atoms` for all structures. Read only.'''
    bonds = cvec_property('structure_bonds', cptr, 'num_bonds', astype = _bonds,
                          read_only = True, per_object = False)
    '''Returns one :class:`Bonds` object for all structures. Read only.'''
    chains = cvec_property('structure_chains', cptr, 'num_chains', astype = _chains,
                           read_only = True, per_object = False)
    gc_color = cvec_property('structure_gc_color', npy_bool)
    gc_select = cvec_property('structure_gc_select', npy_bool)
    gc_shape = cvec_property('structure_gc_shape', npy_bool)
    names = cvec_property('structure_name', string)
    num_atoms = cvec_property('structure_num_atoms', size_t, read_only = True)
    '''Returns the number of atoms in each AtomicStructure. Read only.'''
    num_bonds = cvec_property('structure_num_bonds', size_t, read_only = True)
    '''Returns the number of bonds in each AtomicStructure. Read only.'''
    num_chains = cvec_property('structure_num_chains', size_t, read_only = True)
    '''Returns the number of chains in each AtomicStructure. Read only.'''
    num_residues = cvec_property('structure_num_residues', size_t, read_only = True)
    '''Returns the number of residues in each AtomicStructure. Read only.'''
    residues = cvec_property('structure_residues', cptr, 'num_residues', astype = _residues,
                             read_only = True, per_object = False)
    '''Returns one :class:`Residues` object for all structures. Read only.'''
    pbg_maps = cvec_property('structure_pbg_map', pyobject, astype = _pseudobond_group_map, read_only = True)
    '''
    Returns a list of dictionaries whose keys are pseudobond
    group categories (strings) and whose values are
    :class:`.Pseudobonds`. Read only.
    '''

# -----------------------------------------------------------------------------
# When C++ object is deleted, delete it from the specified pointer array.
#
def remove_deleted_pointers(array):
    remove_deleted_c_pointers(array)
    import weakref
    weakref.finalize(array, pointer_array_freed, id(array))

remove_deleted_c_pointers = c_function('remove_deleted_c_pointers', args = [ctypes.py_object])
pointer_array_freed = c_function('pointer_array_freed', args = [ctypes.c_void_p])
