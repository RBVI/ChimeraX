# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2022 Regents of the University of California. All rights reserved.
# This software is provided pursuant to the ChimeraX license agreement, which
# covers academic and commercial uses. For more information, see
# <http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
#
# This file is part of the ChimeraX library. You can also redistribute and/or
# modify it under the GNU Lesser General Public License version 2.1 as
# published by the Free Software Foundation. For more details, see
# <https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html>
#
# This file is distributed WITHOUT ANY WARRANTY; without even the implied
# warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. This notice
# must be embedded in or attached to all copies, including partial copies, of
# the software or any revisions or derivations thereof.
# === UCSF ChimeraX Copyright ===

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

There are collections Atoms, Bonds, Pseudobonds, Residues, Chains, AtomicStructures.

Some attributes return collections instead of numpy arrays.  For example,
atoms.residues returns a Residues collection that has one residue for each atom
in the collection atoms.  If only a collection unique residues are desired,
use atoms.unique_residues.

Collections have base class :class:`.Collection` which provides many standard methods
such as length, iteration, indexing with square brackets, index of an element,
intersections, unions, subtraction, filtering....

Collections are mostly immutable.  The only case in which their contents
can be altered is if C++ objects they hold are deleted in which case those objects
are automatically removed from the collection.  Because they are mutable they
cannot be used as keys in dictionary or added to sets.
'''
from numpy import uint8, int32, uint32, float64, float32, uintp, byte, integer, empty, array
npy_bool = bool
from .molc import string, cptr, pyobject, set_cvec_pointer, pointer, size_t
from . import molobject
from .molobject import c_function, c_array_function, cvec_property, Atom
import ctypes

def _atoms(p):
    return Atoms(p)
def _atoms_or_nones(p):
    return [Atom.c_ptr_to_py_inst(ptr) if ptr else None for ptr in p]
def _non_null_atoms(p):
    return Atoms(p[p!=0])
def _bonds(p):
    return Bonds(p)
def _coordsets(p):
    return CoordSets(p)
def _pseudobond_groups(p):
    return PseudobondGroups(p)
def _pseudobonds(p):
    return Pseudobonds(p)
def _elements(p):
    return Elements(p)
def _residues(p):
    return Residues(p)
def _non_null_residues(p):
    return Residues(p[p!=0])
def _chains(p):
    return Chains(p)
def _non_null_chains(p):
    return Chains(p[p!=0])
def _atomic_structures(p):
    return AtomicStructures(p)
def structure_datas(p):
    return StructureDatas(p)
def _atoms_pair(p):
    return (Atoms(p[:,0].copy()), Atoms(p[:,1].copy()))
def _pseudobond_group_map(a):
    from . import molobject
    return [molobject._pseudobond_group_map(p) for p in a]

# -----------------------------------------------------------------------------
#
from chimerax.core.state import State
class Collection(State):
    '''
    Base class of all molecular data collections that provides common
    methods such as length, iteration, indexing with square brackets,
    intersection, union, subtracting, and filtering.  By design, a
    Collection is immutable except that deleted items are automatically
    removed.
    '''
    def __init__(self, items, object_class):
        import numpy
        if items is None:
            # Empty
            pointers = numpy.empty((0,), cptr)
        elif isinstance(items, numpy.ndarray) and items.dtype == numpy.uintp:
            # C++ pointers array
            pointers = numpy.ascontiguousarray(items)
        else:
            # presume iterable of objects of the object_class
            try:
                pointers = numpy.array([i._c_pointer.value for i in items], cptr)
            except Exception:
                t = str(type(items))
                if isinstance(items, numpy.ndarray):
                    t += ' type %s' % str(items.dtype)
                raise ValueError('Collection items of unrecognized type "%s"' % t)
        self._pointers = pointers
        self._lookup = None		# LookupTable for optimized intersections
        self._object_class = object_class
        set_cvec_pointer(self, pointers)
        remove_deleted_pointers(pointers)

    def __eq__(self, items):
        if not isinstance(items, Collection):
            return False
        import numpy
        return numpy.array_equal(items._pointers, self._pointers)

    def hash(self):
        '''
        Can be used for quickly determining if collections have the same elements in the same order.
        Objects are automatically deleted from the collection when the C++ object is deleted.
        So this hash value will not be valid if the collection changes.  This is not the __hash__
        special Python method and it is not supported to use collections as keys of dictionaries
        or elements of sets since they are mutable (deletions automatically remove items).
        '''
        from hashlib import sha1
        return sha1(self._pointers.view(uint8)).digest()
    def __len__(self):
        '''Number of objects in collection.'''
        return len(self._pointers)
    def __bool__(self):
        return len(self) > 0
    def __iter__(self):
        '''Iterator over collection objects.'''
        if not hasattr(self, '_object_list') or len(self._object_list) > len(self._pointers):
            c = self._object_class
            self._object_list = [c.c_ptr_to_py_inst(p) for p in self._pointers]
        return iter(self._object_list)
    def __getitem__(self, i):
        '''Indexing of collection objects using square brackets, *e.g.* c[i].'''
        import numpy
        if isinstance(i,(int,integer)):
            v = self._object_class.c_ptr_to_py_inst(self._pointers[i])
        elif isinstance(i, (slice, numpy.ndarray)):
            v = self.__class__(self._pointers[i])
        else:
            raise IndexError('Only integer indices allowed for %s, got %s'
                % (self.__class__.__name__, str(type(i))))
        return v
    def index(self, object):
        '''Find the position of the first occurence of an object in a collection.'''
        f = c_function('pointer_index',
                       args = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p],
                       ret = ctypes.c_ssize_t)
        i = f(self._c_pointers, len(self), object._c_pointer)
        return i
    def indices(self, objects):
        '''Return int32 array indicating for each element in objects its index of the
        first occurence in the collection, or -1 if it does not occur in the collection.'''
        return self._lookup_table.indices(objects)

    @property
    def object_class(self):
        return self._object_class
    @property
    def objects_class(self):
        return self.__class__
    @property
    def pointers(self):
        return self._pointers

    @property
    def _lookup_table(self):
        if self._lookup is None:
            self._lookup = LookupTable(self)
        return self._lookup
    
    def __or__(self, objects):
        '''The or operator | takes the union of two collections removing duplicates.'''
        return self.merge(objects)
    def __and__(self, objects):
        '''The and operator & takes the intersection of two collections removing duplicates.'''
        return self.intersect(objects)
    def __add__(self, objects):
        '''The addition operator "+" returns a new collection containing all the items
        from the collections being added.  Duplicates are not removed.'''
        if self.__class__ != objects.__class__:
            raise TypeError("Cannot add different Collection subclasses")
        return concatenate((self, objects))
    def __sub__(self, objects):
        '''The subtract operator "-" subtracts one collection from another as sets,
        eliminating all duplicates.'''
        return self.subtract(objects)

    def copy(self):
        '''Shallow copy, since Collections are immutable.'''
        return self.__class__(self._pointers)

    def intersect(self, objects):
        '''Return a new collection that is the intersection with the *objects* :class:`.Collection`.'''
        pointers = self.pointers[self.mask(objects)]
        return self.__class__(pointers)
    def intersects(self, objects):
        '''Whether this collection has any element in common
        with the *objects* :class:`.Collection`. Returns bool.'''
        return objects._lookup_table.includes_any(self)
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
    def filter(self, mask_or_indices):
        '''Return a subset of the collection as a new collection.

        Parameters
        ----------
        mask_or_indices : numpy bool array (mask) or int array (indices)
          Bool length must match the length of the collection and filters out items where
          the bool array is False.
        '''
        return self.__class__(self._pointers[mask_or_indices])
    def mask(self, objects):
        '''Return bool array indicating for each object in current set whether that
        object appears in the argument objects.'''
        return objects._lookup_table.includes_each(self)
    def merge(self, objects):
        '''Return a new collection combining this one with the *objects* :class:`.Collection`.
        All duplicates are removed.'''
        import numpy
        return self.__class__(numpy.union1d(self._pointers, objects._pointers))
    def subtract(self, objects):
        '''Return a new collection subtracting the *objects* :class:`.Collection` from this one.
        All duplicates are removed.  Currently does not preserve order'''
        import numpy
        return self.__class__(numpy.setdiff1d(self._pointers, objects._pointers))
    def unique(self):
        '''Return a new collection containing the unique elements from this one, preserving order.'''
        return self.__class__(unique_ordered(self._pointers))
    def instances(self, instantiate=True):
        '''Returns a list of the Python instances.  If 'instantiate' is False, then for
        those items that haven't yet been instantiated, None will be returned.'''
        if instantiate:
            return [self._object_class.c_ptr_to_py_inst(p) for p in self._pointers]
        return [self._object_class.c_ptr_to_existing_py_inst(p) for p in self._pointers]
    STATE_VERSION = 1
    def take_snapshot(self, session, flags):
        return {'version': self.STATE_VERSION,
                'pointers': self.session_save_pointers(session)}
    @classmethod
    def restore_snapshot(cls, session, data):
        if data['version'] > cls.STATE_VERSION:
            raise ValueError("Don't know how to restore Collections from this session"
                             " (session version [{}] > code version [{}]);"
                             " update your ChimeraX".format(data['version'], self.STATE_VERSION))
        c_pointers = cls.session_restore_pointers(session, data['pointers'])
        return cls(c_pointers)
    @classmethod
    def session_restore_pointers(cls, session, data):
        raise NotImplementedError(
            self.__class__.__name__ + " has not implemented session_restore_pointers")
    def session_save_pointers(self, session):
        raise NotImplementedError(
            self.__class__.__name__ + " has not implemented session_save_pointers")

class LookupTable:
    '''C++ set of pointers for fast lookup.'''
    def __init__(self, collection):
        self._collection = collection
        self._cpp_table = None		# Pointer to C++ set
        self._size = None		# Size of C++ set

    def __del__(self):
        self._delete_table()

    def _delete_table(self):
        if self._cpp_table:
            delete_table = c_function('pointer_table_delete',
                                      args = [ctypes.c_void_p])
            delete_table(self._cpp_table)
            self._cpp_table = None
            self._size = None

    @property
    def _cpp_table_pointer(self):
        if self._cpp_table is None or len(self._collection) != self._size:
            self._delete_table()
            create_table = c_function('pointer_table_create',
                                      args = [ctypes.c_void_p, ctypes.c_size_t],
                                      ret = ctypes.c_void_p)
            c = self._collection
            self._size = s = len(c)
            self._cpp_table = create_table(c._c_pointers, s)
        return self._cpp_table
    
    def includes_any(self, collection):
        incl_any = c_function('pointer_table_includes_any',
                              args = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t],
                              ret = ctypes.c_bool)
        return incl_any(self._cpp_table_pointer, collection._c_pointers, len(collection))
    
    def includes_each(self, collection):
        incl_each = c_function('pointer_table_includes_each',
                               args = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p])
        n = len(collection)
        mask = empty((n,), npy_bool)
        incl_each(self._cpp_table_pointer, collection._c_pointers, n, pointer(mask))
        return mask

    def indices(self, collection):
        indices = c_function('pointer_table_indices',
                             args = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p])
        n = len(collection)
        ind = empty((n,), int32)
        indices(self._cpp_table_pointer, collection._c_pointers, n, pointer(ind))
        return ind
    
def concatenate(collections, object_class = None, remove_duplicates = False):
    '''Concatenate any number of collections returning a new collection.
    All collections must have the same type.

    Parameters
    ----------
    collections : sequence of :class:`.Collection` objects
    '''
    if len(collections) == 0:
        c = object_class(None)
    else:
        import numpy
        p = numpy.concatenate([a._pointers for a in collections])
        if remove_duplicates:
            p = unique_ordered(p)    # Preserve order when duplicates are removed.
        cl = collections[0].__class__ if object_class is None else object_class
        c = cl(p)
    return c

def unique_ordered(a):
    '''Return unique elements of numpy array a preserving order.'''
    from numpy import unique
    indices = unique(a, return_index = True)[1]
    indices.sort()
    return a[indices]

def depluralize(word):
    if word.endswith('ii'):
        return word[:-2] + 'ius'
    if word.endswith('ses'):
        return word[:-2]
    if word.endswith('s'):
        return word[:-1]
    return word

# -----------------------------------------------------------------------------
#
class StructureDatas(Collection):
    '''
    Collection of C++ atomic structure objects.
    '''
    def __init__(self, mol_pointers):
        Collection.__init__(self, mol_pointers, molobject.StructureData)

    alt_loc_change_notifies = cvec_property('structure_alt_loc_change_notify', npy_bool)
    '''Whether notifications are issued when altlocs are changed.  Should only be
    set to true when temporarily changing alt locs in a Python script. Numpy bool array.'''
    ss_change_notifies = cvec_property('structure_ss_change_notify', npy_bool)
    '''Whether notifications are issued when secondary structure is changed.  Should only be
    set to true when temporarily changing secondary structure in a Python script. Numpy bool array.'''
    active_coordsets = cvec_property('structure_active_coordset', cptr, astype = _coordsets,
        read_only = True,
        doc="Returns a :class:`CoordSets` of the active coordset of each structure. Read only.")
    atoms = cvec_property('structure_atoms', cptr, 'num_atoms', astype = _atoms,
                          read_only = True, per_object = False)
    '''A single :class:`.Atoms` containing atoms for all structures. Read only.'''
    bonds = cvec_property('structure_bonds', cptr, 'num_bonds', astype = _bonds,
                          read_only = True, per_object = False)
    '''A single :class:`.Bonds` object containing bonds for all structures. Read only.'''
    chains = cvec_property('structure_chains', cptr, 'num_chains', astype = _chains,
                           read_only = True, per_object = False)
    '''A single :class:`.Chains` object containing chains for all structures. Read only.'''
    lower_case_chains = cvec_property('structure_lower_case_chains', npy_bool)
    '''A numpy bool array of lower_case_names of each structure.'''
    num_atoms = cvec_property('structure_num_atoms', size_t, read_only = True)
    '''Number of atoms in each structure. Read only.'''
    num_bonds = cvec_property('structure_num_bonds', size_t, read_only = True)
    '''Number of bonds in each structure. Read only.'''
    num_chains = cvec_property('structure_num_chains', size_t, read_only = True)
    '''Number of chains in each structure. Read only.'''
    num_residues = cvec_property('structure_num_residues', size_t, read_only = True)
    '''Number of residues in each structure. Read only.'''
    residues = cvec_property('structure_residues', cptr, 'num_residues', astype = _residues,
                             read_only = True, per_object = False)
    '''A single :class:`Residues` object containing residues for all structures. Read only.'''
    pbg_maps = cvec_property('structure_pbg_map', pyobject, astype = _pseudobond_group_map, read_only = True)
    '''
    Returns a list of dictionaries whose keys are pseudobond
    group categories (strings) and whose values are
    :class:`.Pseudobonds`. Read only.
    '''
    ribbon_tether_scales = cvec_property('structure_ribbon_tether_scale', float32)
    '''Returns an array of scale factors for ribbon tethers.'''
    ribbon_tether_sides = cvec_property('structure_ribbon_tether_sides', int32)
    '''Returns an array of numbers of sides for ribbon tethers.'''
    ribbon_tether_shapes = cvec_property('structure_ribbon_tether_shape', int32)
    '''Returns an array of shapes for ribbon tethers.'''
    metadata = cvec_property('metadata', pyobject, read_only = True)
    '''Return a list of dictionaries with metadata. Read only.'''
    def set_metadata_entry(self, key, values):
        """Set metadata dictionary entry"""
        n = len(self)
        f = c_array_function('set_metadata_entry', args=(pyobject, pyobject), per_object=False)
        f(self._c_pointers, n, key, values)
    ribbon_tether_opacities = cvec_property('structure_ribbon_tether_opacity', float32)
    '''Returns an array of opacity scale factor for ribbon tethers.'''
    ribbon_show_spines = cvec_property('structure_ribbon_show_spine', npy_bool)
    '''Returns an array of booleans of whether to show ribbon spines.'''
    ribbon_orientations = cvec_property('structure_ribbon_orientation', int32)
    '''Returns an array of ribbon orientations.'''
    ss_assigneds = cvec_property('structure_ss_assigned', npy_bool, doc =
    '''
    Whether secondary structure has been assigned, either from data in the
    original structure file, or from an algorithm (e.g. dssp command)
    ''')

    # Graphics changed flags used by rendering code.  Private.
    _graphics_changeds = cvec_property('structure_graphics_change', int32)

    @property
    def autochains(self):
        return array([s.autochain for s in self])
    
    @autochains.setter
    def autochains(self, ac):
        for s in self:
            s.autochain = ac

    @property
    def names(self):
        return array([s.name for s in self])

    @names.setter
    def names(self, nm):
        for s in self:
            s.name = nm

    # res_numbering can be int or string, so go through Python layer
    @property
    def res_numberings(self):
        return array([s.res_numbering for s in self])

    @res_numberings.setter
    def res_numberings(self, rn):
        for s in self:
            s.res_numbering = rn

# -----------------------------------------------------------------------------
#
class AtomicStructures(StructureDatas):
    '''
    Collection of Python atomic structure objects.
    '''
    def __init__(self, mol_pointers):
        from . import AtomicStructure
        Collection.__init__(self, mol_pointers, AtomicStructure)

    # so setattr knows that attr exists (used by selection inspector);
    # also, don't want to directly set Structure.display, want to go through Model.display
    @property
    def displays(self):
        return array([s.display for s in self])
    @displays.setter
    def displays(self, d):
        for s in self:
            s.display = d

    @property
    def visibles(self):
        return array([s.visible for s in self])

    @classmethod
    def session_restore_pointers(cls, session, data):
        return array([s._c_pointer.value for s in data], dtype=cptr)
    def session_save_pointers(self, session):
        return [s for s in self]

# -----------------------------------------------------------------------------
#
class Atoms(Collection):
    '''
    An ordered collection of atom objects. This offers better performance
    than using a list of atoms.  It provides methods to access atom attributes such
    as coordinates as numpy arrays. Atoms directly accesses the C++ atomic data
    without creating Python :py:class:`Atom` objects which require much more memory
    and are slower to use in computation.
    '''
    # replicate Atom class constants
    SPHERE_STYLE = Atom.SPHERE_STYLE
    BALL_STYLE = Atom.BALL_STYLE
    STICK_STYLE = Atom.STICK_STYLE
    HIDE_RIBBON = Atom.HIDE_RIBBON
    HIDE_ISOLDE = Atom.HIDE_ISOLDE
    HIDE_NUCLEOTIDE = Atom.HIDE_NUCLEOTIDE
    BBE_MIN = Atom.BBE_MIN
    BBE_RIBBON = Atom.BBE_RIBBON
    BBE_MAX = Atom.BBE_MAX

    bfactors = cvec_property('atom_bfactor', float32)
    bonds = cvec_property('atom_bonds', cptr, 'num_bonds', astype = _bonds, read_only = True,
        per_object = False,
        doc=":class:`Bonds` object where either endpoint atom is in this collection. If any of "
        "the atoms in this collection are bonded to each other, then there will be duplicate bonds "
        "in the result, so call .unique() on that if duplicates are problematic.")
    @property
    def by_chain(self):
        '''Return list of triples of structure, chain id, and Atoms for each chain.'''
        f = c_function('atom_by_chain', args = [ctypes.c_void_p, ctypes.c_size_t],
                       ret = ctypes.py_object)
        sca = f(self._c_pointers, len(self))
        abc = tuple((s, cid, Atoms(aptr)) for s, cid, aptr in sca)
        return abc
    @property
    def by_structure(self):
        "Return list of 2-tuples of (structure, Atoms for that structure)."
        astruct = self.structures._pointers
        return [(us, self.filter(astruct==us._c_pointer.value)) for us in self.unique_structures]
    chain_ids = cvec_property('atom_chain_id', string, read_only = True)
    colors = cvec_property('atom_color', uint8, 4,
        doc="Returns a :mod:`numpy` Nx4 array of uint8 RGBA values. Can be set "
        "with such an array (or equivalent sequence), or with a single RGBA value.")
    @property
    def average_ribbon_color(self):
        "Average ribbon color as length 4 unit8 RGBA values."
        f = c_function('atom_average_ribbon_color',
                       args = [ctypes.c_void_p, ctypes.c_size_t],
                       ret = ctypes.py_object)
        return f(self._c_pointers, len(self))
    coords = cvec_property('atom_coord', float64, 3,
        doc="Returns a :mod:`numpy` Nx3 array of XYZ values. Can be set.")
    coord_indices = cvec_property('atom_coord_index', uint32, read_only = True,
        doc="Coordinate index of atom in coordinate set.")
    displays = cvec_property('atom_display', npy_bool,
        doc="Controls whether the Atoms should be displayed. Returns a :mod:`numpy` array of "
        "boolean values.  Can be set with such an array (or equivalent sequence), or with a "
        "single boolean value.")
    draw_modes = cvec_property('atom_draw_mode', uint8,
        doc="Controls how the Atoms should be depicted. The values are integers, SPHERE_STYLE, "
        "BALL_STYLE or STICK_STYLE as documented in the :class:`.Atom` class. Returns a "
        ":mod:`numpy` array of integers.  Can be set with such an array (or equivalent sequence), "
        "or with a single integer value.")
    elements = cvec_property('atom_element', cptr, astype = _elements, read_only = True,
        doc="Returns a :class:`Elements` whose data items correspond in a 1-to-1 fashion with the "
        "items in the Atoms.  Read only.")
    element_names = cvec_property('atom_element_name', string, read_only = True,
        doc="Returns a numpy array of chemical element names. Read only.")
    element_numbers = cvec_property('atom_element_number', uint8, read_only = True,
        doc="Returns a :mod:`numpy` array of atomic numbers (integers). Read only.")
    hides = cvec_property('atom_hide', int32,
        doc="Whether atom is hidden (overrides display).  Returns a :mod:`numpy` array of int32 bitmask."
        "\n\nPossible values:\n\n"
        "HIDE_RIBBON\n"
        "    Hide mask for backbone atoms in ribbon.\n\n"
        "Can be set with such an array (or equivalent sequence), or with a single "
        "integer value.")

    def set_hide_bits(self, bit_mask):
        """Set Atom's hide bits in bit mask"""
        n = len(self)
        f = c_array_function('set_atom_hide_bits', args=(uint32,), per_object=False)
        f(self._c_pointers, n, bit_mask)

    def clear_hide_bits(self, bit_mask):
        """Clear Atom's hide bits in bit mask"""
        n = len(self)
        f = c_array_function('clear_atom_hide_bits', args=(uint32,), per_object=False)
        f(self._c_pointers, n, bit_mask)

    idatm_types = cvec_property('atom_idatm_type', string,
        doc="Returns a numpy array of IDATM types.  Can be set with such an array (or equivalent "
        "sequence), or with a single string.")
    in_chains = cvec_property('atom_in_chain', npy_bool, read_only = True,
        doc="Whether each atom belong to a polymer. Returns numpy bool array. Read only.")

    def is_backbones(self, bb_extent=molobject.Atom.BBE_MAX):
        n = len(self)
        values = empty((n,), npy_bool)
        f = c_array_function('atom_is_backbone', args=(int32,), ret=npy_bool, per_object=False)
        f(self._c_pointers, n, bb_extent, pointer(values))
        return values

    is_riboses = cvec_property('atom_is_ribose', npy_bool, read_only = True,
        doc="Whether each atom is part of an nucleic acid ribose moiety."
            " Returns numpy bool array. Read only.")
    is_side_chains = cvec_property('atom_is_side_chain', npy_bool, read_only = True,
        doc="Whether each atom is part of an amino/nucleic acid sidechain."
            " Includes atoms needed to connect to backbone (CA/ribose)."
            " Returns numpy bool array. Read only.")
    is_side_connectors = cvec_property('atom_is_side_connector', npy_bool, read_only = True,
        doc="Whether each atom is needed to connect to backbone (CA/ribose)."
            " Returns numpy bool array. Read only.")
    is_side_onlys = cvec_property('atom_is_side_only', npy_bool, read_only = True,
        doc="Whether each atom is part of an amino/nucleic acid sidechain."
            " Does not include atoms needed to connect to backbone (CA/ribose)."
            " Returns numpy bool array. Read only.")
    names = cvec_property('atom_name', string,
        doc="Returns a numpy array of atom names.  Can be set with such an array (or equivalent "
        "sequence), or with a single string.  Atom names are limited to 4 characters.")
    neighbors = cvec_property('atom_neighbors', cptr, 'num_bonds', astype = _atoms, read_only = True,
        per_object = False,
        doc=":class:`Atoms` object where each atom is bonded to an atom is in this collection. If any of "
        "the atoms in this collection are bonded to each other, then there will be duplicate atoms "
        "in the result, so call .unique() on that if duplicates are problematic.")
    num_alt_locs = cvec_property('atom_num_alt_locs', size_t, read_only = True)
    '''Number of alt locs in each atom.  Zero for atoms without alt locs.  Read only.'''
    num_bonds = cvec_property('atom_num_bonds', size_t, read_only = True)
    '''Number of bonds in each atom. Read only.'''
    @property
    def num_residues(self):
        "Total number of residues for atoms."
        f = c_function('atom_num_residues',
                       args = [ctypes.c_void_p, ctypes.c_size_t],
                       ret = ctypes.c_size_t)
        return f(self._c_pointers, len(self))
    occupancies = cvec_property('atom_occupancy', float32)

    @property
    def intra_bonds(self):
        ":class:`Bonds` object where both endpoint atoms are in this collection"
        f = c_function('atom_intra_bonds', args = [ctypes.c_void_p, ctypes.c_size_t],
            ret = ctypes.py_object)
        return _bonds(f(self._c_pointers, len(self)))

    from . import interatom_pseudobonds
    intra_pseudobonds = property(interatom_pseudobonds)

    radii = cvec_property('atom_radius', float32,
        doc="Returns a :mod:`numpy` array of radii.  Can be set with such an array (or equivalent "
        "sequence), or with a single floating-point number.")
    default_radii = cvec_property('atom_default_radius', float32, read_only = True,
        doc="Returns a :mod:`numpy` array of default radii.")
    def use_default_radii(self):
        f = c_function('atom_use_default_radius', args = [ctypes.c_void_p, ctypes.c_size_t])
        f(self._c_pointers, len(self))
    def display_radii(self, ball_scale, bond_radius):
        r = self.radii.copy()
        dm = self.draw_modes
        from . import Atom
        r[dm == Atom.BALL_STYLE] *= ball_scale
        smask = (dm == Atom.STICK_STYLE)
        if smask.any():
            r[smask] = self.filter(smask).maximum_bond_radii(bond_radius)
        return r
    def maximum_bond_radii(self, default_radius = 0.2):
        "Return maximum bond radius for each atom.  Used for stick style atom display."
        f = c_function('atom_maximum_bond_radius', args = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_float, ctypes.c_void_p])
        n = len(self)
        r = empty((n,), float32)
        f(self._c_pointers, n, default_radius, pointer(r))
        return r
    residues = cvec_property('atom_residue', cptr, astype = _residues, read_only = True,
        doc="Returns a :class:`Residues` whose data items correspond in a 1-to-1 fashion with the "
        "items in the Atoms.  Read only.")
    ribbon_coords = cvec_property('atom_ribbon_coord', float64, 3, doc =
        '''Returns a :mod:`numpy` Nx3 array of XYZ values.
        Raises error if any atom does nt have a ribbon coordinate.
        Can be set.''')
    effective_coords = cvec_property('atom_effective_coord', float64, 3, read_only=True,
        doc='''Returns a :mod:`numpy` Nx3 array of XYZ values.
        Return the atom's ribbon_coord if the residue is displayed as a ribbon and
        has a ribbon coordinate, otherwise return the current coordinate.
        ''')
    pb_coords = effective_coords
    effective_scene_coords = cvec_property('atom_effective_scene_coord', float64, 3, read_only=True,
        doc='''Returns a :mod:`numpy` Nx3 array of XYZ values.
        Return the atom's ribbon_coord if the residue is displayed as a ribbon and
        has a ribbon coordinate, otherwise return the current coordinate in scene coordinate system.
        ''')
    pb_scene_coords = effective_scene_coords
    @property
    def scene_bounds(self):
        "Return scene bounds of atoms including instances of all parent models."
        blist = []
        from chimerax.geometry import sphere_bounds, copy_tree_bounds, union_bounds
        for m, a in self.by_structure:
            ba = sphere_bounds(a.coords, a.radii)
            ib = copy_tree_bounds(ba,
                [d.positions for d in m.drawing_lineage])
            blist.append(ib)
        return union_bounds(blist)
    _scene_coords_tmp = cvec_property('atom_scene_coord', float64, 3, read_only = True)
    def _set_scene_coords(self, xyz):
        n = len(self)
        if n == 0: return
        if len(xyz) != n:
            raise ValueError('Atoms._set_scene_coords: wrong number of coords %d for %d atoms'
                             % (len(xyz), n))
        structs = self.unique_structures
        gtable = array(tuple(s.scene_position.inverse().matrix for s in structs), float64)
        from .molc import pointer
        f = c_function('atom_set_scene_coords',
            args = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p,
                    ctypes.c_size_t, ctypes.c_void_p, ctypes.c_void_p])
        f(self._c_pointers, n, structs._c_pointers, len(structs), pointer(gtable), pointer(xyz))
    scene_coords = property(_scene_coords_tmp.fget, _set_scene_coords, doc =
        '''Atoms' coordinates in the global scene coordinate system.
        This accounts for the :class:`Drawing` positions for the hierarchy
        of models each atom belongs to.''')
    selected = cvec_property('atom_selected', npy_bool,
        doc="numpy bool array whether each Atom is selected.")
    selecteds = selected
    @property
    def num_selected(self):
        "Number of selected atoms."
        f = c_function('atom_num_selected',
                       args = [ctypes.c_void_p, ctypes.c_size_t],
                       ret = ctypes.c_size_t)
        return f(self._c_pointers, len(self))
    has_selected_bonds = cvec_property('atom_has_selected_bond', npy_bool, read_only = True)
    '''For each atom is any connected bond selected.'''
    serial_numbers = cvec_property('atom_serial_number', int32, doc="Serial numbers of atoms")
    @property
    def shown_atoms(self):
        '''
        Subset of Atoms including atoms that are displayed and those that are hidden because
        the ribbon is displayed and have displayed structure and displayed parent models.
        '''
        ribbon_atoms = (self.residues.ribbon_displays & ((self.hides & self.HIDE_RIBBON) != 0))
        da = self.filter(self.displays | ribbon_atoms)
        datoms = concatenate([a for m, a in da.by_structure
                              if m.display and m.parents_displayed], Atoms)
        return datoms
    structure_categories = cvec_property('atom_structure_category', string, read_only=True,
        doc="Numpy array of whether atom is ligand, ion, etc.")
    structures = cvec_property('atom_structure', pyobject, astype = AtomicStructures, read_only=True,
        doc="Returns an :class:`AtomicStructure` for each atom. Read only.")
    def transform(self, place):
        f = c_function('atom_transform',
            args=(ctypes.c_void_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_double)))
        f(self._c_pointers, len(self), pointer(place.matrix))
        from .triggers import get_triggers
        get_triggers().activate_trigger("atoms transformed", (self, place))
    @property
    def unique_residues(self):
        '''The unique :class:`.Residues` for these atoms.'''
        return self.residues.unique()
    @property
    def unique_chain_ids(self):
        '''The unique chain IDs as a numpy array of strings.'''
        return unique_ordered(self.chain_ids)
    @property
    def unique_structures(self):
        "The unique structures as an :class:`.AtomicStructures` collection"
        return self.structures.unique()
    @property
    def full_residues(self):
        '''The :class:`.Residues` all of whose atoms are in this :class:`.Atoms` instance'''
        all_residues = self.unique_residues
        extra = (all_residues.atoms - self).unique_residues
        return all_residues - extra
    @property
    def full_structures(self):
        '''The :class:`.Structures` all of whose atoms are in this :class:`.Atoms` instance'''
        all_structures = self.unique_structures
        extra = (all_structures.atoms - self).unique_structures
        return all_structures - extra
    @property
    def single_structure(self):
        "Do all atoms belong to a single :class:`.Structure`"
        p = self.structures._pointers
        return len(p) == 0 or (p == p[0]).all()
    visibles = cvec_property('atom_visible', npy_bool, read_only=True,
        doc="Returns whether the Atom should be visible (displayed and not hidden). Returns a "
        ":mod:`numpy` array of boolean values.  Read only.")
    alt_locs = cvec_property('atom_alt_loc', string,
                         doc='Returns current alternate location indicators')

    def __init__(self, atom_pointers = None):
        Collection.__init__(self, atom_pointers, molobject.Atom)

    def delete(self):
        '''Delete the C++ Atom objects'''
        c_function('atom_delete',
            args = [ctypes.c_void_p, ctypes.c_size_t])(self._c_pointers, len(self))

    def update_ribbon_backbone_atom_visibility(self):
        '''Update the 'hide' status for ribbon backbone atoms, which
            are hidden unless any of its neighbors are visible.'''
        f = c_function('atom_update_ribbon_backbone_atom_visibility',
                       args = [ctypes.c_void_p, ctypes.c_size_t])
        f(self._c_pointers, len(self))

    def have_alt_loc(self, loc):
        if isinstance(loc, str):
            loc = loc.encode('utf-8')
        n = len(self)
        values = empty((n,), npy_bool)
        f = c_array_function('atom_has_alt_loc', args=(byte,), ret=npy_bool, per_object=False)
        f(self._c_pointers, n, loc, pointer(values))
        return values

    has_aniso_u = cvec_property('atom_has_aniso_u', npy_bool, read_only=True,
        doc='Boolean array identifying which atoms have anisotropic temperature factors.')

    @property
    def aniso_u(self):
        '''Anisotropic temperature factors, returns Nx3x3 array of numpy float32.
        If any of the atoms does not have temperature factors it raises a ValueError exception.
        Read only.'''
        f = c_function('atom_aniso_u', args = (ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p))
        from numpy import empty, float32
        n = len(self)
        ai = empty((n,3,3), float32)
        f(self._c_pointers, n, pointer(ai))
        return ai

    def _get_aniso_u6(self):
        '''Get anisotropic temperature factors as a Nx6 array of numpy float32 containing
        (u11,u22,u33,u12,u13,u23) for each atom. If any of the atoms does not have
        temperature factors raise a ValueError exception.'''
        f = c_function('atom_aniso_u6', args = (ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p))
        from numpy import empty, float32
        n = len(self)
        ai = empty((n,6), float32)
        f(self._c_pointers, n, pointer(ai))
        return ai
    def _set_aniso_u6(self, u6):
        '''Set anisotropic temperature factors as a Nx6 element numpy float32 array
        representing the unique elements of the symmetrix matrix
        containing (u11, u22, u33, u12, u13, u23) for each atom.'''
        n = len(self)
        if u6 is None:
            f = c_function('clear_atom_aniso_u6', args = (ctypes.c_void_p, ctypes.c_size_t))
            f(self._c_pointers, n)
            return
        f = c_function('set_atom_aniso_u6', args = (ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p))
        from numpy import empty, float32
        ai = empty((n,6), float32)
        ai[:] = u6
        f(self._c_pointers, n, pointer(ai))
    aniso_u6 = property(_get_aniso_u6, _set_aniso_u6)

    def residue_sums(self, atom_values):
        '''Compute per-residue sum of atom float values.  Return unique residues and array of residue sums.'''
        f = c_function('atom_residue_sums', args=(ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_double)),
                       ret=ctypes.py_object)
        rp, rsums = f(self._c_pointers, len(self), pointer(atom_values))
        return Residues(rp), rsums

    @classmethod
    def session_restore_pointers(cls, session, data):
        structures, atom_ids = data
        return array([s.session_id_to_atom(i) for s, i in zip(structures, atom_ids)], dtype=cptr)
    def session_save_pointers(self, session):
        structures = self.structures
        atom_ids = [s.session_atom_to_id(ptr) for s, ptr in zip(structures, self._c_pointers)]
        return [structures, array(atom_ids)]

# -----------------------------------------------------------------------------
#
class Bonds(Collection):
    '''
    Collection of C++ bonds.
    '''
    def __init__(self, bond_pointers = None):
        Collection.__init__(self, bond_pointers, molobject.Bond)

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
    displays = cvec_property('bond_display', npy_bool)
    '''
    Controls whether the Bonds should be displayed.
    Returns a :mod:`numpy` array of bool.  Can be
    set with such an array (or equivalent sequence), or with a
    single value.  Bonds are shown only if display is
    true, hide is false, and both atoms are shown.
    '''
    visibles = cvec_property('bond_visible', npy_bool, read_only = True)
    '''
    Returns whether the Bonds should be visible regardless
    of whether the atoms on either end is shown.
    Returns a :mod:`numpy` array of bool.  Read only.
    '''
    halfbonds = cvec_property('bond_halfbond', npy_bool)
    '''
    Controls whether the Bonds should be colored in "halfbond"
    mode, *i.e.* each half colored the same as its endpoint atom.
    Returns a :mod:`numpy` array of boolean values.  Can be
    set with such an array (or equivalent sequence), or with a
    single boolean value.
    '''
    lengths = cvec_property('bond_length', float32, read_only = True)
    '''
    Returns a :mod:`numpy` array of bond lengths. Read only.
    '''
    radii = cvec_property('bond_radius', float32)
    '''
    Returns a :mod:`numpy` array of bond radii (half thicknesses).
    Can be set with such an array (or equivalent sequence), or with a
    single floating-point number.
    '''
    selected = cvec_property('bond_selected', npy_bool)
    '''numpy bool array whether each Bond is selected.'''
    selecteds = selected
    ends_selected = cvec_property('bond_ends_selected', npy_bool, read_only = True)
    '''For each bond are both of its endpoint atoms selected.'''
    showns = cvec_property('bond_shown', npy_bool, read_only = True)
    '''
    Whether each bond is displayed, visible and has both atoms shown,
    and at least one atom is not Sphere style.
    '''
    structures = cvec_property('bond_structure', pyobject, astype = AtomicStructures, read_only = True)
    '''Returns an :class:`.StructureDatas` with the structure for each bond. Read only.'''
    @property
    def unique_structures(self):
        "The unique structures as an :class:`.AtomicStructures` collection"
        return self.structures.unique()

    @property
    def unique_structures(self):
        "The unique structures as an :class:`.AtomicStructures` collection"
        return self.structures.unique()

    @property
    def by_structure(self):
        "Return list of 2-tuples of (structure, Bonds for that structure)."
        bstruct = self.structures._pointers
        return [(us, self.filter(bstruct==us._c_pointer.value)) for us in self.unique_structures]

    def delete(self):
        '''Delete the C++ Bonds objects'''
        c_function('bond_delete',
            args = [ctypes.c_void_p, ctypes.c_size_t])(self._c_pointers, len(self))

    @property
    def num_shown(self):
        '''Number of bonds shown.'''
        f = c_function('bonds_num_shown', args = [ctypes.c_void_p, ctypes.c_size_t], ret = ctypes.c_size_t)
        return f(self._c_pointers, len(self))

    @property
    def num_selected(self):
        "Number of selected bonds."
        f = c_function('bonds_num_selected',
                       args = [ctypes.c_void_p, ctypes.c_size_t],
                       ret = ctypes.c_size_t)
        return f(self._c_pointers, len(self))

    @property
    def half_colors(self):
        '''2N x 4 RGBA uint8 numpy array of half bond colors.'''
        f = c_function('bond_half_colors', args = [ctypes.c_void_p, ctypes.c_size_t], ret = ctypes.py_object)
        return f(self._c_pointers, len(self))

    def halfbond_cylinder_placements(self, opengl_array = None):
        '''Return Places for halfbond cylinders specified by 2N 4x4 float matrices.'''
        n = len(self)
        if opengl_array is None or len(opengl_array) != 2*n:
            from numpy import empty, float32
            opengl_array = empty((2*n,4,4), float32)
        f = c_function('bond_halfbond_cylinder_placements',
                       args = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p])
        f(self._c_pointers, n, pointer(opengl_array))
        from chimerax.geometry import Places
        return Places(opengl_array = opengl_array)
        
    @classmethod
    def session_restore_pointers(cls, session, data):
        structures, bond_ids = data
        return array([s.session_id_to_bond(i) for s, i in zip(structures, bond_ids)], dtype=cptr)
    def session_save_pointers(self, session):
        structures = self.structures
        bond_ids = [s.session_bond_to_id(ptr) for s, ptr in zip(structures, self._c_pointers)]
        return [structures, array(bond_ids)]

# -----------------------------------------------------------------------------
#
class Elements(Collection):
    '''
    Holds a collection of C++ Elements (chemical elements) and provides access to some of
    their attributes.  Used for the same reasons as the :class:`Atoms` class.
    '''
    def __init__(self, element_pointers):
        Collection.__init__(self, element_pointers, molobject.Element)

    names = cvec_property('element_name', string, read_only = True)
    '''Returns a numpy array of chemical element names. Read only.'''
    numbers = cvec_property('element_number', uint8, read_only = True)
    '''Returns a :mod:`numpy` array of atomic numbers (integers). Read only.'''
    masses = cvec_property('element_mass', float32, read_only = True)
    '''Returns a :mod:`numpy` array of atomic masses,
    taken from http://en.wikipedia.org/wiki/List_of_elements_by_atomic_weight.
    Read only.'''
    is_alkali_metal = cvec_property('element_is_alkali_metal', npy_bool, read_only = True)
    '''Returns a :mod:`numpy` array of booleans, where True indicates the
    element is atom an alkali metal. Read only.'''
    is_halogen = cvec_property('element_is_halogen', npy_bool, read_only = True)
    '''Returns a :mod:`numpy` array of booleans, where True indicates the
    element is atom a halogen. Read only.'''
    is_metal = cvec_property('element_is_metal', npy_bool, read_only = True)
    '''Returns a :mod:`numpy` array of booleans, where True indicates the
    element is atom a metal. Read only.'''
    is_noble_gas = cvec_property('element_is_noble_gas', npy_bool, read_only = True)
    '''Returns a :mod:`numpy` array of booleans, where True indicates the
    element is atom a noble gas. Read only.'''
    valences = cvec_property('element_valence', uint8, read_only = True)
    '''Returns a :mod:`numpy` array of atomic valence numbers (integers). Read only.'''

    @classmethod
    def session_restore_pointers(cls, session, data):
        f = c_function('element_number_get_element', args = (ctypes.c_int,), ret = ctypes.c_void_p)
        return array([f(en) for en in data], dtype=cptr)
    def session_save_pointers(self, session):
        return self.numbers

# -----------------------------------------------------------------------------
#
class Pseudobonds(Collection):
    '''
    Holds a collection of C++ PBonds (pseudobonds) and provides access to some of
    their attributes. It has the same attributes as the
    :class:`Bonds` class and works in an analogous fashion.
    '''
    def __init__(self, pbond_pointers = None):
        Collection.__init__(self, pbond_pointers, molobject.Pseudobond)

    atoms = cvec_property('pseudobond_atoms', cptr, 2, astype = _atoms_pair, read_only = True)
    '''
    Returns a two-tuple of :class:`Atoms` objects.
    For each bond, its endpoint atoms are in the matching
    position in the two :class:`Atoms` collections. Read only.
    '''
    colors = cvec_property('pseudobond_color', uint8, 4)
    '''
    Returns a :mod:`numpy` Nx4 array of uint8 RGBA values.  Can be
    set with such an array (or equivalent sequence), or with a single
    RGBA value.
    '''
    displays = cvec_property('pseudobond_display', npy_bool)
    '''
    Controls whether the pseudobonds should be displayed.
    Returns a :mod:`numpy` array of bool.  Can be
    set with such an array (or equivalent sequence), or with a
    single value.  Pseudobonds are shown only if display is
    true, hide is false, and both atoms are shown.
    '''
    groups = cvec_property('pseudobond_group', cptr, astype = _pseudobond_groups, read_only = True)
    '''
    Returns a :py:class:`PseudobondGroups` collection
    of the pseudobond groups these pseudobonds belong to
    '''
    halfbonds = cvec_property('pseudobond_halfbond', npy_bool)
    '''
    Controls whether the pseudobonds should be colored in "halfbond"
    mode, *i.e.* each half colored the same as its endpoint atom.
    Returns a :mod:`numpy` array of boolean values.  Can be
    set with such an array (or equivalent sequence), or with a
    single boolean value.
    '''
    radii = cvec_property('pseudobond_radius', float32)
    '''
    Returns a :mod:`numpy` array of pseudobond radii (half thicknesses).
    Can be set with such an array (or equivalent sequence), or with a
    single floating-point number.
    '''
    selected = cvec_property('pseudobond_selected', npy_bool)
    '''numpy bool array whether each Pseudobond is selected.'''
    selecteds = selected
    showns = cvec_property('pseudobond_shown', npy_bool, read_only = True)
    '''
    Whether each pseudobond is displayed, visible and has both atoms displayed.
    '''
    shown_when_atoms_hiddens = cvec_property('pseudobond_shown_when_atoms_hidden', npy_bool, doc =
    '''Controls whether the pseudobond is shown when the endpoint atoms are not
    explictly displayed (atom.display == False) but are implicitly shown by a
    ribbon or somesuch (atom.hide != 0).  Defaults to True.''')

    def delete(self):
        '''Delete the C++ Pseudobond objects'''
        c_function('pseudobond_delete',
            args = [ctypes.c_void_p, ctypes.c_size_t])(self._c_pointers, len(self))

    @property
    def lengths(self):
        '''Distances between pseudobond end points.'''
        a1, a2 = self.atoms
        v = a1.scene_coords - a2.scene_coords
        from numpy import sqrt
        return sqrt((v*v).sum(axis=1))

    @property
    def half_colors(self):
        '''2N x 4 RGBA uint8 numpy array of half bond colors.'''
        f = c_function('pseudobond_half_colors', args = [ctypes.c_void_p, ctypes.c_size_t], ret = ctypes.py_object)
        return f(self._c_pointers, len(self))

    def between_atoms(self, atoms):
        '''Return mask of those pseudobonds which have both ends in the given set of atoms.'''
        a1, a2 = self.atoms
        return a1.mask(atoms) & a2.mask(atoms)

    @property
    def unique_groups(self):
        return self.groups.unique()

    @property
    def unique_structures(self):
        '''The unique structures as a :class:`.StructureDatas` collection'''
        a1, a2 = self.atoms
        s = concatenate((a1.unique_structures, a2.unique_structures), AtomicStructures, remove_duplicates = True)
        return s

    @property
    def by_group(self):
        "Return list of 2-tuples of (PseudobondGroup, Pseudobonds for that group)."
        gps = self.groups
        gpp = gps._pointers
        return [(g, self.filter(gpp==g._c_pointer.value)) for g in gps.unique()]

    @property
    def num_selected(self):
        "Number of selected pseudobonds."
        f = c_function('pseudobonds_num_selected',
                       args = [ctypes.c_void_p, ctypes.c_size_t],
                       ret = ctypes.c_size_t)
        return f(self._c_pointers, len(self))

    def with_group(self, pbg):
        return self.filter(self.groups._pointers == pbg.cpp_pointer)

    _ses_ids = cvec_property('pseudobond_get_session_id', int32, read_only = True,
        doc="Used internally to save/restore in sessions")
    @staticmethod
    def session_restore_pointers(session, data):
        groups, ids = data
        f = c_function('pseudobond_group_resolve_session_id',
            args = [ctypes.c_void_p, ctypes.c_int], ret = ctypes.c_void_p)
        ptrs = array([f(grp_ptr, id) for grp_ptr, id in zip(groups._c_pointers, ids)], dtype=cptr)
        return ptrs
    def session_save_pointers(self, session):
        return [self.groups, self._ses_ids]

# -----------------------------------------------------------------------------
#
class Residues(Collection):
    '''
    Collection of C++ residue objects.
    '''
    def __init__(self, residue_pointers = None):
        Collection.__init__(self, residue_pointers, molobject.Residue)

    atoms = cvec_property('residue_atoms', cptr, 'num_atoms', astype = _atoms, read_only = True, per_object = False, doc =
    '''Return :class:`.Atoms` belonging to each residue all as a single collection. Read only.''')
    centers = cvec_property('residue_center', float64, 3, read_only = True)
    '''Average of atom positions as a numpy length 3 array, 64-bit float values.'''
    chains = cvec_property('residue_chain', cptr, astype = _non_null_chains, read_only = True, doc =
    '''Return :class:`.Chains` for residues. Residues with no chain are omitted. Read only.''')
    chain_ids = cvec_property('residue_chain_id', string, doc =
    '''Returns a numpy array of chain IDs.''')
    mmcif_chain_ids = cvec_property('residue_mmcif_chain_id', string, read_only = True, doc =
    '''Returns a numpy array of chain IDs. Read only.''')
    insertion_codes = cvec_property('residue_insertion_code', string, doc =
    '''Returns a numpy array of insertion codes.  An empty string indicates no insertion code.''')
    is_helix = cvec_property('residue_is_helix', npy_bool, doc =
    '''Returns a numpy bool array whether each residue is in a protein helix''')
    is_helices = is_helix
    is_missing_heavy_template_atoms = cvec_property('residue_is_missing_heavy_template_atoms', npy_bool,
    read_only = True, doc = '''Returns a numpy bool array whether each residue is missing heavy atoms relative to its template.  If no template, returns False.''')
    is_strand = cvec_property('residue_is_strand', npy_bool, doc =
    '''Returns a numpy bool array whether each residue is in a protein sheet''')
    is_strands = is_strand
    names = cvec_property('residue_name', string, doc =
    '''Returns a numpy array of residue names.''')
    num_atoms = cvec_property('residue_num_atoms', size_t, read_only = True, doc =
    '''Returns a numpy integer array of the number of atoms in each residue. Read only.''')
    numbers = cvec_property('residue_number', int32, doc =
    '''
    Returns a :mod:`numpy` array of residue sequence numbers, as provided by
    whatever data source the structure came from, so not necessarily consecutive,
    or starting from 1, *etc.*.
    ''')
    polymer_types = cvec_property('residue_polymer_type', uint8, read_only = True, doc =
    '''Returns a numpy int array of residue types. Read only.''')
    principal_atoms = cvec_property('residue_principal_atom', cptr, astype = _atoms_or_nones,
        read_only = True, doc =
    '''List of the 'chain trace' :class:`.Atom`\\ s or None (for residues without such an atom).

    Normally returns the C4' from a nucleic acid since that is always present,
    but in the case of a P-only trace it returns the P.''')
    existing_principal_atoms = cvec_property('residue_principal_atom', cptr, astype = _non_null_atoms, read_only = True, doc =
    '''Like the principal_atoms property, but returns a :class:`.Residues` collection omitting Nones''')
    ribbon_displays = cvec_property('residue_ribbon_display', npy_bool, doc =
    '''A numpy bool array whether to display each residue in ribbon style.''')
    ribbon_colors = cvec_property('residue_ribbon_color', uint8, 4, doc =
    '''
    A :mod:`numpy` Nx4 array of uint8 RGBA values.  Can be
    set with such an array (or equivalent sequence), or with a single
    RGBA value.
    ''')
    ribbon_adjusts = cvec_property('residue_ribbon_adjust', float32, doc =
    '''A numpy float array of adjustment factors for the position of ribbon
    control points.  Factors range from zero to one, with zero being using the
    actual atomic coordinates as control point, and one being using the idealized
    secondary structure position as control point.  A negative value means to
    use the default of zero for turns and helices and 0.7 for strands.''')
    ribbon_hide_backbones = cvec_property('residue_ribbon_hide_backbone', npy_bool, doc =
    '''A :mod:`numpy` array of booleans. Whether a ribbon automatically hides
    the residue backbone atoms.''')
    ring_displays = cvec_property('residue_ring_display', npy_bool, doc =
    '''A numpy bool array whether to fill rings in each residue.''')
    ring_colors = cvec_property('residue_ring_color', uint8, 4, doc =
    '''
    A :mod:`numpy` Nx4 array of uint8 RGBA values.  Can be
    set with such an array (or equivalent sequence), or with a single
    RGBA value.
    ''')
    thin_rings = cvec_property('residue_thin_rings', npy_bool, doc =
    '''A numpy bool array whether to filled rings are thin in each residue.''')
    secondary_structure_ids = cvec_property('residue_secondary_structure_id', int32,
        read_only = True, doc =
    '''
    A :mod:`numpy` array of integer secondary structure ids.  Every helix, sheet, coil
    has a unique integer id.  The ids depend on the collection of residues on the fly and are
    not persistent. Read only.
    ''')
    selected = cvec_property('residue_selected', npy_bool, read_only = True,
        doc="numpy bool array whether any Atom in each Residue is selected. Read only.")
    ss_ids = cvec_property('residue_ss_id', int32, doc =
    '''
    A :mod:`numpy` array of integer secondary structure IDs, determined by the input file.
    For a PDB file, for helices, the ID is the same as in the HELIX record; for strands,
    it starts as 1 for the strand nearest the N terminus, and increments for each strand
    out to the C terminus.
    ''')
    ss_types = cvec_property('residue_ss_type', int32, doc =
    '''Returns a numpy integer array of secondary structure types (one of: Residue.SS_COIL, Residue.SS_HELIX, Residue.SS_STRAND [or SS_SHEET])''')
    structures = cvec_property('residue_structure', pyobject, astype = AtomicStructures, read_only = True, doc =
    '''Returns :class:`.StructureDatas` collection containing structures for each residue.''')

    def delete(self):
        '''Delete the C++ Residue objects'''
        c_function('residue_delete',
            args = [ctypes.c_void_p, ctypes.c_size_t])(self._c_pointers, len(self))

    @property
    def chi1s(self):
        return [r.chi1 for r in self]

    @chi1s.setter
    def chi1s(self, chi1):
        for r in self:
            r.chi1 = chi1

    @property
    def chi2s(self):
        return [r.chi2 for r in self]

    @chi2s.setter
    def chi2s(self, chi2):
        for r in self:
            r.chi2 = chi2

    @property
    def chi3s(self):
        return [r.chi3 for r in self]

    @chi3s.setter
    def chi3s(self, chi3):
        for r in self:
            r.chi3 = chi3

    @property
    def chi4s(self):
        return [r.chi4 for r in self]

    @chi4s.setter
    def chi4s(self, chi4):
        for r in self:
            r.chi4 = chi4

    @property
    def omegas(self):
        return [r.omega for r in self]

    @omegas.setter
    def omegas(self, omega):
        for r in self:
            r.omega = omega

    @property
    def phis(self):
        return [r.phi for r in self]

    @phis.setter
    def phis(self, phi):
        for r in self:
            r.phi = phi

    @property
    def psis(self):
        return [r.psi for r in self]

    @psis.setter
    def psis(self, psi):
        for r in self:
            r.psi = psi

    @property
    def unique_structures(self):
        '''The unique structures as a :class:`.StructureDatas` collection'''
        return self.structures.unique()

    @property
    def unique_names(self):
        '''The unique names as a numpy array of strings.'''
        return unique_ordered(self.names)

    @property
    def unique_chain_ids(self):
        '''The unique chain IDs as a numpy array of strings.'''
        return unique_ordered(self.chain_ids)

    @property
    def unique_chains(self):
        '''The unique chains as a :class:`.Chains` collection'''
        return self.chains.unique()

    @property
    def by_chain(self):
        '''Return list of structure, chain id, and Residues for each chain.'''
        chains = []
        for m, residues in self.by_structure:
            cids = residues.chain_ids
            for cid in unique_ordered(cids):
                chains.append((m, cid, residues.filter(cids == cid)))
        return chains

    @property
    def by_structure(self):
        '''Return list of pairs of structure and Residues for that structure.'''
        rmol = self.structures._pointers
        return [(m, self.filter(rmol==m._c_pointer.value)) for m in self.unique_structures]

    @property
    def unique_ids(self):
        '''
        A :mod:`numpy` array of uintp (unsigned integral type large enough to hold a pointer).
        Multiple copies of the same residue in the collection will have the same integer value
        in the returned array. Read only.
        '''
        return self._pointers

    @property
    def unique_sequences(self):
        '''
        Return a list of sequence strings and a :mod:`numpy` array giving an integer index for each residue.
        Index 0 is for residues that are not part of a chain (empty string).
        '''
        from numpy import empty, int32
        seq_ids = empty((len(self),), int32)
        f = c_function('residue_unique_sequences',
                       args = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p],
                       ret = ctypes.py_object)
        seqs = f(self._c_pointers, len(self), pointer(seq_ids))
        return seqs, seq_ids

    def ribbon_clear_hides(self):
        self.clear_hide_bits(Atom.HIDE_RIBBON)

    def clear_hide_bits(self, mask, atoms_only=False):
        '''Clear the hide bit for all atoms and bonds in given residues.'''
        f = c_function('residue_clear_hide_bits',
                       args = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_bool])
        f(self._c_pointers, len(self), mask, atoms_only)

    def set_alt_locs(self, loc):
        if isinstance(loc, str):
            loc = loc.encode('utf-8')
        f = c_array_function('residue_set_alt_loc', args=(byte,), per_object=False)
        f(self._c_pointers, len(self), loc)

    @classmethod
    def session_restore_pointers(cls, session, data):
        structures, residue_ids = data
        return array([s.session_id_to_residue(i) for s, i in zip(structures, residue_ids)], dtype=cptr)
    def session_save_pointers(self, session):
        structures = self.structures
        residue_ids = [s.session_residue_to_id(ptr) for s, ptr in zip(structures, self._c_pointers)]
        return [structures, array(residue_ids)]


# -----------------------------------------------------------------------------
#
class Rings(Collection):
    '''
    Collection of C++ ring objects.
    '''
    def __init__(self, ring_pointers = None, rings = None):
        if rings is not None:
            # Extract C pointers from list of Python Ring objects.
            ring_pointers = array([r._c_pointer.value for r in rings], cptr)
        Collection.__init__(self, ring_pointers, molobject.Ring)
        # Create the list of Rings immediately, so that their info gets cached before they
        # possibly get destroyed by other calls
        c = self._object_class
        self._object_list = [c.c_ptr_to_py_inst(p) for p in self._pointers]


# -----------------------------------------------------------------------------
#
class Chains(Collection):
    '''
    Collection of C++ chain objects.
    '''

    def __init__(self, chain_pointers):
        Collection.__init__(self, chain_pointers, molobject.Chain)

    chain_ids = cvec_property('sseq_chain_id', string)
    '''A numpy array of string chain ids for each chain.'''
    structures = cvec_property('sseq_structure', pyobject, astype = AtomicStructures, read_only = True)
    '''A :class:`.StructureDatas` collection containing structures for each chain.'''
    existing_residues = cvec_property('sseq_residues', cptr, 'num_residues',
        astype = _non_null_residues, read_only = True, per_object = False)
    '''A :class:`Residues` containing the existing residues of all chains. Read only.'''
    num_existing_residues = cvec_property('sseq_num_existing_residues', size_t, read_only = True)
    '''A numpy integer array containing the number of existing residues in each chain.'''
    num_residues = cvec_property('sseq_num_residues', size_t, read_only = True)
    '''A numpy integer array containing the number of residues in each chain.'''
    polymer_types = cvec_property('sseq_polymer_type', uint8, read_only = True, doc =
    '''Returns a numpy int array of residue types. Same values as Residues.polymer_types except shouldn't return PT_NONE.''')

    @classmethod
    def session_restore_pointers(cls, session, data):
        structures, chain_ses_ids = data
        return array([s.session_id_to_chain(i) for s, i in zip(structures, chain_ses_ids)], dtype=cptr)
    def session_save_pointers(self, session):
        structures = self.structures
        chain_ses_ids = [s.session_chain_to_id(ptr) for s, ptr in zip(structures, self._c_pointers)]
        return [structures, array(chain_ses_ids)]

# -----------------------------------------------------------------------------
#
class PseudobondGroupDatas(Collection):
    '''
    Collection of C++ pseudobond group objects.
    '''
    def __init__(self, pbg_pointers):
        Collection.__init__(self, pbg_pointers, molobject.PseudobondGroupData)

    colors = cvec_property('pseudobond_group_color', uint8, 4,
        doc="Returns a :mod:`numpy` Nx4 array of uint8 RGBA values. Can be set "
        "with such an array (or equivalent sequence), or with a single RGBA value.")
    halfbonds = cvec_property('pseudobond_group_halfbond', npy_bool)
    '''
    Controls whether the pseudobonds should be colored in "halfbond"
    mode, *i.e.* each half colored the same as its endpoint atom.
    Returns a :mod:`numpy` array of boolean values.  Can be
    set with such an array (or equivalent sequence), or with a
    single boolean value.
    '''
    pseudobonds = cvec_property('pseudobond_group_pseudobonds', cptr, 'num_pseudobonds',
                                astype = _pseudobonds, read_only = True, per_object = False)
    '''A single :class:`.Pseudobonds` object containing pseudobonds for all groups. Read only.'''
    names = cvec_property('pseudobond_group_category', string, read_only = True)
    '''A numpy string array of categories of each group.'''
    num_pseudobonds = cvec_property('pseudobond_group_num_pseudobonds', size_t, read_only = True)
    '''Number of pseudobonds in each group. Read only.'''
    radii = cvec_property('pseudobond_group_radius', float32,
        doc="Returns a :mod:`numpy` array of radii.  Can be set with such an array (or equivalent "
        "sequence), or with a single floating-point number.")

    # 'displays' being defined as property only so that setattr (used by selection inspector)
    # knows that it exists.  Not defining as a vector property since we actually want to go
    # through Model.display
    @property
    def displays(self):
        return array([pbg.display for pbg in self])
    @displays.setter
    def displays(self, d):
        for pbg in self:
            pbg.display = d

# -----------------------------------------------------------------------------
#
class PseudobondGroups(PseudobondGroupDatas):
    '''
    Collection of Python pseudobond group objects.
    '''
    def __init__(self, pbg_pointers):
        from . import pbgroup
        Collection.__init__(self, pbg_pointers, pbgroup.PseudobondGroup)

    @property
    def dashes(self):
        return array([pbg.dashes for pbg in self])

    @dashes.setter
    def dashes(self, n):
        for pbg in self:
            pbg.dashes = n

    @property
    def names(self):
        return array([pbg.name for pbg in self])

    @classmethod
    def session_restore_pointers(cls, session, data):
        return array([s._c_pointer.value for s in data], dtype=cptr)
    def session_save_pointers(self, session):
        return [s for s in self]

# -----------------------------------------------------------------------------
#
class CoordSets(Collection):
    '''
    Collection of C++ coordsets.
    '''
    def __init__(self, cs_pointers = None):
        Collection.__init__(self, cs_pointers, molobject.CoordSet)

    ids = cvec_property('coordset_id', uint32, read_only = True,
        doc="ID numbers of coordsets")
    structures = cvec_property('coordset_structure', pyobject, astype = AtomicStructures, read_only=True,
        doc="Returns an :class:`AtomicStructure` for each coordset. Read only.")

    @property
    def unique_structures(self):
        '''The unique structures as a :class:`.AtomicStructures` collection'''
        return self.structures.unique()

# -----------------------------------------------------------------------------
#
class Structures(StructureDatas):
    '''
    Collection of Python structure objects.
    '''
    def __init__(self, mol_pointers):
        from . import Structure
        Collection.__init__(self, mol_pointers, Structure)

    # so setattr knows that attr exists (used by selection inspector);
    # also, don't want to directly set Structure.display, want to go through Model.display
    @property
    def displays(self):
        return array([s.display for s in self])
    @displays.setter
    def displays(self, d):
        for s in self:
            s.display = d

    @property
    def visibles(self):
        return array([s.visible for s in self])

    @classmethod
    def session_restore_pointers(cls, session, data):
        return array([s._c_pointer.value for s in data], dtype=cptr)
    def session_save_pointers(self, session):
        return [s for s in self]

# -----------------------------------------------------------------------------
# For making collections from lists of objects.
#
def object_pointers(objects):
    pointers = array(tuple(o._c_pointer.value for o in objects), dtype = cptr,)
    return pointers

# -----------------------------------------------------------------------------
# When C++ object is deleted, delete it from the specified pointer array.
#
def remove_deleted_pointers(array):
    remove_deleted_c_pointers(array)
    import weakref
    weakref.finalize(array, pointer_array_freed, id(array))

remove_deleted_c_pointers = c_function('remove_deleted_c_pointers', args = [ctypes.py_object])
pointer_array_freed = c_function('pointer_array_freed', args = [ctypes.c_void_p])
