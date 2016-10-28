# vim: set expandtab shiftwidth=4 softtabstop=4:

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

Collections are immutable so can be hashed.  The only case in which their contents
can be altered is if C++ objects they hold are deleted in which case those objects
are automatically removed from the collection.
'''
from numpy import uint8, int32, uint32, float64, float32, uintp, byte, bool as npy_bool, integer, empty, unique, array
from .molc import string, cptr, pyobject, cvec_property, set_cvec_pointer, c_function, c_array_function, pointer, ctype_type_to_numpy
from . import molobject
import ctypes
size_t = ctype_type_to_numpy[ctypes.c_size_t]   # numpy dtype for size_t

def _atoms(p):
    return Atoms(p)
def _atoms_or_nones(p):
    from .molobject import object_map, Atom
    return [object_map(ptr, Atom) if ptr else None for ptr in p]
def _non_null_atoms(p):
    return Atoms(p[p!=0])
def _bonds(p):
    return Bonds(p)
def _pseudobond_groups(p):
	return PseudobondGroups(p)
def _pseudobonds(p):
	return Pseudobonds(p)
def _elements(p):
    return Elements(p)
def _residues(p):
    return Residues(p)
def _non_null_residues(p):
    from .molarray import Residues
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
from ..state import State
class Collection(State):
    '''
    Base class of all molecular data collections that provides common
    methods such as length, iteration, indexing with square brackets,
    intersection, union, subtracting, and filtering.  By design, a
    Collection is immutable.
    '''
    def __init__(self, items, object_class, objects_class):
        if items is None:
            # Empty Atoms
            import numpy
            pointers = numpy.empty((0,), cptr)
        elif type(items) in [list, tuple]:
            # presumably items of the object_class
            import numpy
            pointers = numpy.array([i._c_pointer.value for i in items], cptr)
        else:
            pointers = items
        self._pointers = pointers
        self._object_class = object_class
        self._objects_class = objects_class
        set_cvec_pointer(self, pointers)
        remove_deleted_pointers(pointers)

    def __eq__(self, atoms):
        import numpy
        return numpy.array_equal(atoms._pointers, self._pointers)
    def hash(self):
        from hashlib import sha1
        return sha1(self._pointers.view(uint8)).digest()
    def __len__(self):
        '''Number of objects in collection.'''
        return len(self._pointers)
    def __bool__(self):
        return len(self) > 0
    def __iter__(self):
        '''Iterator over collection objects.'''
        if not hasattr(self, '_object_list'):
            from .molobject import object_map
            c = self._object_class
            self._object_list = [object_map(p,c) for p in self._pointers]
        return iter(self._object_list)
    def __getitem__(self, i):
        '''Indexing of collection objects using square brackets, *e.g.* c[i].'''
        import numpy
        if isinstance(i,(int,integer)):
            from .molobject import object_map
            v = object_map(self._pointers[i], self._object_class)
        elif isinstance(i, (slice, numpy.ndarray)):
            v = self._objects_class(self._pointers[i])
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

    @property
    def object_class(self):
        return self._object_class
    @property
    def objects_class(self):
        return self._objects_class
    @property
    def pointers(self):
        return self._pointers

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

    def copy(self):
        '''Shallow copy, since Collections are immutable.'''
        return self._objects_class(self._pointers)

    def intersect(self, objects):
        '''Return a new collection that is the intersection with the *objects* :class:`.Collection`.'''
        import numpy
        return self._objects_class(numpy.intersect1d(self._pointers, objects._pointers))
    def intersects(self, objects):
        '''Whether this collection has any element in common with the *objects* :class:`.Collection`. Returns bool.'''
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
    def filter(self, mask_or_indices):
        '''Return a subset of the collection as a new collection.

        Parameters
        ----------
        mask_or_indices : numpy bool array (mask) or int array (indices)
          Bool length must match the length of the collection and filters out items where
          the bool array is False.
        '''
        return self._objects_class(self._pointers[mask_or_indices])
    def mask(self, objects):
        '''Return bool array indicating for each object in current set whether that
        object appears in the argument objects.'''
        f = c_function('pointer_mask', args = [ctypes.c_void_p, ctypes.c_size_t,
                                               ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p])
        mask = empty((len(self),), npy_bool)
        f(self._c_pointers, len(self), objects._c_pointers, len(objects), pointer(mask))
        return mask
    def indices(self, objects):
        '''Return int32 array indicating for each object in current set the index of
        that object in the argument objects, or -1 if it does not occur in objects.'''
        f = c_function('pointer_indices', args = [ctypes.c_void_p, ctypes.c_size_t,
                                               ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p])
        ind = empty((len(self),), int32)
        f(self._c_pointers, len(self), objects._c_pointers, len(objects), pointer(ind))
        return ind
    def merge(self, objects):
        '''Return a new collection combining this one with the *objects* :class:`.Collection`.
        All duplicates are removed.'''
        import numpy
        return self._objects_class(numpy.union1d(self._pointers, objects._pointers))
    def subtract(self, objects):
        '''Return a new collection subtracting the *objects* :class:`.Collection` from this one.
        All duplicates are removed.  Currently does not preserve order'''
        import numpy
        return self._objects_class(numpy.setdiff1d(self._pointers, objects._pointers))
    def unique(self):
        '''Return a new collection containing the unique elements from this one, preserving order.'''
        indices = unique(self._pointers, return_index = True)[1]
        indices.sort()
        return self.objects_class(self._pointers[indices])

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
    def reset_state(self, session):
        self._pointers = numpy.empty((0,), cptr)
    @classmethod
    def session_restore_pointers(cls, session, data):
        raise NotImplementedError(
            self.__class__.__name__ + " has not implemented session_restore_pointers")
    def session_save_pointers(self, session):
        raise NotImplementedError(
            self.__class__.__name__ + " has not implemented session_save_pointers")

def concatenate(collections, object_class = None, remove_duplicates = False):
    '''Concatenate any number of collections returning a new collection.
    All collections must have the same type.
    
    Parameters
    ----------
    collections : sequence of :class:`.Collection` objects
    '''
    cl = collections[0]._objects_class if object_class is None else object_class
    if len(collections) == 0:
        c = object_class()
    else:
        import numpy
        p = numpy.concatenate([a._pointers for a in collections])
        if remove_duplicates:
            pu, i = numpy.unique(p, return_index = True)
            p = p[numpy.sort(i)]	# Preserve order when duplicates are removed.
        c = cl(p)
    return c

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
class Atoms(Collection):
    '''
    Bases: :class:`.Collection`

    An ordered collection of atom objects. This offers better performance
    than using a list of atoms.  It provides methods to access atom attributes such
    as coordinates as numpy arrays. Atoms directly accesses the C++ atomic data
    without creating Python :py:class:`Atom` objects which require much more memory
    and are slower to use in computation.
    '''
    SPHERE_STYLE, BALL_STYLE, STICK_STYLE = range(3)

    bfactors = cvec_property('atom_bfactor', float32)
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
    def by_structure(self):
        "Return list of 2-tuples of (structure, Atoms for that structure)."
        astruct = self.structures._pointers
        return [(us, self.filter(astruct==us._c_pointer.value)) for us in self.unique_structures]
    chain_ids = cvec_property('atom_chain_id', string, read_only = True)
    colors = cvec_property('atom_color', uint8, 4,
        doc="Returns a :mod:`numpy` Nx4 array of uint8 RGBA values. Can be set "
        "with such an array (or equivalent sequence), or with a single RGBA value.")
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
    names = cvec_property('atom_name', string,
        doc="Returns a numpy array of atom names.  Can be set with such an array (or equivalent "
        "sequence), or with a single string.  Atom names are limited to 4 characters.")
    @property
    def num_selected(self):
        "Number of selected atoms."
        f = c_function('atom_num_selected',
                       args = [ctypes.c_void_p, ctypes.c_size_t],
                       ret = ctypes.c_size_t)
        return f(self._c_pointers, len(self))
    hides = cvec_property('atom_hide', int32,
        doc="Controls whether the Atom is hidden (overrides display). Returns a :mod:`numpy` "
        "array of int32 values.  Possible values:\nHIDE_RIBBON\n    Hide mask for backbone atoms "
        "in ribbon.\nCan be set with such an array (or equivalent sequence), or with a single "
        "integer value.")
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
    is_sidechains = cvec_property('atom_is_sidechain', npy_bool, read_only = True,
        doc="Whether each atom is part of an amino/nucleic acid sidechain."
            " Returns numpy bool array. Read only.")
    occupancy = cvec_property('atom_occupancy', float32)
    @property
    def inter_bonds(self):
        ":class:`Bonds` object where both endpoint atoms are in this collection"
        f = c_function('atom_inter_bonds', args = [ctypes.c_void_p, ctypes.c_size_t],
            ret = ctypes.py_object)
        return _bonds(f(self._c_pointers, len(self)))
    radii = cvec_property('atom_radius', float32,
        doc="Returns a :mod:`numpy` array of radii.  Can be set with such an array (or equivalent "
        "sequence), or with a single floating-point number.")
    residues = cvec_property('atom_residue', cptr, astype = _residues, read_only = True,
        doc="Returns a :class:`Residues` whose data items correspond in a 1-to-1 fashion with the "
        "items in the Atoms.  Read only.")
    @property
    def scene_bounds(self):
        "Return scene bounds of atoms including instances of all parent models."
        blist = []
        from ..geometry import sphere_bounds, copy_tree_bounds, union_bounds
        for m, a in self.by_structure:
            ba = sphere_bounds(a.coords, a.radii)
            ib = copy_tree_bounds(ba,
                [d.positions for d in m.drawing_lineage])
            blist.append(ib)
        return union_bounds(blist)
    def _get_scene_coords(self):
        n = len(self)
        from numpy import array, empty, float64
        xyz = empty((n,3), float64)
        if n == 0: return xyz
        structs = self.unique_structures
        gtable = array(tuple(s.scene_position.matrix for s in structs), float64)
        from .molc import pointer
        f = c_function('atom_scene_coords',
            args = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p,
                    ctypes.c_size_t, ctypes.c_void_p, ctypes.c_void_p])
        f(self._c_pointers, n, structs._c_pointers, len(structs), pointer(gtable), pointer(xyz))
        return xyz
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
    scene_coords = property(_get_scene_coords, _set_scene_coords)
    '''Atoms' coordinates in the global scene coordinate system.
    This accounts for the :class:`Drawing` positions for the hierarchy
    of models each atom belongs to.'''
    selected = cvec_property('atom_selected', npy_bool,
        doc="numpy bool array whether each Atom is selected.")
    @property
    def shown_atoms(self):
        '''
        Subset of Atoms including atoms that are displayed or have ribbon displayed
        and have displayed structure and displayed parent models.
        '''
        from .molobject import Atom
        da = self.filter(self.displays | self.residues.ribbon_displays)
        datoms = concatenate([a for m, a in da.by_structure
                              if m.display and m.parents_displayed], Atoms)
        return datoms
    structure_categories = cvec_property('atom_structure_category', string, read_only=True,
        doc="Numpy array of whether atom is ligand, ion, etc.")
    structures = cvec_property('atom_structure', cptr, astype=_atomic_structures, read_only=True,
        doc="Returns an :class:`AtomicStructure` for each atom. Read only.")
    @property
    def unique_residues(self):
        '''The unique :class:`.Residues` for these atoms.'''
        return _residues(unique(self.residues._pointers))
    @property
    def unique_chain_ids(self):
        '''The unique chain IDs as a numpy array of strings.'''
        return unique(self.chain_ids)
    @property
    def unique_structures(self):
        "The unique structures as an :class:`.AtomicStructures` collection"
        return _atomic_structures(unique(self.structures._pointers))
    @property
    def single_structure(self):
        "Do all atoms belong to a single :class:`.Structure`"
        p = self.structures._pointers
        return len(p) == 0 or (p == p[0]).all()
    visibles = cvec_property('atom_visible', npy_bool, read_only=True,
        doc="Returns whether the Atom should be visible (displayed and not hidden). Returns a "
        ":mod:`numpy` array of boolean values.  Read only.")
    alt_locs = cvec_property('atom_alt_loc', byte, astype=bytearray,
                         doc='Returns current alternate location indicators')

    def __init__(self, c_pointers = None):
        Collection.__init__(self, c_pointers, molobject.Atom, Atoms)

    def delete(self):
        '''Delete the C++ Atom objects'''
        c_function('atom_delete',
            args = [ctypes.c_void_p, ctypes.c_size_t])(self._c_pointers, len(self))

    def update_ribbon_visibility(self):
        '''Update the 'hide' status for ribbon control point atoms, which
            are hidden unless any of its neighbors are visible.'''
        f = c_function('atom_update_ribbon_visibility',
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

    def residue_sums(self, atom_values):
        '''Compute per-residue sum of atom float values.  Return unique residues and array of residue sums.'''
        f = c_function('atom_residue_sums', args=(ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_double)),
                       ret=ctypes.py_object)
        rp, rsums = f(self._c_pointers, len(self), pointer(atom_values))
        return Residues(rp), rsums
        
    @classmethod
    def session_restore_pointers(cls, session, data):
        structures, atom_ids = data
        return array([s.session_id_to_atom(i) for s, i in zip(structures, atom_ids)])
    def session_save_pointers(self, session):
        structures = self.structures
        atom_ids = [s.session_atom_to_id(ptr) for s, ptr in zip(structures, self._c_pointers)]
        return [structures, array(atom_ids)]

# -----------------------------------------------------------------------------
#
class Bonds(Collection):
    '''
    Bases: :class:`.Collection`

    Collection of C++ bonds.
    '''
    def __init__(self, bond_pointers = None):
        Collection.__init__(self, bond_pointers, molobject.Bond, Bonds)

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
    visibles = cvec_property('bond_visible', int32, read_only = True)
    '''
    Returns whether the Bonds should be visible.  If hidden, the
    return value is Never; otherwise, same as display.
    Returns a :mod:`numpy` array of integers.  Read only.
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
    showns = cvec_property('bond_shown', npy_bool, read_only = True)
    '''
    Whether each bond is displayed, visible and has both atoms shown,
    and at least one atom is not Sphere style.
    '''
    structures = cvec_property('bond_structure', cptr, astype = _atomic_structures, read_only = True)
    '''Returns an :class:`.StructureDatas` with the structure for each bond. Read only.'''

    @property
    def num_shown(self):
        '''Number of bonds shown.'''
        f = c_function('bonds_num_shown', args = [ctypes.c_void_p, ctypes.c_size_t], ret = ctypes.c_size_t)
        return f(self._c_pointers, len(self))

    @property
    def half_colors(self):
        '''2N x 4 RGBA uint8 numpy array of half bond colors.'''
        f = c_function('bond_half_colors', args = [ctypes.c_void_p, ctypes.c_size_t], ret = ctypes.py_object)
        return f(self._c_pointers, len(self))

    @classmethod
    def session_restore_pointers(cls, session, data):
        structures, bond_ids = data
        return array([s.session_id_to_bond(i) for s, i in zip(structures, bond_ids)])
    def session_save_pointers(self, session):
        structures = self.structures
        bond_ids = [s.session_bond_to_id(ptr) for s, ptr in zip(structures, self._c_pointers)]
        return [structures, array(bond_ids)]

# -----------------------------------------------------------------------------
#
class Elements(Collection):
    '''
    Bases: :class:`.Collection`

    Holds a collection of C++ Elements (chemical elements) and provides access to some of
    their attributes.  Used for the same reasons as the :class:`Atoms` class.
    '''
    def __init__(self, element_pointers):
        Collection.__init__(self, element_pointers, molobject.Element, Elements)

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
        return [f(en) for en in data]
    def session_save_pointers(self, session):
        structures = self.structures
        return self.numbers

# -----------------------------------------------------------------------------
#
class Pseudobonds(Collection):
    '''
    Bases: :class:`.Collection`

    Holds a collection of C++ PBonds (pseudobonds) and provides access to some of
    their attributes. It has the same attributes as the
    :class:`Bonds` class and works in an analogous fashion.
    '''
    def __init__(self, pbond_pointers = None):
        Collection.__init__(self, pbond_pointers, molobject.Pseudobond, Pseudobonds)

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
    showns = cvec_property('pseudobond_shown', npy_bool, read_only = True)
    '''
    Whether each pseudobond is displayed, visible and has both atoms displayed.
    '''

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

    _ses_ids = cvec_property('pseudobond_get_session_id', int32, read_only = True,
        doc="Used internally to save/restore in sessions")
    @staticmethod
    def session_restore_pointers(session, data):
        groups, ids = data
        f = c_function('pseudobond_group_resolve_session_id',
            args = [ctypes.c_void_p, ctypes.c_int], ret = ctypes.c_void_p)
        ptrs = [f(grp_ptr, id) for grp_ptr, id in zip(groups._c_pointers, ids)]
        return Pseudobonds(array(ptrs))
    def session_save_pointers(self, session):
        return [self.groups, self._ses_ids]

# -----------------------------------------------------------------------------
#
class Residues(Collection):
    '''
    Bases: :class:`.Collection`

    Collection of C++ residue objects.
    '''
    def __init__(self, residue_pointers = None, residues = None):
        if residues is not None:
            # Extract C pointers from list of Python Residue objects.
            residue_pointers = array([r._c_pointer.value for r in residues], cptr)
        Collection.__init__(self, residue_pointers, molobject.Residue, Residues)

    atoms = cvec_property('residue_atoms', cptr, 'num_atoms', astype = _atoms, read_only = True, per_object = False, doc =
    '''Return :class:`.Atoms` belonging to each residue all as a single collection. Read only.''')
    centers = cvec_property('residue_center', float64, 3, read_only = True)
    '''Average of atom positions as a numpy length 3 array, 64-bit float values.'''
    chains = cvec_property('residue_chain', cptr, astype = _non_null_chains, read_only = True, doc =
    '''Return :class:`.Chains` for residues. Residues with no chain are omitted. Read only.''')
    chain_ids = cvec_property('residue_chain_id', string, read_only = True, doc =
    '''Returns a numpy array of chain IDs. Read only.''')
    mmcif_chain_ids = cvec_property('residue_mmcif_chain_id', string, read_only = True, doc =
    '''Returns a numpy array of chain IDs. Read only.''')
    insertion_codes = cvec_property('residue_insertion_code', string, doc =
    '''Returns a numpy array of insertion codes.  An empty string indicates no insertion code.''')
    is_helix = cvec_property('residue_is_helix', npy_bool, doc =
    '''Returns a numpy bool array whether each residue is in a protein helix''')
    is_strand = cvec_property('residue_is_strand', npy_bool, doc =
    '''Returns a numpy bool array whether each residue is in a protein sheet''')
    names = cvec_property('residue_name', string, read_only = True, doc =
    '''Returns a numpy array of residue names. Read only.''')
    num_atoms = cvec_property('residue_num_atoms', size_t, read_only = True, doc =
    '''Returns a numpy integer array of the number of atoms in each residue. Read only.''')
    numbers = cvec_property('residue_number', int32, read_only = True, doc =
    '''
    Returns a :mod:`numpy` array of residue sequence numbers, as provided by
    whatever data source the structure came from, so not necessarily consecutive,
    or starting from 1, *etc.* Read only.
    ''')
    polymer_types = cvec_property('residue_polymer_type', int32, read_only = True, doc =
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
    ribbon_styles = cvec_property('residue_ribbon_style', int32, doc =
    '''A numpy int array of cartoon styles.  See constants in :class:Ribbon.''')
    ribbon_adjusts = cvec_property('residue_ribbon_adjust', float32, doc =
    '''A numpy float array of adjustment factors for the position of ribbon
    control points.  Factors range from zero to one, with zero being using the
    actual atomic coordinates as control point, and one being using the idealized
    secondary structure position as control point.  A negative value means to
    use the default of zero for turns and helices and 0.7 for strands.''')
    ribbon_hide_backbones = cvec_property('residue_ribbon_hide_backbone', npy_bool, doc =
    '''A :mod:`numpy` array of booleans. Whether a ribbon automatically hides
    the residue backbone atoms.''')
    secondary_structure_ids = cvec_property('residue_secondary_structure_id', int32,
        read_only = True, doc =
    '''
    A :mod:`numpy` array of integer secondary structure ids.  Every helix, sheet, coil
    has a unique integer id.  The ids depend on the collection of residues on the fly and are
    not persistent. Read only.
    ''')
    ss_ids = cvec_property('residue_ss_id', int32, doc =
    '''
    A :mod:`numpy` array of integer secondary structure IDs, determined by the input file.
    For a PDB file, for helices, the ID is the same as in the HELIX record; for strands,
    it starts as 1 for the strand nearest the N terminus, and increments for each strand
    out to the C terminus.
    ''')
    structures = cvec_property('residue_structure', cptr, astype = _atomic_structures,
        read_only = True, doc =
    '''Returns :class:`.StructureDatas` collection containing structures for each residue.''')

    @property
    def unique_structures(self):
        '''The unique structures as a :class:`.StructureDatas` collection'''
        return StructureDatas(unique(self.structures._pointers))

    @property
    def unique_chain_ids(self):
        '''The unique chain IDs as a numpy array of strings.'''
        return unique(self.chain_ids)

    @property
    def unique_chains(self):
        '''The unique chains as a :class:`.Chains` collection'''
        return _chains(unique(self.chains._pointers))

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
        Return :mod:`numpy` array giving an integer index for each residue and a list of sequence strings.
        Index 0 is for residues that are not part of a chain (empty string).
        '''
        from numpy import empty, int32
        seq_ids = empty((len(self),), int32)
        f = c_function('residue_unique_sequences',
                       args = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p],
                       ret = ctypes.py_object)
        seqs = f(self._c_pointers, len(self), pointer(seq_ids))
        return seqs, seq_ids

    def get_polymer_spline(self, orient):
        '''Return a tuple of spline center and guide coordinates for a
	polymer chain.  Residues in the chain that do not have a center
	atom will have their display bit turned off.  Center coordinates
	are returned as a numpy array.  Guide coordinates are only returned
	if all spline atoms have matching guide atoms; otherwise, None is
	returned for guide coordinates.'''
        f = c_function('residue_polymer_spline',
                       args = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int],
                       ret = ctypes.py_object)
        any_display, atom_pointers, centers, guides = f(self._c_pointers, len(self), orient)
        if atom_pointers is None:
            atoms = None
        else:
            atoms = Atoms(atom_pointers)
        return any_display, atoms, centers, guides

    def ribbon_clear_hides(self):
        '''Clear the hide bit for all atoms in given residues.'''
        f = c_function('residue_ribbon_clear_hide',
                       args = [ctypes.c_void_p, ctypes.c_size_t])
        f(self._c_pointers, len(self))

    def set_alt_locs(self, loc):
        if isinstance(loc, str):
            loc = loc.encode('utf-8')
        f = c_array_function('residue_set_alt_loc', args=(byte,), per_object=False)
        f(self._c_pointers, len(self), loc)

    def set_secondary_structures(self, ss_type, value):
        '''See Residue.set_secondary_structure()'''
        if value == molobject.Residue.SS_HELIX:
            f = c_array_function('residue_set_ss_helix', args=(npy_bool,), per_object=False)
        else:
            f = c_array_function('residue_set_ss_sheet', args=(npy_bool,), per_object=False)
        f(self._c_pointers, len(self), value)

# -----------------------------------------------------------------------------
#
class Chains(Collection):
    '''
    Bases: :class:`.Collection`

    Collection of C++ chain objects.
    '''

    def __init__(self, chain_pointers):
        Collection.__init__(self, chain_pointers, molobject.Chain, Chains)

    chain_ids = cvec_property('sseq_chain_id', string, read_only = True)
    '''A numpy array of string chain ids for each chain. Read only.'''
    structures = cvec_property('sseq_structure', cptr, astype = _atomic_structures, read_only = True)
    '''A :class:`.StructureDatas` collection containing structures for each chain.'''
    existing_residues = cvec_property('sseq_residues', cptr, 'num_residues',
        astype = _non_null_residues, read_only = True, per_object = False)
    '''A :class:`Residues` containing the existing residues of all chains. Read only.'''
    num_existing_residues = cvec_property('sseq_num_existing_residues', size_t, read_only = True)
    '''A numpy integer array containing the number of existing residues in each chain.'''
    num_residues = cvec_property('sseq_num_residues', size_t, read_only = True)
    '''A numpy integer array containing the number of residues in each chain.'''

    @classmethod
    def session_restore_pointers(cls, session, data):
        structures, chain_ses_ids = data
        return array([s.session_id_to_chain(i) for s, i in zip(structures, chain_ses_ids)])
    def session_save_pointers(self, session):
        structures = self.structures
        chain_ses_ids = [s.session_chain_to_id(ptr) for s, ptr in zip(structures, self._c_pointers)]
        return [structures, array(chain_ses_ids)]

# -----------------------------------------------------------------------------
#
class StructureDatas(Collection):
    '''
    Bases: :class:`.Collection`

    Collection of C++ atomic structure objects.
    '''
    def __init__(self, mol_pointers):
        Collection.__init__(self, mol_pointers, molobject.StructureData, StructureDatas)

    atoms = cvec_property('structure_atoms', cptr, 'num_atoms', astype = _atoms,
                          read_only = True, per_object = False)
    '''A single :class:`.Atoms` containing atoms for all structures. Read only.'''
    bonds = cvec_property('structure_bonds', cptr, 'num_bonds', astype = _bonds,
                          read_only = True, per_object = False)
    '''A single :class:`.Bonds` object containing bonds for all structures. Read only.'''
    chains = cvec_property('structure_chains', cptr, 'num_chains', astype = _chains,
                           read_only = True, per_object = False)
    '''A single :class:`.Chains` object containing chains for all structures. Read only.'''
    lower_case_chains = cvec_property('structure_lower_case_chains', npy_bool, read_only=True)
    '''A numpy bool array of lower_case_names of each structure.'''
    names = cvec_property('structure_name', string)
    '''A numpy string array of names of each structure.'''
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
    
# -----------------------------------------------------------------------------
#
class AtomicStructures(StructureDatas):
    '''
    Bases: :class:`.StructureDatas`

    Collection of Python atomic structure objects.
    '''
    def __init__(self, mol_pointers):
        from .structure import AtomicStructure
        Collection.__init__(self, mol_pointers, AtomicStructure, AtomicStructures)

    @classmethod
    def session_restore_pointers(cls, session, data):
        return array([s._c_pointer.value for s in data])
    def session_save_pointers(self, session):
        return [s for s in self]

# -----------------------------------------------------------------------------
#
class PseudobondGroupDatas(Collection):
    '''
    Bases: :class:`.Collection`

    Collection of C++ pseudobond group objects.
    '''
    def __init__(self, pbg_pointers):
        Collection.__init__(self, pbg_pointers, molobject.PseudobondGroupData,
			PseudobondGroupDatas)

    pseudobonds = cvec_property('pseudobond_group_pseudobonds', cptr, 'num_pseudobonds',
		astype = _pseudobonds, read_only = True, per_object = False)
    '''A single :class:`.Pseudobonds` object containing pseudobonds for all groups. Read only.'''
    names = cvec_property('pseudobond_group_category', string, read_only = True)
    '''A numpy string array of categories of each group.'''
    num_bonds = cvec_property('pseudobond_group_num_pseudobonds', size_t, read_only = True)
    '''Number of pseudobonds in each group. Read only.'''

# -----------------------------------------------------------------------------
#
class PseudobondGroups(PseudobondGroupDatas):
    '''
    Bases: :class:`.PseudobondGroupDatas`

    Collection of Python pseudobond group objects.
    '''
    def __init__(self, pbg_pointers):
        from . import pbgroup
        Collection.__init__(self, pbg_pointers, pbgroup.PseudobondGroup, PseudobondGroups)

    @classmethod
    def session_restore_pointers(cls, session, data):
        return array([s._c_pointer.value for s in data])
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
