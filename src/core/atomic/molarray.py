# vim: set expandtab shiftwidth=4 softtabstop=4:
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
from numpy import uint8, int32, float64, float32, bool as npy_bool, integer, empty, unique, array
from .molc import string, cptr, pyobject, cvec_property, set_cvec_pointer, c_function, pointer, ctype_type_to_numpy
from . import molobject
import ctypes
size_t = ctype_type_to_numpy[ctypes.c_size_t]   # numpy dtype for size_t

def _atoms(a):
    return Atoms(a)
def _bonds(b):
    return Bonds(b)
def _pseudobonds(p):
	return Pseudobonds(p)
def _elements(e):
    return Elements(e)
def _residues(r):
    return Residues(r)
def _non_null_residues(r):
    from .molarray import Residues
    return Residues(r[r!=0])
def _chains(c):
    return Chains(c)
def _atomic_structures(p):
    return AtomicStructures(p)
def _atomic_structure_datas(p):
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
class Collection:
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
        if not isinstance(i,(int,integer)):
            raise IndexError('Only integer indices allowed for %s, got %s'
                % (self.__class__.__name__, str(type(i))))
        from .molobject import object_map
        return object_map(self._pointers[i], self._object_class)
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
        All duplicates are removed.'''
        import numpy
        return self._objects_class(numpy.setdiff1d(self._pointers, objects._pointers))

def concatenate(collections, object_class = None):
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
        from numpy import concatenate as concat
        c = cl(concat([a._pointers for a in collections]))
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
class BaseSpheres(type):
    """'Base class' for Atoms, Spheres, etc.

       We avoid the difficulties of interacting through ctypes to the C++ layer
       by using BaseSpheres as a metaclass for it's 'derived' classes, which places
       properties into them during class layout.
    """
    def __new__(meta, name, bases, attrs):
        related_class_info = {
            'Atoms': ('AtomicStructure', 'structures', _atomic_structures, 'Bond')
        }
        if name != meta.__name__:
            doc_marker = "[BS]"
            doc_add = \
                "{}: Generic attributes added by {}'s {} metaclass (and therefore usable " \
                "by in any class whose metaclass is {}, such as Atoms, Spheres, etc.) " \
                "are noted by preceding their documentation with '{}'.".format(
                doc_marker, name, meta.__name__, meta.__name__, doc_marker)
            if "__doc__" in attrs:
                doc = attrs['__doc__']
                if doc:
                    if doc[-1] == '\n':
                        newlines = '\n'
                    else:
                        newlines = '\n\n'
                else:
                    newlines = ""
            else:
                doc = ""
                newlines = ""
            attrs["__doc__"] = doc + newlines + doc_add

            # define some vars useful later
            sphere_class = name[:-1]
            sphere = sphere_class.lower()
            spheres = sphere + 's'
            (container_class, containers_nickname,
                containers_collective, conn_name) = related_class_info[name]
            container_prep = "an" if container_class[0] in "AEIOU" else "a"
            container_collection = container_class + "s"
            container_nickname = containers_nickname[:-1]
            connection = conn_name.lower()
            connections = connection + "s"
            conns_name = conn_name + "s"

            # Property tuples are: (generic attr name, specific attr name,
            # other args to cvec_property, keywords to cvec_property, doc string).
            # If the specific attr name is None, then it's the same as the
            # generic attr name.
            #
            # Some shared properties (e.g. names) are defined in the C++ layer
            # for some classes but not others.  Those properties are defined
            # directly in the final class rather than via the metaclass.
            properties = [
                ('colors', None, (uint8, 4), {},
                    "Returns a :mod:`numpy` Nx4 array of uint8 RGBA values. Can be "
                    "set with such an array (or equivalent sequence), or with a single "
                    "RGBA value."),
                ('coords', None, (float64, 3), {},
                    "Returns a :mod:`numpy` Nx3 array of XYZ values. Can be set."),
                ('displays', None, (npy_bool,), {},
                    "Controls whether the {} should be displayed. "
                    "Returns a :mod:`numpy` array of boolean values.  Can be "
                    "set with such an array (or equivalent sequence), or with a "
                    "single boolean value.".format(name)),
                ('draw_modes', None, (uint8,), {},
                    "Controls how the {} should be depicted. "
                    "The values are integers, SPHERE_STYLE, BALL_STYLE "
                    "or STICK_STYLE as documented in the :class:`.{}` class. "
                    "Returns a :mod:`numpy` array of integers.  Can be "
                    "set with such an array (or equivalent sequence), or with a "
                    "single integer value. ".format(name, sphere_class)),
                ('graphs', containers_nickname, (cptr,),
                    { 'astype': containers_collective, 'read_only': True },
                    "Returns {} :class:`.{}` for each {}. Read only."
                    .format(container_prep, container_class, sphere)),
                ('radii', None, (float32,), {},
                    "Returns a :mod:`numpy` array of radii.  Can be "
                    "set with such an array (or equivalent sequence), or with a single "
                    "floating-point number."),
                ('selected', None, (npy_bool,), {},
                    "numpy bool array whether each {} is selected.".format(sphere)),
                ('visibles', None, (npy_bool,), { 'read_only': True },
                    "Returns whether the {} should be visible "
                    "(displayed and not hidden). Returns a :mod:`numpy` array "
                    "of boolean values.  Read only.".format(name)),
            ]
            prefix = sphere + "_"
            for generic_attr_name, specific_attr_name, args, kw, doc in properties:
                if specific_attr_name is None:
                    prop_attr_name = depluralize(generic_attr_name)
                else:
                    prop_attr_name = depluralize(specific_attr_name)
                attrs[generic_attr_name] = cvec_property(prefix + prop_attr_name,
                    *args, **kw, doc = doc_marker + " " + doc)
                if specific_attr_name is not None:
                    attrs[specific_attr_name] = cvec_property(prefix + prop_attr_name,
                        *args, **kw, doc = doc)

            # define __init__ method in attrs dict
            exec("def __init__(self, c_pointers = None):\n"
                "    Collection.__init__(self, c_pointers, molobject.{}, {})\n"
                .format(sphere_class, name),
                globals(), attrs)

            # define by_graph property (and class-specific version) in attrs dict
            base_doc = "Return list of 2-tuples of ({}, {} for that {}).".format(
                container_nickname, name, container_nickname)
            for prop_name, doc_add in [
                    ('graph', doc_marker + " "), (container_nickname, "")]:
                exec("@property\n"
                    "def by_{}(self):\n"
                    "    '''{}'''\n"
                    "    agraph = self.graphs._pointers\n"
                    "    return [(g, self.filter(agraph==g._c_pointer.value))\n"
                    "        for g in self.unique_graphs]\n".format(prop_name,
                    doc_add + base_doc), globals(), attrs)

            # define delete method in attrs dict
            exec("def delete(self):\n"
                "    '''Delete the C++ {} objects'''\n"
                "    mols = self.unique_graphs\n"
                "    c_function('{}_delete', args = [ctypes.c_void_p,\n"
                "                ctypes.c_size_t])(self._c_pointers, len(self))\n".format(
                sphere_class, sphere), globals(), attrs)

            # define inter_connections property (and class-specific version) in attrs dict
            base_doc = ":class:`{} object` where both {} are in this collection".format(
                conns_name, spheres)
            for prop_name, doc_add in [
                    ('connections', doc_marker + " "), (connections, "")]:
                exec("@property\n"
                    "def inter_{}(self):\n"
                    "    '''{}'''\n"
                    "    f = c_function('{}_inter_{}',\n"
                    "        args = [ctypes.c_void_p, ctypes.c_size_t],\n"
                    "        ret = ctypes.py_object)\n"
                    "    return {}(f(self._c_pointers, len(self)))\n".format(prop_name,
                    doc_add + base_doc, sphere, connections, conns_name), globals(), attrs)

            # define num_selected property in attrs dict
            exec("@property\n"
                "def num_selected(self):\n"
                "    '''{} Number of selected {}.'''\n"
                "    f = c_function('{}_num_selected',\n"
                "                   args = [ctypes.c_void_p, ctypes.c_size_t],\n"
                "                   ret = ctypes.c_size_t)\n"
                "    return f(self._c_pointers, len(self))\n".format(doc_marker,
                spheres, sphere), globals(), attrs)

            # define scene_bounds property in attrs dict
            exec("@property\n"
                "def scene_bounds(self):\n"
                "    '''Return scene bounds of {} including instances of all parent models.'''\n"
                "    blist = []\n"
                "    from ..geometry import sphere_bounds, copy_tree_bounds, union_bounds\n"
                "    for m, a in self.by_graph:\n"
                "        ba = sphere_bounds(a.coords, a.radii)\n"
                "        ib = copy_tree_bounds(ba,\n"
                "            [d.positions for d in m.drawing_lineage])\n"
                "        blist.append(ib)\n"
                "    return union_bounds(blist)\n".format(spheres), globals(), attrs)

            # define scene_coords property in attrs dict
            exec("@property\n"
                "def scene_coords(self):\n"
                "    '''\n"
                "    {} coordinates in the global scene coordinate system.\n"
                "    This accounts for the :class:`Drawing` positions for the hierarchy\n"
                "    of models each {} belongs to.\n"
                "    '''\n"
                "    n = len(self)\n"
                "    from numpy import array, empty, float64\n"
                "    xyz = empty((n,3), float64)\n"
                "    if n == 0:\n"
                "        return xyz\n"
                "    graphs = self.unique_graphs\n"
                "    gtable = array(tuple(g.scene_position.matrix for g in graphs), float64)\n"
                "    from .molc import pointer\n"
                "    f = c_function('atom_scene_coords',\n"
                "        args = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p,\n"
                "                ctypes.c_size_t, ctypes.c_void_p, ctypes.c_void_p])\n"
                "    f(self._c_pointers, n, graphs._c_pointers, len(graphs),\n"
                "        pointer(gtable), pointer(xyz))\n"
                "    return xyz\n".format(name, sphere), globals(), attrs)

            # define unique_graphs property (and class-specific version) in attrs dict
            base_doc = "The unique {} as an :class:`.{}` collection".format(
                containers_nickname, container_collection)
            for prop_namer, doc_add in [
                    ('graphs', doc_marker + " "), (containers_nickname, "")]:
                exec("@property\n"
                    "def unique_{}(self):\n"
                    "    '''{}'''\n"
                    "    return {}(unique(self.graphs._pointers))\n".format(
                    prop_namer, doc_add + base_doc, container_collection), globals(), attrs)

        return super().__new__(meta, name, bases, attrs)

# -----------------------------------------------------------------------------
#
class Atoms(Collection, metaclass=BaseSpheres):
    '''
    Bases: :class:`.Collection`

    An ordered collection of atom objects. This offers better performance
    than using a list of atoms.  It provides methods to access atom attributes such
    as coordinates as numpy arrays. Atoms directly accesses the C++ atomic data
    without creating Python :py:class:`Atom` objects which require much more memory
    and are slower to use in computation.
    '''

    bfactors = cvec_property('atom_bfactor', float32)
    chain_ids = cvec_property('atom_chain_id', string, read_only = True)
    elements = cvec_property('atom_element', cptr, astype = _elements, read_only = True)
    '''
    Returns a :class:`Elements` whose data items
    correspond in a 1-to-1 fashion with the items in the Atoms.
    Read only. 
    '''
    element_names = cvec_property('atom_element_name', string, read_only = True)
    '''Returns a numpy array of chemical element names. Read only.'''
    element_numbers = cvec_property('atom_element_number', uint8, read_only = True)
    '''Returns a :mod:`numpy` array of atomic numbers (integers). Read only.'''
    in_chains = cvec_property('atom_in_chain', npy_bool, read_only = True)
    '''Whether each atom belong to a polymer. Returns numpy bool array. Read only.'''
    names = cvec_property('atom_name', string)
    '''Returns a numpy array of atom names.  Canbe
    set with such an array (or equivalent sequence), or with a single
    string.  Atom names are limited to 4 characters.'''
    residues = cvec_property('atom_residue', cptr, astype = _residues, read_only = True)
    '''
    Returns a :class:`Residues` whose data items
    correspond in a 1-to-1 fashion with the items in the Atoms.
    Read only. 
    '''
    structure_categories = cvec_property('atom_structure_category', string, read_only=True)
    '''Numpy array of whether atom is ligand, ion, etc.'''

    @property
    def unique_residues(self):
        '''The unique :class:`Residues` for these atoms.'''
        return Residues(unique(self.residues._pointers))

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
    def shown_atoms(self):
        '''
        Subset of Atoms including atoms that are displayed with
        displayed structure and displayed parents.
        '''
        da = self.filter(self.displays | self.residues.ribbon_displays)
        datoms = concatenate([a for m, a in da.by_structure
                              if m.display and m.parents_displayed], Atoms)
        return datoms
    shown_spheres = shown_atoms

    def update_ribbon_visibility(self):
        '''Update the 'hide' status for ribbon control point atoms, which
	are hidden unless any of its neighbors are visible.'''
        f = c_function('atom_update_ribbon_visibility',
                       args = [ctypes.c_void_p, ctypes.c_size_t])
        f(self._c_pointers, len(self))

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
    '''Returns an :class:`.AtomicStructureDatas` with the structure for each bond. Read only.'''

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
    Whether each pseudobond is displayed, visible and has both atoms shown.
    '''

    @property
    def half_colors(self):
        '''2N x 4 RGBA uint8 numpy array of half bond colors.'''
        f = c_function('pseudobond_half_colors', args = [ctypes.c_void_p, ctypes.c_size_t], ret = ctypes.py_object)
        return f(self._c_pointers, len(self))

# -----------------------------------------------------------------------------
#
class Residues(Collection):
    '''
    Bases: :class:`.Collection`

    Collection of C++ residue objects.
    '''
    def __init__(self, residue_pointers = None):
        Collection.__init__(self, residue_pointers, molobject.Residue, Residues)

    atoms = cvec_property('residue_atoms', cptr, 'num_atoms', astype = _atoms, read_only = True, per_object = False)
    '''Return :class:`.Atoms` belonging to each residue all as a single collection. Read only.'''
    chain_ids = cvec_property('residue_chain_id', string, read_only = True)
    '''Returns a numpy array of chain IDs. Read only.'''
    is_helix = cvec_property('residue_is_helix', npy_bool)
    '''Returns a numpy bool array whether each residue is in a protein helix. Read only.'''
    is_sheet = cvec_property('residue_is_sheet', npy_bool)
    '''Returns a numpy bool array whether each residue is in a protein sheet. Read only.'''
    structures = cvec_property('residue_structure', cptr, astype = _atomic_structures, read_only = True)
    '''Returns :class:`.AtomicStructureDatas` collection containing structures for each residue.'''
    names = cvec_property('residue_name', string, read_only = True)
    '''Returns a numpy array of residue names. Read only.'''
    num_atoms = cvec_property('residue_num_atoms', size_t, read_only = True)
    '''Returns a numpy integer array of the number of atoms in each residue. Read only.'''
    numbers = cvec_property('residue_number', int32, read_only = True)
    '''
    Returns a :mod:`numpy` array of residue sequence numbers, as provided by
    whatever data source the structure came from, so not necessarily consecutive,
    or starting from 1, *etc.* Read only.
    '''
    ss_id = cvec_property('residue_ss_id', int32)
    '''
    numpy array of integer secondary structure ids.
    '''
    strs = cvec_property('residue_str', string, read_only = True)
    '''
    Returns a numpy array of strings that encapsulates each
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
    '''A numpy bool array whether to display each residue in ribbon style.'''
    ribbon_colors = cvec_property('residue_ribbon_color', uint8, 4)
    '''
    A :mod:`numpy` Nx4 array of uint8 RGBA values.  Can be
    set with such an array (or equivalent sequence), or with a single
    RGBA value.
    '''
    ribbon_styles = cvec_property('residue_ribbon_style', int32)
    '''A numpy int array of cartoon styles.  See constants in :class:Ribbon.'''
    ribbon_adjusts = cvec_property('residue_ribbon_adjust', float32)
    '''A numpy float array of adjustment factors for the position of ribbon
    control points.  Factors range from zero to one, with zero being using the
    actual atomic coordinates as control point, and one being using the idealized
    secondary structure position as control point.  A negative value means to
    use the default of zero for turns and helices and 0.7 for strands.'''
    ribbon_hide_backbones = cvec_property('residue_ribbon_hide_backbone', npy_bool)
    '''A :mod:`numpy` array of booleans. Whether a ribbon automatically hides
    the residue backbone atoms.'''

    @property
    def unique_structures(self):
        '''The unique structures as an :class:`.AtomicStructureDatas` collection'''
        return AtomicStructureDatas(unique(self.structures._pointers))

    @property
    def unique_chain_ids(self):
        '''The unique chain IDs as a numpy array of strings.'''
        return unique(self.chain_ids)

    @property
    def by_structure(self):
        '''Return list of pairs of structure and Residues for that structure.'''
        rmol = self.structures._pointers
        return [(m, self.filter(rmol==m._c_pointer.value)) for m in self.unique_structures]

    def get_polymer_spline(self):
        '''Return a tuple of spline center and guide coordinates for a
	polymer chain.  Residues in the chain that do not have a center
	atom will have their display bit turned off.  Center coordinates
	are returned as a numpy array.  Guide coordinates are only returned
	if all spline atoms have matching guide atoms; otherwise, None is
	returned for guide coordinates.'''
        f = c_function('residue_polymer_spline',
                       args = [ctypes.c_void_p, ctypes.c_size_t],
                       ret = ctypes.py_object)
        atom_pointers, centers, guides = f(self._c_pointers, len(self))
        atoms = Atoms(atom_pointers)
        return atoms, centers, guides

# -----------------------------------------------------------------------------
#
class Chains(Collection):
    '''
    Bases: :class:`.Collection`

    Collection of C++ chain objects.
    '''

    def __init__(self, chain_pointers):
        Collection.__init__(self, chain_pointers, molobject.Chain, Chains)

    chain_ids = cvec_property('chain_chain_id', string, read_only = True)
    '''A numpy array of string chain ids for each chain. Read only.'''
    structures = cvec_property('chain_structure', cptr, astype = _atomic_structures, read_only = True)
    '''A :class:`.AtomicStructureDatas` collection containing structures for each chain.'''
    existing_residues = cvec_property('chain_residues', cptr, 'num_residues',
        astype = _non_null_residues, read_only = True, per_object = False)
    '''A :class:`Residues` containing the existing residues of all chains. Read only.'''
    num_existing_residues = cvec_property('chain_num_existing_residues', size_t, read_only = True)
    '''A numpy integer array containing the number of existing residues in each chain.'''
    num_residues = cvec_property('chain_num_residues', size_t, read_only = True)
    '''A numpy integer array containing the number of residues in each chain.'''

# -----------------------------------------------------------------------------
#
class AtomicStructureDatas(Collection):
    '''
    Bases: :class:`.Collection`

    Collection of C++ atomic structure objects.
    '''
    def __init__(self, mol_pointers):
        Collection.__init__(self, mol_pointers, molobject.AtomicStructureData, AtomicStructureDatas)

    atoms = cvec_property('structure_atoms', cptr, 'num_atoms', astype = _atoms,
                          read_only = True, per_object = False)
    '''A single :class:`.Atoms` containing atoms for all structures. Read only.'''
    bonds = cvec_property('structure_bonds', cptr, 'num_bonds', astype = _bonds,
                          read_only = True, per_object = False)
    '''A single :class:`.Bonds` object containing bonds for all structures. Read only.'''
    chains = cvec_property('structure_chains', cptr, 'num_chains', astype = _chains,
                           read_only = True, per_object = False)
    '''A single :class:`.Chains` object containing chains for all structures. Read only.'''
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

# -----------------------------------------------------------------------------
#
class AtomicStructures(AtomicStructureDatas):
    '''
    Bases: :class:`.AtomicStructureDatas`

    Collection of Python atomic structure objects.
    '''
    def __init__(self, mol_pointers):
        from . import structure
        Collection.__init__(self, mol_pointers, structure.AtomicStructure, AtomicStructures)

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

# -----------------------------------------------------------------------------
# When C++ object is deleted, delete it from the specified pointer array.
#
def remove_deleted_pointers(array):
    remove_deleted_c_pointers(array)
    import weakref
    weakref.finalize(array, pointer_array_freed, id(array))

remove_deleted_c_pointers = c_function('remove_deleted_c_pointers', args = [ctypes.py_object])
pointer_array_freed = c_function('pointer_array_freed', args = [ctypes.c_void_p])
