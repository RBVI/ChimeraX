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

from chimerax.core.state import State, StateManager
from numpy import uint8, int32, uint32, float64, float32, byte, bool as npy_bool
from .molc import CFunctions, string, cptr, pyobject, set_c_pointer, pointer, size_t
import ctypes

# -------------------------------------------------------------------------------
# Access functions from libmolc C library.
#
_atomic_c_functions = CFunctions('libmolc')
c_property = _atomic_c_functions.c_property
cvec_property = _atomic_c_functions.cvec_property
c_function = _atomic_c_functions.c_function
c_array_function = _atomic_c_functions.c_array_function

# -------------------------------------------------------------------------------
# These routines convert C++ pointers to Python objects and are used for defining
# the object properties.
#
def _atoms(p):
    from .molarray import Atoms
    return Atoms(p)
def _atom_pair(p):
    return (Atom.c_ptr_to_py_inst(p[0]), Atom.c_ptr_to_py_inst(p[1]))
def _atom_or_none(p):
    return Atom.c_ptr_to_py_inst(p) if p else None
def _bonds(p):
    from .molarray import Bonds
    return Bonds(p)
def _chain(p):
    if not p:
        return None
    return Chain.c_ptr_to_py_inst(p)
def _coordset(p):
    return CoordSet.c_ptr_to_py_inst(p)
def _element(p):
    return Element.c_ptr_to_py_inst(p)
def _pseudobonds(p):
    from .molarray import Pseudobonds
    return Pseudobonds(p)
def _residue(p):
    return Residue.c_ptr_to_py_inst(p)
def _residues(p):
    from .molarray import Residues
    return Residues(p)
def _rings(p):
    from .molarray import Rings
    return Rings(p)
def _non_null_residues(p):
    from .molarray import Residues
    return Residues(p[p!=0])
def _residue_or_none(p):
    return Residue.c_ptr_to_py_inst(p) if p else None
def _residues_or_nones(p):
    return [_residue_or_none(rptr) for rptr in p]
def _chains(p):
    from .molarray import Chains
    return Chains(p)
def _atomic_structure(p):
    return StructureData.c_ptr_to_py_inst(p) if p else None
def _pseudobond_group(p):
    return PseudobondGroupData.c_ptr_to_py_inst(p)
def _pseudobond_group_map(pbgc_map):
    pbg_map = dict((name, _pseudobond_group(pbg)) for name, pbg in pbgc_map.items())
    return pbg_map

def has_custom_attrs(klass, inst):
    for attr_name, attr_info in klass._attr_registration.reg_attr_info.items():
        if hasattr(inst, attr_name):
            return True
    return False

def get_custom_attrs(klass, inst):
    custom_attrs = []
    from .attr_registration import NO_DEFAULT
    for attr_name, attr_info in klass._attr_registration.reg_attr_info.items():
        if hasattr(inst, attr_name):
            registrant, default_value, attr_type = attr_info
            val = getattr(inst, attr_name)
            if default_value == NO_DEFAULT or val != default_value:
                custom_attrs.append((attr_name, val))
    return custom_attrs

def set_custom_attrs(inst, ses_data):
    for attr_name, val in ses_data.get('custom attrs', []):
        setattr(inst, attr_name, val)

def all_python_instances():
    f = c_function('all_python_instances', args = (), ret = ctypes.py_object)
    return f()

from .cymol import CyAtom
class Atom(CyAtom, State):

    # used by custom-attr registration code
    @property
    def has_custom_attrs(self):
        return has_custom_attrs(Atom, self)

    def take_snapshot(self, session, flags):
        data = {'structure': self.structure,
                'ses_id': self.structure.session_atom_to_id(self._c_pointer),
                'custom attrs': get_custom_attrs(Atom, self)}
        return data

    @staticmethod
    def restore_snapshot(session, data):
        a = Atom.c_ptr_to_py_inst(data['structure'].session_id_to_atom(data['ses_id']))
        set_custom_attrs(a, data)
        return a
Atom.set_py_class(Atom)

# -----------------------------------------------------------------------------
#
class Bond(State):
    '''
    Bond connecting two atoms.

    To create a Bond use the :class:`.AtomicStructure` new_bond() method.
    '''
    def __init__(self, bond_pointer):
        set_c_pointer(self, bond_pointer)

    # cpp_pointer and deleted are "base class" methods, though for performance reasons
    # we are placing them directly in each class rather than using a base class,
    # and for readability by most programmers we avoid using metaclasses
    @property
    def cpp_pointer(self):
        '''Value that can be passed to C++ layer to be used as pointer (Python int)'''
        return self._c_pointer.value

    @property
    def deleted(self):
        '''Has the C++ side been deleted?'''
        return not hasattr(self, '_c_pointer')

    def __lt__(self, other):
        # for sorting (objects of the same type)
        s1, s2 = self.atoms
        o1, o2 = other.atoms
        return s1 < o1 if s1 != o1 else s2 < o2

    def __str__(self):
        return self.string()

    def atomspec(self):
        return a1.atomspec() + a2.atomspec()

    atoms = c_property('bond_atoms', cptr, 2, astype = _atom_pair, read_only = True,
        doc = "Supported API. "
        "Two-tuple of :py:class:`Atom` objects that are the bond end points.")
    color = c_property('bond_color', uint8, 4, doc =
        "Supported API. Color RGBA length 4 sequence/array. Values in range 0-255")
    display = c_property('bond_display', npy_bool, doc =
        "Supported API.  Whether to display the bond if both atoms are shown. "
        "Can be overriden by the hide attribute.")
    halfbond = c_property('bond_halfbond', npy_bool, doc = "Supported API. Whether to "
        "color the each half of the bond nearest an end atom to match that atom color, "
        "or use a single color and the bond color attribute.  Boolean value.")
    radius = c_property('bond_radius', float32,
        doc = "Displayed cylinder radius for the bond.")
    hide = c_property('bond_hide', int32, doc = "Supported API. Whether bond is hidden "
        "(overrides display). Integer bitmask. Use Atom.HIDE_* constants for hide bits.")
    def set_hide_bits(self, bit_mask):
        "Set Atom's hide bits in bit mask"
        f = c_array_function('set_bond_hide_bits', args=(uint32,), per_object=False)
        b_ref = ctypes.byref(self._c_pointer)
        f(b_ref, 1, bit_mask)
    def clear_hide_bits(self, bit_mask):
        "Clear Atom's hide bits in bit mask"
        f = c_array_function('clear_bond_hide_bits', args=(uint32,), per_object=False)
        b_ref = ctypes.byref(self._c_pointer)
        f(b_ref, 1, bit_mask)
    selected = c_property('bond_selected', npy_bool, doc =
        "Supported API. Whether the bond is selected.")
    ends_selected = c_property('bond_ends_selected', npy_bool, read_only = True,
        doc = "Whether both bond end atoms are selected.")
    shown = c_property('bond_shown', npy_bool, read_only = True, doc = "Supported API. "
        "Whether bond is visible and both atoms are shown and at least one is not "
        " Sphere style. Read only.")
    structure = c_property('bond_structure', pyobject, read_only = True, doc =
        "Supported API. :class:`.AtomicStructure` the bond belongs to.")
    visible = c_property('bond_visible', npy_bool, read_only = True, doc =
        "Supported API. Whether bond is display and not hidden. Read only.")
    length = c_property('bond_length', float32, read_only = True, doc =
        "Supported API. Bond length. Read only.")

    def other_atom(self, atom):
        "Supported API. 'atom' should be one of the atoms in the bond.  Return the other atom."
        f = c_function('bond_other_atom', args = (ctypes.c_void_p, ctypes.c_void_p), ret = ctypes.c_void_p)
        o = f(self._c_pointer, atom._c_pointer)
        return Atom.c_ptr_to_py_inst(o)

    def delete(self):
        "Supported API. Delete this Bond from it's Structure"
        f = c_function('bond_delete', args = (ctypes.c_void_p, ctypes.c_size_t))
        c = f(self._c_pointer_ref, 1)

    # used by custom-attr registration code
    @property
    def has_custom_attrs(self):
        return has_custom_attrs(Bond, self)

    def rings(self, cross_residues=False, all_size_threshold=0):
        '''Return :class:`.Rings` collection of rings this Bond is involved in.

        If 'cross_residues' is False, then rings that cross residue boundaries are not
        included.  If 'all_size_threshold' is zero, then return only minimal rings, of
        any size.  If it is greater than zero, then return all rings not larger than the
        given value.

        The returned rings are quite emphemeral, and shouldn't be cached or otherwise
        retained for long term use.  They may only live until the next call to rings()
        [from anywhere, including C++].
        '''
        f = c_function('bond_rings', args = (ctypes.c_void_p, ctypes.c_bool, ctypes.c_int),
                ret = ctypes.py_object)
        return _rings(f(self._c_pointer, cross_residues, all_size_threshold))

    @property
    def session(self):
        "Session that this Bond is in"
        return self.structure.session

    def side_atoms(self, side_atom):
        '''All the atoms on the same side of the bond as side_atom.

           'side_atom' has to be one of the two bond atoms, and the returned atoms will include
           'side_atom'.  Missing-structure pseudobonds are treated as connecting their atoms for
           the purpose of computing the side atoms.  If bond is part of a ring or cycle then
           ValueError will be thrown.
        '''
        f = c_function('bond_side_atoms', args = (ctypes.c_void_p, ctypes.c_void_p),
            ret = ctypes.py_object)
        return _atoms(f(self._c_pointer, side_atom._c_pointer))

    @property
    def smaller_side(self):
        '''Returns the bond atom on the side of the bond with fewer total atoms attached'''
        f = c_function('bond_smaller_side', args = (ctypes.c_void_p,), ret = ctypes.py_object)
        return f(self._c_pointer)

    @property
    def polymeric_start_atom(self):
        f = c_function('bond_polymeric_start_atom', args = (ctypes.c_void_p,), ret = ctypes.py_object)
        return f(self._c_pointer)

    def string(self, style = None):
        "Supported API.  Get text representation of Bond"
        " (also used by __str__ for printing)"
        a1, a2 = self.atoms
        bond_sep = " \N{Left Right Arrow} "
        return a1.string(style=style) + bond_sep + a2.string(style=style, relative_to=a1)

    def take_snapshot(self, session, flags):
        data = {'structure': self.structure,
                'ses_id': self.structure.session_bond_to_id(self._c_pointer),
                'custom attrs': get_custom_attrs(Bond, self)}
        return data

    @staticmethod
    def restore_snapshot(session, data):
        b = Bond.c_ptr_to_py_inst(data['structure'].session_id_to_bond(data['ses_id']))
        set_custom_attrs(b, data)
        return b

# -----------------------------------------------------------------------------
#
class Pseudobond(State):
    '''
    A Pseudobond is a graphical line between atoms, for example depicting a distance
    or a gap in an amino acid chain, often shown as a dotted or dashed line.
    Pseudobonds can join atoms belonging to different :class:`.AtomicStructure`\\ s
    which is not possible with a :class:`Bond`\\ .

    To create a Pseudobond use the :class:`PseudobondGroup` new_pseudobond() method.
    '''
    def __init__(self, pbond_pointer):
        set_c_pointer(self, pbond_pointer)

    # cpp_pointer and deleted are "base class" methods, though for performance reasons
    # we are placing them directly in each class rather than using a base class,
    # and for readability by most programmers we avoid using metaclasses
    @property
    def cpp_pointer(self):
        '''Value that can be passed to C++ layer to be used as pointer (Python int)'''
        return self._c_pointer.value

    @property
    def deleted(self):
        '''Has the C++ side been deleted?'''
        return not hasattr(self, '_c_pointer')

    __lt__ = Bond.__lt__
    __str__ = Bond.__str__
    string = Bond.string

    atoms = c_property('pseudobond_atoms', cptr, 2, astype = _atom_pair, read_only = True,
        doc = "Supported API. Two-tuple of :py:class:`Atom` objects that are the bond end points.")
    color = c_property('pseudobond_color', uint8, 4,
        doc = "Supported API. Color RGBA length 4 sequence/array. Values in range 0-255")
    display = c_property('pseudobond_display', npy_bool, doc =
        "Whether to display the bond if both atoms are shown. "
        "Can be overriden by the hide attribute.")
    group = c_property('pseudobond_group', cptr, astype = _pseudobond_group, read_only = True,
        doc = "Supported API. :py:class:`.pbgroup.PseudobondGroup` that this pseudobond belongs to")
    halfbond = c_property('pseudobond_halfbond', npy_bool, doc =
        "Supported API. Whether to color the each half of the bond nearest an end atom to match "
        " that atom color, or use a single color and the bond color attribute.  Boolean value.")
    radius = c_property('pseudobond_radius', float32, doc =
        "Displayed cylinder radius for the bond.")
    selected = c_property('pseudobond_selected', npy_bool, doc =
        "Supported API. Whether the pseudobond is selected.")
    shown = c_property('pseudobond_shown', npy_bool, read_only = True, doc =
        "Supported API. Whether bond is visible and both atoms are shown. Read only.")
    shown_when_atoms_hidden = c_property('pseudobond_shown_when_atoms_hidden', npy_bool, doc =
    '''Normally, whether a pseudbond is shown only depends on the endpoint atoms' 'display'
    attribute and not on those atoms' 'hide' attribute, on the theory that the hide bits
    are only set when the atoms are being depicted by some non-default representation (such
    as ribbons) and that therefore the pseudobonds should still display to "hidden" atoms.
    However, if 'shown_when_atoms_hidden' is False then the pseudobonds will never be displayed
    if either endpoint atom is hidden (regardless of the display attribute), but will honor
    the 'display' attribute when the atoms aren't hidden.  Defaults to True.''')

    def delete(self):
        "Supported API. Delete this pseudobond from it's group"
        f = c_function('pseudobond_delete', args = (ctypes.c_void_p, ctypes.c_size_t))
        c = f(self._c_pointer_ref, 1)

    @property
    def length(self):
        "Supported API. Distance between centers of two bond end point atoms."
        a1, a2 = self.atoms
        v = a1.scene_coord - a2.scene_coord
        from math import sqrt
        return sqrt((v*v).sum())

    def other_atom(self, atom):
        "Supported API. 'atom' should be one of the atoms in the bond.  Return the other atom."
        a1,a2 = self.atoms
        return a2 if atom is a1 else a1

    @property
    def session(self):
        "Session that this Pseudobond is in"
        return self.atoms[0].structure.session

    _ses_id = c_property('pseudobond_get_session_id', int32, read_only = True,
        doc="Used by session save/restore internals")

    # used by custom-attr registration code
    @property
    def has_custom_attrs(self):
        return has_custom_attrs(Pseudobond, self)

    def take_snapshot(self, session, flags):
        data = {'group': self.group,
                'ses_id': self._ses_id,
                'custom attrs': get_custom_attrs(Pseudobond, self)}
        return data

    @staticmethod
    def restore_snapshot(session, data):
        f = c_function('pseudobond_group_resolve_session_id',
            args = [ctypes.c_void_p, ctypes.c_int], ret = ctypes.c_void_p)
        if isinstance(data, dict):
            group = data['group']
            ses_id = data['ses_id']
        else:
            group, ses_id = data
        pb = Pseudobond.c_ptr_to_py_inst(f(group._c_pointer, ses_id))
        if isinstance(data, dict):
            set_custom_attrs(pb, data)
        return pb

# -----------------------------------------------------------------------------
#
class PseudobondGroupData:
    '''
    A group of pseudobonds typically used for one purpose such as display
    of distances or missing segments.

    This base class of :class:`.PseudobondGroup` represents the C++ data while
    the derived class handles rendering the pseudobonds.

    To create a PseudobondGroup use the :class:`PseudobondManager` get_group() method.
    '''

    GROUP_TYPE_NORMAL = 1
    GROUP_TYPE_COORD_SET = 2

    def __init__(self, pbg_pointer):
        set_c_pointer(self, pbg_pointer)

    # cpp_pointer and deleted are "base class" methods, though for performance reasons
    # we are placing them directly in each class rather than using a base class,
    # and for readability by most programmers we avoid using metaclasses
    @property
    def cpp_pointer(self):
        '''Value that can be passed to C++ layer to be used as pointer (Python int)'''
        return self._c_pointer.value

    @property
    def deleted(self):
        '''Has the C++ side been deleted?'''
        return not hasattr(self, '_c_pointer')

    # Model class uses _name, so...
    _category = c_property('pseudobond_group_category', string, read_only = True,
        doc = "Name of the pseudobond group.  Read only string.")
    color = c_property('pseudobond_group_color', uint8, 4,
        doc="Sets the color attribute of current pseudobonds and new pseudobonds")
    group_type = c_property('pseudobond_group_group_type', uint8, read_only = True, doc=
        "PseudobondGroup.GROUP_TYPE_NORMAL is a normal group,"
        "PseudobondGroup.GROUP_TYPE_COORD_SET is a per-coord-set pseudobond group")
    halfbond = c_property('pseudobond_group_halfbond', npy_bool,
        doc = "Sets the halfbond attribute of current pseudobonds and new pseudobonds")
    num_pseudobonds = c_property('pseudobond_group_num_pseudobonds', size_t, read_only = True,
        doc = "Number of pseudobonds in group. Read only.")
    pseudobonds = c_property('pseudobond_group_pseudobonds', cptr, 'num_pseudobonds',
        astype = _pseudobonds, read_only = True,
        doc = "Group pseudobonds as a :class:`.Pseudobonds` collection. Read only.")
    radius = c_property('pseudobond_group_radius', float32,
        doc = "Sets the radius attribute of current pseudobonds and new pseudobonds")
    structure = c_property('pseudobond_group_structure', pyobject,
        read_only = True, doc ="Structure that pseudobond group is owned by.  "
        "Returns None if called on a group managed by the global pseudobond manager")

    def change_name(self, name):
        f = c_function('pseudobond_group_change_category',
            args = (ctypes.c_void_p, ctypes.c_char_p))
        try:
            f(self._c_pointer, name.encode('utf-8'))
        except TypeError:
            from chimerax.core.errors import UserError
            raise UserError("Another pseudobond group is already named '%s'" % name)

    def clear(self):
        '''Delete all pseudobonds in group'''
        f = c_function('pseudobond_group_clear', args = (ctypes.c_void_p,))
        f(self._c_pointer)

    def delete_pseudobond(self, pb):
        '''Delete a specific pseudobond from a group'''
        f = c_function('pseudobond_group_delete_pseudobond',
            args = (ctypes.c_void_p, ctypes.c_void_p))
        f(self._c_pointer, pb._c_pointer)

    def get_num_pseudobonds(self, cs_id):
        '''Get the number of pseudobonds for a particular coordinate set. Use the 'num_pseudobonds'
        property to get the number of pseudobonds for the current coordinate set.'''
        f = c_function('pseudobond_group_get_num_pseudobonds',
                       args = (ctypes.c_void_p, ctypes.c_int,),
                       ret = ctypes.c_size_t)
        return f(self._c_pointer, cs_id)

    def get_pseudobonds(self, cs_id):
        '''Get the pseudobonds for a particular coordinate set. Use the 'pseudobonds'
        property to get the pseudobonds for the current coordinate set.'''
        from numpy import empty
        ai = empty((self.get_num_pseudobonds(cs_id),), cptr)
        f = c_function('pseudobond_group_get_pseudobonds',
                       args = (ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p),
                       ret = ctypes.c_void_p)
        f(self._c_pointer, cs_id, pointer(ai))
        return _pseudobonds(ai)

    def new_pseudobond(self, atom1, atom2, cs_id = None):
        '''Create a new pseudobond between the specified :class:`Atom` objects.
        If the pseudobond group supports per-coordset pseudobonds, you may
        specify a coordinate set ID (defaults to the current coordinate set).'''
        if cs_id is None:
            f = c_function('pseudobond_group_new_pseudobond',
                           args = (ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p),
                           ret = ctypes.py_object)
            pb = f(self._c_pointer, atom1._c_pointer, atom2._c_pointer)
        else:
            f = c_function('pseudobond_group_new_pseudobond_csid',
                           args = (ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int),
                           ret = ctypes.py_object)
            pb = f(self._c_pointer, atom1._c_pointer, atom2._c_pointer, cs_id)
        return pb

    # Graphics changed flags used by rendering code.  Private.
    _SHAPE_CHANGE = 0x1
    _COLOR_CHANGE = 0x2
    _SELECT_CHANGE = 0x4
    _RIBBON_CHANGE = 0x8
    _ADDDEL_CHANGE = 0x10
    _DISPLAY_CHANGE = 0x20
    _ALL_CHANGE = 0x2f
    _graphics_changed = c_property('pseudobond_group_graphics_change', int32)


# -----------------------------------------------------------------------------
#
class PseudobondManager(StateManager):
    '''Per-session singleton pseudobond manager keeps track of all
    :class:`.PseudobondGroupData` objects.'''

    def __init__(self, session):
        self.session = session
        f = c_function('pseudobond_create_global_manager', args = (ctypes.c_void_p,),
            ret = ctypes.c_void_p)
        set_c_pointer(self, f(session.change_tracker._c_pointer))
        self.session.triggers.add_handler("begin save session",
            lambda *args: self._ses_call("save_setup"))
        self.session.triggers.add_handler("end save session",
            lambda *args: self._ses_call("save_teardown"))
        self.session.triggers.add_handler("begin restore session",
            lambda *args: self._ses_call("restore_setup"))
        self.session.triggers.add_handler("end restore session",
            lambda *args: self._ses_call("restore_teardown"))

    def _delete_group(self, pbg):
        f = c_function('pseudobond_global_manager_delete_group',
                       args = (ctypes.c_void_p, ctypes.c_void_p), ret = None)
        f(self._c_pointer, pbg._c_pointer)

    def get_group(self, name, create = True):
        '''Get an existing :class:`.PseudobondGroup` or create a new one with the given name.'''
        f = c_function('pseudobond_global_manager_get_group',
                       args = (ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int),
                       ret = ctypes.c_void_p)
        pbg = f(self._c_pointer, name.encode('utf-8'), create)
        if not pbg:
            return None
        # C++ layer doesn't know how to create Python global pseudobond groups, because it can't
        # supply session arg, so see if the group already exists (and return that if so),
        # otherwise create the group and inform the C++ layer
        inst = PseudobondGroupData.c_ptr_to_existing_py_inst(pbg)
        if not inst:
            from .pbgroup import PseudobondGroup
            inst = PseudobondGroup(pbg, session=self.session)
            f = c_function('set_pseudobondgroup_py_instance', args = (ctypes.c_void_p, ctypes.py_object))
            f(pbg, inst)
        return inst

    @property
    def group_map(self):
        '''Returns a dict that maps from :class:`.PseudobondGroup` name to group'''
        f = c_function('pseudobond_global_manager_group_map',
                       args = (ctypes.c_void_p,),
                       ret = ctypes.py_object)
        ptr_map = f(self._c_pointer)
        obj_map = {}
        for cat, pbg_ptr in ptr_map.items():
            # get the python pbg instance if it already exists; otherwise create it
            # and inform the C++ layer
            obj = PseudobondGroupData.c_ptr_to_existing_py_inst(pbg_ptr)
            if not obj:
                from .pbgroup import PseudobondGroup
                obj = PseudobondGroup(pbg_ptr, session=self.session)
                f = c_function('set_pbgroup_py_instance',
                    args = (ctypes.c_void_p, ctypes.py_object))
                f(pbg_ptr, obj)
            obj_map[cat] = obj
        return obj_map

    # used by custom-attr registration code
    @property
    def has_custom_attrs(self):
        return has_custom_attrs(PseudobondManager, self)

    def take_snapshot(self, session, flags):
        '''Gather session info; return version number'''
        f = c_function('pseudobond_global_manager_session_info',
                    args = (ctypes.c_void_p, ctypes.py_object), ret = ctypes.c_int)
        retvals = []
        version = f(self._c_pointer, retvals)
        # remember the structure->int mapping the pseudobonds used...
        f = c_function('pseudobond_global_manager_session_save_structure_mapping',
                       args = (ctypes.c_void_p,), ret = ctypes.py_object)
        ptr_map = f(self._c_pointer)
        # mapping is ptr->int, change to int->obj
        obj_map = {}
        for ptr, ses_id in ptr_map.items():
            # shouldn't be _creating_ any objects, so pass None as the type
            obj_map[ses_id] = PseudobondGroup.c_ptr_to_py_inst(ptr)
        data = {'version': version,
                'mgr data':retvals,
                'structure mapping': obj_map,
                'custom attrs': get_custom_attrs(PseudobondManager, self)}
        return data

    @staticmethod
    def restore_snapshot(session, data):
        pbm = session.pb_manager
        # restore the int->structure mapping the pseudobonds use...
        ptr_mapping = {}
        for ses_id, structure in data['structure mapping'].items():
            ptr_mapping[ses_id] = structure._c_pointer.value
        f = c_function('pseudobond_global_manager_session_restore_structure_mapping',
                       args = (ctypes.c_void_p, ctypes.py_object))
        f(pbm._c_pointer, ptr_mapping)
        ints, floats, misc = data['mgr data']
        f = c_function('pseudobond_global_manager_session_restore',
                args = (ctypes.c_void_p, ctypes.c_int,
                        ctypes.py_object, ctypes.py_object, ctypes.py_object))
        f(pbm._c_pointer, data['version'], ints, floats, misc)
        set_custom_attrs(pbm, data)
        return pbm

    def reset_state(self, session):
        # Need to call delete() on the models, since just clearing out the C++
        # will cause an error when the last reference to the Python object goes
        # away, which causes delete() to get called
        for pbg in list(self.group_map.values()):
            pbg.delete()

    def _ses_call(self, func_qual):
        f = c_function('pseudobond_global_manager_session_' + func_qual, args=(ctypes.c_void_p,))
        f(self._c_pointer)


# -----------------------------------------------------------------------------
#
class Residue(State):
    '''
    A group of atoms such as an amino acid or nucleic acid. Every atom in
    an :class:`.AtomicStructure` belongs to a residue, including solvent and ions.

    To create a Residue use the :class:`.AtomicStructure` new_residue() method.
    '''

    SS_COIL = 0
    SS_HELIX = 1
    SS_SHEET = SS_STRAND = 2

    water_res_names = set(["HOH", "WAT", "H2O", "D2O", "TIP3"])

    def __init__(self, residue_pointer):
        set_c_pointer(self, residue_pointer)

    # cpp_pointer and deleted are "base class" methods, though for performance reasons
    # we are placing them directly in each class rather than using a base class,
    # and for readability by most programmers we avoid using metaclasses
    @property
    def cpp_pointer(self):
        '''Value that can be passed to C++ layer to be used as pointer (Python int)'''
        return self._c_pointer.value

    def delete(self):
        '''Delete this Residue from it's Structure'''
        f = c_function('residue_delete', args = (ctypes.c_void_p, ctypes.c_size_t))
        c = f(self._c_pointer_ref, 1)

    @property
    def deleted(self):
        '''Has the C++ side been deleted?'''
        return not hasattr(self, '_c_pointer')

    def __lt__(self, other):
        # for sorting (objects of the same type)
        if self.structure != other.structure:
            return self.structure < other.structure

        if self.chain_id != other.chain_id:
            return self.chain_id < other.chain_id

        return self.number < other.number \
            if self.number != other.number else self.insertion_code < other.insertion_code

    def __str__(self):
        return self.string()

    def atomspec(self):
        res_str = ":" + str(self.number) + self.insertion_code
        chain_str = '/' + self.chain_id if not self.chain_id.isspace() else ""
        return self.structure.atomspec() + chain_str + res_str

    atoms = c_property('residue_atoms', cptr, 'num_atoms', astype = _atoms, read_only = True,
        doc = "Supported API. :class:`.Atoms` collection containing all atoms of the residue.")
    center = c_property('residue_center', float64, 3, read_only = True,
        doc = "Average of atom positions as a numpy length 3 array, 64-bit float values.")
    chain = c_property('residue_chain', cptr, astype = _chain, read_only = True,
        doc = "Supported API. :class:`.Chain` that this residue belongs to, if any. Read only.")
    chain_id = c_property('residue_chain_id', string, read_only = True,
        doc = "Supported API. PDB chain identifier. Limited to 4 characters. Read only string.")
    mmcif_chain_id = c_property('residue_mmcif_chain_id', string, read_only = True,
        doc = "mmCIF chain identifier. Limited to 4 characters. Read only string.")
    @property
    def description(self):
        '''Description of residue (if available) from HETNAM/HETSYN records or equivalent'''
        return getattr(self.structure, '_hetnam_descriptions', {}).get(self.name, None)
    insertion_code = c_property('residue_insertion_code', string,
        doc = "Supported API. PDB residue insertion code. 1 character or empty string.")
    is_helix = c_property('residue_is_helix', npy_bool, doc=
        "Supported API. Whether this residue belongs to a protein alpha helix. Boolean value. ")
    is_strand = c_property('residue_is_strand', npy_bool, doc=
        "Supported API. Whether this residue belongs to a protein beta sheet. Boolean value. ")
    PT_NONE = 0
    "Residue polymer type = none."
    PT_AMINO = 1
    "Residue polymer type = amino acid."
    PT_NUCLEIC = 2
    "Residue polymer type = nucleotide."
    polymer_type = c_property('residue_polymer_type', uint8, read_only = True,
        doc = "Supported API.  Polymer type of residue. Integer value.")
    name = c_property('residue_name', string,
        doc = "Supported API. Residue name. Maximum length 4 characters.")
    num_atoms = c_property('residue_num_atoms', size_t, read_only = True,
        doc = "Supported API. Number of atoms belonging to the residue. Read only.")
    number = c_property('residue_number', int32, read_only = True,
        doc = "Supported API. Integer sequence position number from input data file. Read only.")
    principal_atom = c_property('residue_principal_atom', cptr, astype = _atom_or_none,
        read_only=True, doc =
        '''The 'chain trace' :class:`.Atom`\\ , if any.  
        Normally returns the C4' from a nucleic acid since that is always present,
        but in the case of a P-only trace it returns the P.''')
    ribbon_display = c_property('residue_ribbon_display', npy_bool,
        doc = "Whether to display the residue as a ribbon/pipe/plank. Boolean value.")
    ribbon_hide_backbone = c_property('residue_ribbon_hide_backbone', npy_bool,
        doc = "Whether a ribbon automatically hides the residue backbone atoms. Boolean value.")
    ribbon_color = c_property('residue_ribbon_color', uint8, 4,
        doc = "Ribbon color RGBA length 4 numpy uint8 array.")
    ribbon_adjust = c_property('residue_ribbon_adjust', float32,
        doc = "Smoothness adjustment factor (no adjustment = 0 <= factor <= 1 = idealized).")
    ss_id = c_property('residue_ss_id', int32,
        doc = "Secondary structure id number. Integer value.")
    ss_type = c_property('residue_ss_type', int32, doc=
        "Supported API. Secondary structure type of residue.  Integer value.  One of Residue.SS_COIL, Residue.SS_HELIX, Residue.SS_SHEET (a.k.a. SS_STRAND)")
    structure = c_property('residue_structure', pyobject, read_only = True,
        doc = "Supported API. ':class:`.AtomicStructure` that this residue belongs to. Read only.")

    def add_atom(self, atom):
        '''Supported API. Add the specified :class:`.Atom` to this residue.
        An atom can only belong to one residue, and all atoms
        must belong to a residue.'''
        f = c_function('residue_add_atom', args = (ctypes.c_void_p, ctypes.c_void_p))
        f(self._c_pointer, atom._c_pointer)

    def bonds_between(self, other_res):
        "Supported API. Return the bonds between this residue and other_res as a Bonds collection."
        f = c_function('residue_bonds_between', args = (ctypes.c_void_p, ctypes.c_void_p),
                ret = ctypes.py_object)
        return _bonds(f(self._c_pointer, other_res._c_pointer))

    def connects_to(self, other_res):
        "Supported API. Return True if this residue is connected by at least one bond "
        " (not pseudobond) to other_res"
        f = c_function('residue_connects_to', args = (ctypes.c_void_p, ctypes.c_void_p),
                ret = ctypes.c_bool)
        return f(self._c_pointer, other_res._c_pointer, ret = ctypes.c_bool)

    def find_atom(self, atom_name):
        '''Supported API. Return the atom with the given name, or None if not found.\n'''
        '''If multiple atoms in the residue have that name, an arbitrary one that matches will'''
        ''' be returned.'''
        f = c_function('residue_find_atom', args = (ctypes.c_void_p, ctypes.c_char_p),
            ret = ctypes.c_void_p)
        return _atom_or_none(f(self._c_pointer, atom_name.encode('utf-8')))

    @property
    def session(self):
        "Session that this Residue is in"
        return self.structure.session

    def set_alt_loc(self, loc):
        "Set the appropiate atoms in the residue to the given (existing) alt loc"
        if isinstance(loc, str):
            loc = loc.encode('utf-8')
        f = c_array_function('residue_set_alt_loc', args=(byte,), per_object=False)
        r_ref = ctypes.byref(self._c_pointer)
        f(r_ref, 1, loc)

    def string(self, residue_only = False, omit_structure = False, style = None):
        "Supported API.  Get text representation of Residue"
        if style == None:
            from chimerax.core.core_settings import settings
            style = settings.atomspec_contents
        ic = self.insertion_code
        if style.startswith("simple"):
            res_str = self.name + " " + str(self.number) + ic
        else:
            res_str = ":" + str(self.number) + ic
        chain_str = '/' + self.chain_id if not self.chain_id.isspace() else ""
        if residue_only:
            return res_str
        if omit_structure:
            return '%s %s' % (chain_str, res_str)
        from .structure import Structure
        if len([s for s in self.structure.session.models.list() if isinstance(s, Structure)]) > 1:
            struct_string = str(self.structure)
            if style.startswith("serial"):
                struct_string += " "
        else:
            struct_string = ""
        from chimerax.core.core_settings import settings
        if style.startswith("simple"):
            return '%s%s %s' % (struct_string, chain_str, res_str)
        if style.startswith("command"):
            return struct_string + chain_str + res_str
        return struct_string

    # used by custom-attr registration code
    @property
    def has_custom_attrs(self):
        return has_custom_attrs(Residue, self)

    def take_snapshot(self, session, flags):
        data = {'structure': self.structure,
                'ses_id': self.structure.session_residue_to_id(self._c_pointer),
                'custom attrs': get_custom_attrs(Residue, self)}
        return data

    @staticmethod
    def restore_snapshot(session, data):
        r = Residue.c_ptr_to_py_inst(data['structure'].session_id_to_residue(data['ses_id']))
        set_custom_attrs(r, data)
        return r


# -----------------------------------------------------------------------------
#
class Ring:
    '''
    A ring in the structure.
    '''

    def __init__(self, ring_pointer):
        set_c_pointer(self, ring_pointer)

    # cpp_pointer and deleted are "base class" methods, though for performance reasons
    # we are placing them directly in each class rather than using a base class,
    # and for readability by most programmers we avoid using metaclasses
    @property
    def cpp_pointer(self):
        '''Value that can be passed to C++ layer to be used as pointer (Python int)'''
        return self._c_pointer.value

    @property
    def deleted(self):
        '''Has the C++ side been deleted?'''
        return not hasattr(self, '_c_pointer')

    aromatic = c_property('ring_aromatic', npy_bool, read_only=True,
        doc="Whether the ring is aromatic. Boolean value.")
    atoms = c_property('ring_atoms', cptr, 'size', astype = _atoms, read_only = True,
        doc=":class:`.Atoms` collection containing the atoms of the ring, "
        "in no particular order (see :meth:`.Ring.ordered_atoms`).")
    bonds = c_property('ring_bonds', cptr, 'size', astype = _bonds, read_only = True,
        doc=":class:`.Bonds` collection containing the bonds of the ring, "
        "in no particular order (see :meth:`.Ring.ordered_bonds`).")
    ordered_atoms = c_property('ring_ordered_atoms', cptr, 'size', astype=_atoms, read_only=True,
        doc=":class:`.Atoms` collection containing the atoms of the ring, in ring order.")
    ordered_bonds = c_property('ring_ordered_bonds', cptr, 'size', astype=_bonds, read_only=True,
        doc=":class:`.Bonds` collection containing the bonds of the ring, in ring order.")
    size = c_property('ring_size', size_t, read_only=True,
        doc="Number of atoms (and bonds) in the ring. Read only.")

    def __eq__(self, r):
        if not isinstance(r, Ring):
            return False
        f = c_function('ring_equal', args=(ctypes.c_void_p, ctypes.c_void_p), ret=ctypes.c_bool)
        e = f(self._c_pointer, r._c_pointer)
        return e

    def __ge__(self, r):
        return self == r or not (self < r)

    def __gt__(self, r):
        return not (self < r or self == r)

    def __hash__(self):
        return id(self)

    def __le__(self, r):
        return self < r or self == r

    def __lt__(self, r):
        if not isinstance(r, Ring):
            return False
        f = c_function('ring_less_than', args=(ctypes.c_void_p, ctypes.c_void_p), ret=ctypes.c_bool)
        lt = f(self._c_pointer, r._c_pointer)
        return lt

    def __ne__(self, r):
        return not self == r


import atexit
# -----------------------------------------------------------------------------
#
class Sequence(State):
    '''
    A polymeric sequence.  Offers string-like interface.
    '''

    SS_HELIX = 'H'
    SS_OTHER = 'O'
    SS_STRAND = 'S'

    nucleic3to1 = lambda rn: c_function('sequence_nucleic3to1', args = (ctypes.c_char_p,),
        ret = ctypes.c_char)(rn.encode('utf-8')).decode('utf-8')
    protein3to1 = lambda rn: c_function('sequence_protein3to1', args = (ctypes.c_char_p,),
        ret = ctypes.c_char)(rn.encode('utf-8')).decode('utf-8')
    rname3to1 = lambda rn: c_function('sequence_rname3to1', args = (ctypes.c_char_p,),
        ret = ctypes.c_char)(rn.encode('utf-8')).decode('utf-8')

    # the following colors for use by alignment/sequence viewers
    default_helix_fill_color = (1.0, 1.0, 0.8)
    default_helix_outline_color = tuple([chan/255.0 for chan in (218, 165, 32)]) # goldenrod
    default_strand_fill_color = (0.88, 1.0, 1.0) # light cyan
    default_strand_outline_color = tuple([0.75*chan for chan in default_strand_fill_color])

    chimera_exiting = False

    def __init__(self, seq_pointer=None, *, name="sequence", characters=""):
        self.attrs = {} # miscellaneous attributes
        self.markups = {} # per-residue (strings or lists)
        self.numbering_start = None
        from chimerax.core.triggerset import TriggerSet
        self.triggers = TriggerSet()
        self.triggers.add_trigger('rename')
        f = c_function('set_sequence_py_instance', args = (ctypes.c_void_p, ctypes.py_object))
        if seq_pointer:
            set_c_pointer(self, seq_pointer)
            f(self._c_pointer, self)
            return # name/characters already exists; don't set
        seq_pointer = c_function('sequence_new',
            args = (ctypes.c_char_p, ctypes.c_char_p), ret = ctypes.c_void_p)(
                name.encode('utf-8'), characters.encode('utf-8'))
        set_c_pointer(self, seq_pointer)
        f(self._c_pointer, self)

    # cpp_pointer and deleted are "base class" methods, though for performance reasons
    # we are placing them directly in each class rather than using a base class,
    # and for readability by most programmers we avoid using metaclasses
    @property
    def cpp_pointer(self):
        '''Value that can be passed to C++ layer to be used as pointer (Python int)'''
        return self._c_pointer.value

    @property
    def deleted(self):
        '''Has the C++ side been deleted?'''
        return not hasattr(self, '_c_pointer')

    characters = c_property('sequence_characters', string, doc=
        "A string representing the contents of the sequence")
    circular = c_property('sequence_circular', npy_bool, doc="Indicates the sequence involves"
        " a circular permutation; the sequence characters have been doubled, and residue"
        " correspondences of the first half implicitly exist in the second half as well."
        " Typically used in alignments to line up with sequences that aren't permuted.")
    name = c_property('sequence_name', string, doc="The sequence name")

    # Some Sequence methods may have to be overridden/disallowed in Chain...

    def __copy__(self, copy_seq=None):
        if copy_seq is None:
            copy_seq = Sequence(name=self.name, characters=self.characters)
        else:
            copy_seq.characters = self.characters
        from copy import copy
        copy_seq.attrs = copy(self.attrs)
        copy_seq.markups = copy(self.markups)
        copy_seq.numbering_start = self.numbering_start
        return copy_seq

    def __del__(self):
        if Sequence.chimera_exiting:
            return
        # __del__ methods that create additional references (which the code in the
        # 'if' below apparently does) can cause __del__ to be called multiple times,
        # so the test below is necessary
        if not self.deleted:
            del_f = c_function('sequence_del_pyobj', args = (ctypes.c_void_p,))
            del_f(self._c_pointer) # will destroy C++ object unless it's an active Chain

    def extend(self, chars):
        """Extend the sequence with the given string"""
        f = c_function('sequence_extend', args = (ctypes.c_void_p, ctypes.c_char_p))
        f(self._c_pointer, chars.encode('utf-8'))
    append = extend

    @property
    def full_name(self):
        return self.name

    def gapped_to_ungapped(self, index):
        f = c_function('sequence_gapped_to_ungapped', args = (ctypes.c_void_p, ctypes.c_int),
            ret = ctypes.c_int)
        g2u = f(self._c_pointer, index)
        if g2u < 0:
            return None
        return g2u

    def __getitem__(self, key):
        return self.characters[key]

    # used by custom-attr registration code
    @property
    def has_custom_attrs(self):
        return has_custom_attrs(Sequence, self)

    def __hash__(self):
        return id(self)

    def __len__(self):
        """Sequence length"""
        f = c_function('sequence_len', args = (ctypes.c_void_p,), ret = ctypes.c_size_t)
        return f(self._c_pointer)

    @staticmethod
    def restore_snapshot(session, data):
        seq = Sequence()
        seq.set_state_from_snapshot(session, data)
        return seq

    def __setitem__(self, key, val):
        chars = self.characters
        if isinstance(key, slice):
            start = key.start if key.start is not None else 0
            stop = key.stop if key.stop is not None else len(chars)
            self.characters = chars[:start] + val + chars[stop:]
        else:
            self.characters = chars[:key] + val + chars[key+1:]

    # no __str__, since it's confusing whether it should be self.name or self.characters

    def set_state_from_snapshot(self, session, data):
        self.name = data['name']
        self.characters = data['characters']
        self.attrs = data.get('attrs', {})
        self.markups = data.get('markups', {})
        self.numbering_start = data.get('numbering_start', None)
        set_custom_attrs(self, data)

    def ss_type(self, loc, loc_is_ungapped=False):
        try:
            ss_markup = self.markups['SS']
        except KeyError:
            return None
        if not loc_is_ungapped:
            loc = self.gapped_to_ungapped(loc)
        if loc is None:
            return None
        ss = ss_markup[loc]
        if ss in "HGI":
            return self.SS_HELIX
        if ss == "E":
            return self.SS_STRAND
        return self.SS_OTHER

    def take_snapshot(self, session, flags):
        data = { 'name': self.name, 'characters': self.characters, 'attrs': self.attrs,
            'markups': self.markups, 'numbering_start': self.numbering_start,
            'custom attrs': get_custom_attrs(Sequence, self)}
        return data

    def ungapped(self):
        """String of sequence without gap characters"""
        f = c_function('sequence_ungapped', args = (ctypes.c_void_p,), ret = ctypes.py_object)
        return f(self._c_pointer)

    def ungapped_to_gapped(self, index):
        f = c_function('sequence_ungapped_to_gapped', args = (ctypes.c_void_p, ctypes.c_int),
            ret = ctypes.c_int)
        return f(self._c_pointer, index)

    def _cpp_rename(self, old_name):
        # called from C++ layer when 'name' attr changed
        self.triggers.activate_trigger('rename', (self, old_name))

    @atexit.register
    def _exiting():
        Sequence.chimera_exiting = True

# -----------------------------------------------------------------------------
#
class StructureSeq(Sequence):
    '''
    A sequence that has associated structure residues.

    Unlike the Chain subclass, StructureSeq will not change in size once created,
    though associated residues may change to None if those residues are deleted/closed.
    '''

    def __init__(self, sseq_pointer=None, *, chain_id=None, structure=None):
        if sseq_pointer is None:
            sseq_pointer = c_function('sseq_new',
                args = (ctypes.c_char_p, ctypes.c_void_p), ret = ctypes.c_void_p)(
                    chain_id.encode('utf-8'), structure._c_pointer)
        super().__init__(sseq_pointer)
        self.triggers.add_trigger('delete')
        self.triggers.add_trigger('modify')
        # description derived from PDB/mmCIF info and set by AtomicStructure constructor
        self.description = None

    def __lt__(self, other):
        # for sorting (objects of the same type)
        if self.structure != other.structure:
            return self.structure < other.structure
        if self.chain_id != other.chain_id:
            return self.chain_id < other.chain_id
        if self is other: # optimization to avoid comparing residue lists if possible
            return False
        return self.residues < other.residues

    chain_id = c_property('sseq_chain_id', string, read_only = True)
    '''Chain identifier. Limited to 4 characters. Read only string.'''
    # characters read-only in StructureSeq/Chain (use bulk_set)
    characters = c_property('sequence_characters', string, doc=
        "A string representing the contents of the sequence. Read only.")
    existing_residues = c_property('sseq_residues', cptr, 'num_residues', astype = _non_null_residues, read_only = True)
    ''':class:`.Residues` collection containing the residues of this sequence with existing structure, in order. Read only.'''
    from_seqres = c_property('sseq_from_seqres', npy_bool, doc = "Was the full sequence "
        " determined from SEQRES (or equivalent) records in the input file")
    num_existing_residues = c_property('sseq_num_existing_residues', size_t, read_only = True)
    '''Number of residues in this sequence with existing structure. Read only.'''
    num_residues = c_property('sseq_num_residues', size_t, read_only = True)
    '''Number of residues belonging to this sequence, including those without structure. Read only.'''
    polymer_type = c_property('sseq_polymer_type', uint8, read_only = True)
    '''Polymer type of this sequence. Same values as Residue.polymer_type, except should not return PT_NONE.'''
    residues = c_property('sseq_residues', cptr, 'num_residues', astype = _residues_or_nones,
        read_only = True, doc = "List containing the residues of this sequence in order. "
        "Residues with no structure will be None. Read only.")
    structure = c_property('sseq_structure', pyobject, read_only = True)
    ''':class:`.AtomicStructure` that this structure sequence comes from. Read only.'''

    # allow append/extend for now, since NeedlemanWunsch uses it

    def bulk_set(self, residues, characters):
        '''Set all residues/characters of StructureSeq. '''
        '''"characters" is a string or a list of characters.'''
        ptrs = [r._c_pointer.value if r else 0 for r in residues]
        if type(characters) == list:
            characters = "".join(characters)
        f = c_function('sseq_bulk_set', args = (ctypes.c_void_p, ctypes.py_object, ctypes.c_char_p))
        f(self._c_pointer, ptrs, characters.encode('utf-8'))

    def __copy__(self):
        f = c_function('sseq_copy', args = (ctypes.c_void_p,), ret = ctypes.c_void_p)
        copy_sseq = StructureSeq(f(self._c_pointer))
        Sequence.__copy__(self, copy_seq = copy_sseq)
        copy_sseq.description = self.description
        return copy_sseq

    @property
    def chain(self):
        try:
            return self.existing_residues[0].chain
        except IndexError:
            return None

    @property
    def full_name(self):
        rem = self.name
        for part in (self.structure.name, "(%s)" % self.structure):
            rem = rem.strip()
            if rem:
                rem = rem.strip()
                if rem.startswith(part):
                    rem = rem[len(part):]
                    continue
            break
        if rem and not rem.isspace():
            name_part = " " + rem.strip()
        else:
            name_part = ""
        return "%s (#%s)%s" % (self.structure.name, self.structure.id_string(), name_part)

    # used by custom-attr registration code
    @property
    def has_custom_attrs(self):
        return has_custom_attrs(Sequence, self) or has_custom_attrs(StructureSeq, self)

    def _get_numbering_start(self):
        if self._numbering_start == None:
            for i, r in enumerate(self.residues):
                if r is None:
                    continue
                if r.deleted:
                    return getattr(self, '_prev_numbering_start', 1)
                break
            else:
                return getattr(self, '_prev_numbering_start', 1)
            pns = self._prev_numbering_start = r.number - i
            return pns
        return self._numbering_start

    def _set_numbering_start(self, ns):
        self._numbering_start = ns

    numbering_start = property(_get_numbering_start, _set_numbering_start)

    @property
    def res_map(self):
        '''Returns a dict that maps from :class:`.Residue` to an ungapped sequence position'''
        f = c_function('sseq_res_map', args = (ctypes.c_void_p,), ret = ctypes.py_object)
        ptr_map = f(self._c_pointer)
        obj_map = {}
        for res_ptr, pos in ptr_map.items():
            res = _residue(res_ptr)
            obj_map[res] = pos
        return obj_map

    def residue_at(self, index):
        '''Return the Residue/None at the (ungapped) position 'index'.'''
        '''  More efficient that self.residues[index] since the entire residues'''
        ''' list isn't built/destroyed.'''
        f = c_function('sseq_residue_at', args = (ctypes.c_void_p, ctypes.c_size_t),
            ret = ctypes.c_void_p)
        return _residue_or_none(f(self._c_pointer, index))

    def residue_before(self, r):
        '''Return the residue at index one less than the given residue,
        or None if no such residue exists.'''
        pos = self.res_map[r]
        return None if pos == 0 else self.residue_at(pos-1)

    def residue_after(self, r):
        '''Return the residue at index one more than the given residue,
        or None if no such residue exists.'''
        pos = self.res_map[r]
        return None if pos+1 >= len(self) else self.residue_at(pos+1)

    @staticmethod
    def restore_snapshot(session, data):
        sseq = StructureSeq(chain_id=data['chain_id'], structure=data['structure'])
        Sequence.set_state_from_snapshot(sseq, session, data['Sequence'])
        sseq.description = data['description']
        sseq.bulk_set(data['residues'], sseq.characters)
        sseq.description = data.get('description', None)
        set_custom_attrs(sseq, data)
        return sseq

    @property
    def session(self):
        "Session that this StructureSeq is in"
        return self.structure.session

    def ss_type(self, loc, loc_is_ungapped=False):
        if not loc_is_ungapped:
            loc = self.gapped_to_ungapped(loc)
        if loc is None:
            return None
        r = self.residue_at(loc)
        if r is None:
            return None
        if r.is_helix:
            return self.SS_HELIX
        if r.is_strand:
            return self.SS_STRAND
        return self.SS_OTHER

    def take_snapshot(self, session, flags):
        data = {
            'Sequence': Sequence.take_snapshot(self, session, flags),
            'chain_id': self.chain_id,
            'description': self.description,
            'residues': self.residues,
            'structure': self.structure,
            'custom attrs': get_custom_attrs(StructureSeq, self)
        }
        return data

    def _cpp_demotion(self):
        # called from C++ layer when this should be demoted to Sequence
        numbering_start = self.numbering_start
        self.__class__ = Sequence
        self.triggers.activate_trigger('delete', self)
        self.numbering_start = numbering_start

# sequence-structure association functions that work on StructureSeqs...

def estimate_assoc_params(sseq):
    '''Estimate the parameters needed to associate a sequence with a Chain/StructureSeq

       Returns a 3-tuple:
           * Estimated total ungapped length, accounting for missing structure

           * A list of continuous sequence segments

           * A list of the estimated size of the gaps between those segments
    '''
    f = c_function('sseq_estimate_assoc_params', args = (ctypes.c_void_p,), ret = ctypes.py_object)
    return f(sseq._c_pointer)

class StructAssocError(ValueError):
    pass

def try_assoc(session, seq, sseq, assoc_params, *, max_errors = 6):
    '''Try to associate StructureSeq 'sseq' with Sequence 'seq'.

       A set of association parameters ('assoc_params') must be provided, typically obtained
       from the :py:func:`estimate_assoc_params` function.  See that function's documentation
       for details of assoc_param's contents.  The maximum number of errors allowed can
       optionally be specified (default: 6).

       The return value is a 2-tuple, consisting of a :py:class:`SeqMatchMap` instance describing
       the association, and the number of errors encountered.

       An unsuccessful association throws StructAssocError.
    '''
    f = c_function('sseq_try_assoc', args = (ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t,
        ctypes.py_object, ctypes.py_object, ctypes.c_int), ret = ctypes.py_object)
    est_len, segments, gaps = assoc_params
    try:
        res_to_pos, errors = f(seq._c_pointer, sseq._c_pointer, est_len, segments, gaps, max_errors)
    except ValueError as e:
        if str(e) == "bad assoc":
            raise StructAssocError(str(e))
        else:
            raise
    mmap = SeqMatchMap(session, seq, sseq)
    for r, i in res_to_pos.items():
        mmap.match(_residue(r), i)
    return mmap, errors

# -----------------------------------------------------------------------------
#
class Chain(StructureSeq):
    '''
    A single polymer chain such as a protein, DNA or RNA strand.
    A chain has a sequence associated with it.  A chain may have breaks.
    Chain objects are not always equivalent to Protein Databank chains.

    '''

    def __str__(self):
        return self.string()

    def atomspec(self):
        chain_str = '/' + self.chain_id if not self.chain_id.isspace() else ""
        return self.structure.atomspec() + chain_str

    def extend(self, chars):
        # disallow extend
        raise AssertionError("extend() called on Chain object")

    @staticmethod
    def restore_snapshot(session, data):
        ptr = data['structure'].session_id_to_chain(data['ses_id'])
        chain = Chain.c_ptr_to_existing_py_inst(ptr)
        if not chain:
            chain = Chain(ptr)
        chain.description = data.get('description', None)
        set_custom_attrs(chain, data)
        return chain

    def string(self):
        from chimerax.core.core_settings import settings
        cmd_style = settings.atomspec_contents == "command-line specifier"
        chain_str = '/' + self.chain_id if not self.chain_id.isspace() else ""
        from .structure import Structure
        if len([s for s in self.structure.session.models.list() if isinstance(s, Structure)]) > 1:
            struct_string = str(self.structure)
        else:
            struct_string = ""
        from chimerax.core.core_settings import settings
        return struct_string + chain_str

    def take_snapshot(self, session, flags):
        data = {
            'description': self.description,
            'ses_id': self.structure.session_chain_to_id(self._c_pointer),
            'structure': self.structure,
            'custom attrs': get_custom_attrs(StructureSeq, self)
        }
        return data

# -----------------------------------------------------------------------------
#
class StructureData:
    '''
    This is a base class of both :class:`.AtomicStructure` and :class:`.Structure`.
    This base class manages the data while the
    derived class handles the graphical 3-dimensional rendering using OpenGL.
    '''

    PBG_METAL_COORDINATION = c_function('structure_PBG_METAL_COORDINATION', args = (),
        ret = ctypes.c_char_p)().decode('utf-8')
    PBG_MISSING_STRUCTURE = c_function('structure_PBG_MISSING_STRUCTURE', args = (),
        ret = ctypes.c_char_p)().decode('utf-8')
    PBG_HYDROGEN_BONDS = c_function('structure_PBG_HYDROGEN_BONDS', args = (),
        ret = ctypes.c_char_p)().decode('utf-8')

    def __init__(self, mol_pointer=None, *, logger=None):
        if mol_pointer is None:
            # Create a new graph
            from .structure import AtomicStructure
            new_func = 'atomic_structure_new' if isinstance(self, AtomicStructure) else 'structure_new'
            mol_pointer = c_function(new_func, args = (ctypes.py_object,), ret = ctypes.c_void_p)(logger)
        set_c_pointer(self, mol_pointer)
        f = c_function('set_structure_py_instance', args = (ctypes.c_void_p, ctypes.py_object))
        f(self._c_pointer, self)
        self._ses_end_handler = None

    # cpp_pointer and deleted are "base class" methods, though for performance reasons
    # we are placing them directly in each class rather than using a base class,
    # and for readability by most programmers we avoid using metaclasses
    @property
    def cpp_pointer(self):
        '''Value that can be passed to C++ layer to be used as pointer (Python int)'''
        return self._c_pointer.value

    @property
    def deleted(self):
        '''Has the C++ side been deleted?'''
        return not hasattr(self, '_c_pointer')

    def delete(self):
        '''Deletes the C++ data for this atomic structure.'''
        if self._ses_end_handler:
            self.session.triggers.remove_handler(self._ses_end_handler)
        c_function('structure_delete', args = (ctypes.c_void_p,))(self._c_pointer)

    active_coordset_change_notify = c_property('structure_active_coordset_change_notify', npy_bool,
    doc='''Whether notifications are issued when the active coordset is changed.  Should only be
    set to true when temporarily changing the active coordset in a Python script. Boolean''')
    active_coordset = c_property('structure_active_coordset', cptr, astype = _coordset,
        read_only = True, doc="Supported API. Currently active :class:`CoordSet`.")
    active_coordset_id = c_property('structure_active_coordset_id', int32)
    '''Index of the active coordinate set.'''
    alt_loc_change_notify = c_property('structure_alt_loc_change_notify', npy_bool, doc=
    '''Whether notifications are issued when altlocs are changed.  Should only be
    set to true when temporarily changing alt locs in a Python script. Boolean''')
    atoms = c_property('structure_atoms', cptr, 'num_atoms', astype = _atoms, read_only = True)
    ''':class:`.Atoms` collection containing all atoms of the structure.'''
    ball_scale = c_property('structure_ball_scale', float32,
        doc = "Scales sphere radius in ball-and-stick style.")
    bonds = c_property('structure_bonds', cptr, 'num_bonds', astype = _bonds, read_only = True)
    ''':class:`.Bonds` collection containing all bonds of the structure.'''
    chains = c_property('structure_chains', cptr, 'num_chains', astype = _chains, read_only = True)
    ''':class:`.Chains` collection containing all chains of the structure.'''
    coordset_ids = c_property('structure_coordset_ids', int32, 'num_coordsets', read_only = True)
    '''Return array of ids of all coordinate sets.'''
    coordset_size = c_property('structure_coordset_size', int32, read_only = True)
    '''Return the size of the active coordinate set array.'''
    lower_case_chains = c_property('structure_lower_case_chains', npy_bool)
    '''Structure has lower case chain ids. Boolean'''
    num_atoms = c_property('structure_num_atoms', size_t, read_only = True)
    '''Number of atoms in structure. Read only.'''
    num_atoms_visible = c_property('structure_num_atoms_visible', size_t, read_only = True)
    '''Number of visible atoms in structure. Read only.'''
    num_bonds = c_property('structure_num_bonds', size_t, read_only = True)
    '''Number of bonds in structure. Read only.'''
    num_bonds_visible = c_property('structure_num_bonds_visible', size_t, read_only = True)
    '''Number of visible bonds in structure. Read only.'''
    num_coordsets = c_property('structure_num_coordsets', size_t, read_only = True)
    '''Number of coordinate sets in structure. Read only.'''
    num_chains = c_property('structure_num_chains', size_t, read_only = True)
    '''Number of chains structure. Read only.'''
    num_residues = c_property('structure_num_residues', size_t, read_only = True)
    '''Number of residues structure. Read only.'''
    residues = c_property('structure_residues', cptr, 'num_residues', astype = _residues, read_only = True)
    ''':class:`.Residues` collection containing the residues of this structure. Read only.'''
    pbg_map = c_property('structure_pbg_map', pyobject, astype = _pseudobond_group_map, read_only = True)
    '''Dictionary mapping name to :class:`.PseudobondGroup` for pseudobond groups
    belonging to this structure. Read only.'''
    metadata = c_property('metadata', pyobject, read_only = True)
    '''Dictionary with metadata. Read only.'''
    pdb_version = c_property('pdb_version', int32)
    '''Dictionary with metadata. Read only.'''
    ribbon_tether_scale = c_property('structure_ribbon_tether_scale', float32)
    '''Ribbon tether thickness scale factor (1.0 = match displayed atom radius, 0=invisible).'''
    ribbon_tether_shape = c_property('structure_ribbon_tether_shape', int32)
    '''Ribbon tether shape. Integer value.'''
    TETHER_CONE = 0
    '''Tether is cone with point at ribbon.'''
    TETHER_REVERSE_CONE = 1
    '''Tether is cone with point at atom.'''
    TETHER_CYLINDER = 2
    '''Tether is cylinder.'''
    ribbon_show_spine = c_property('structure_ribbon_show_spine', npy_bool)
    '''Display ribbon spine. Boolean.'''
    ribbon_orientation = c_property('structure_ribbon_orientation', int32)
    '''Ribbon orientation. Integer value.'''
    RIBBON_ORIENT_GUIDES = 1
    '''Ribbon orientation from guide atoms.'''
    RIBBON_ORIENT_ATOMS = 2
    '''Ribbon orientation from interpolated atoms.'''
    RIBBON_ORIENT_CURVATURE = 3
    '''Ribbon orientation perpendicular to ribbon curvature.'''
    RIBBON_ORIENT_PEPTIDE = 4
    '''Ribbon orientation perpendicular to peptide planes.'''
    ribbon_display_count = c_property('structure_ribbon_display_count', int32, read_only = True)
    '''Return number of residues with ribbon display set. Integer.'''
    ribbon_tether_sides = c_property('structure_ribbon_tether_sides', int32)
    '''Number of sides for ribbon tether. Integer value.'''
    ribbon_tether_opacity = c_property('structure_ribbon_tether_opacity', float32)
    '''Ribbon tether opacity scale factor (relative to the atom).'''
    ribbon_mode_helix = c_property('structure_ribbon_mode_helix', int32)
    '''Ribbon mode for helices. Integer value.'''
    ribbon_mode_strand = c_property('structure_ribbon_mode_strand', int32)
    '''Ribbon mode for strands. Integer value.'''
    RIBBON_MODE_DEFAULT = 0
    '''Default ribbon mode showing secondary structure with ribbons.'''
    RIBBON_MODE_ARC = 1
    '''Ribbon mode showing secondary structure as an arc (tube or plank).'''
    RIBBON_MODE_WRAP = 2
    '''Ribbon mode showing helix as ribbon wrapped around tube.'''

    def ribbon_orients(self, residues=None):
        '''Return array of orientation values for given residues.'''
        if residues is None:
            residues = self.residues
        f = c_function('structure_ribbon_orient', args = (ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t), ret = ctypes.py_object)
        return f(self._c_pointer, residues._c_pointers, len(residues))

    ss_assigned = c_property('structure_ss_assigned', npy_bool, doc =
        "Has secondary structure been assigned, either by data in original structure file "
        "or by some algorithm (e.g. dssp command)")

    def _copy(self):
        f = c_function('structure_copy', args = (ctypes.c_void_p,), ret = ctypes.c_void_p)
        p = f(self._c_pointer)
        return p

    def add_coordset(self, id, xyz):
        '''Add a coordinate set with the given id.'''
        if xyz.dtype != float64:
            raise ValueError('add_coordset(): array must be float64, got %s' % xyz.dtype.name)
        f = c_function('structure_add_coordset',
                       args = (ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t))
        f(self._c_pointer, id, pointer(xyz), len(xyz))

    def add_coordsets(self, xyzs, replace = True):
        '''Add coordinate sets.  If 'replace' is True, clear out existing coordinate sets first'''
        if len(xyzs.shape) != 3:
            raise ValueError('add_coordsets(): array must be (frames)x(atoms)x3-dimensional')
        if xyzs.shape[1] != self.num_atoms:
            raise ValueError('add_coordsets(): second dimension of coordinate array'
                ' must be same as number of atoms')
        if xyzs.shape[2] != 3:
            raise ValueError('add_coordsets(): third dimension of coordinate array'
                ' must be 3 (xyz)')
        if xyzs.dtype != float64:
            raise ValueError('add_coordsets(): array must be float64, got %s' % xyzs.dtype.name)
        f = c_function('structure_add_coordsets',
                       args = (ctypes.c_void_p, ctypes.c_bool, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t))
        f(self._c_pointer, replace, pointer(xyzs), *xyzs.shape[:2])

    def coordset(self, cs_id):
        '''Return the CoordSet for the given coordset ID'''
        f = c_function('structure_py_obj_coordset', args = (ctypes.c_void_p, ctypes.c_int),
            ret = ctypes.py_object)
        return f(self._c_pointer, cs_id)

    def connect_structure(self, chain_starters, chain_enders, conect_atoms, mod_res):
        '''Generate connectivity.  See connect_structure in connectivity.rst for more details.
        
        chain_starters and chain_enders are lists of residues.
        conect_atoms is a list of atoms.
        mod_res is a list of residues (not MolResId's).'''
        f = c_function('structure_connect',
                       args = (ctypes.c_void_p, ctypes.py_object, ctypes.py_object, ctypes.py_object, ctypes.py_object),
                       ret = ctypes.c_int)
        starters = list([r._c_pointer.value for r in chain_starters])
        enders = list([r._c_pointer.value for r in chain_enders])
        conect = list([a._c_pointer.value for a in conect_atoms])
        mod = list([r._c_pointer.value for r in mod_res])
        return f(self._c_pointer, starters, enders, conect, mod)

    def delete_alt_locs(self):
        '''Incorporate current alt locs as "regular" atoms and remove other alt locs'''
        f = c_function('structure_delete_alt_locs', args = (ctypes.c_void_p,))(self._c_pointer)

    def delete_atom(self, atom):
        '''Delete the specified Atom.'''
        f = c_function('structure_delete_atom', args = (ctypes.c_void_p, ctypes.c_void_p))
        f(self._c_pointer, atom._c_pointer)

    @property
    def molecules(self):
        '''Return a tuple of :class:`.Atoms` objects each containing atoms for one molecule.
           Missing-structure pseudobonds are consider to connect parts of a molecule.
        '''
        f = c_function('structure_molecules', args = (ctypes.c_void_p,), ret = ctypes.py_object)
        atom_arrays = f(self._c_pointer)
        from .molarray import Atoms
        return tuple(Atoms(aa) for aa in atom_arrays)

    def new_atom(self, atom_name, element):
        '''Create a new :class:`.Atom` object. It must be added to a :class:`.Residue` object
        belonging to this structure before being used.  'element' can be a string (atomic symbol),
        an integer (atomic number), or an Element instance'''
        if not isinstance(element, Element):
            element = Element.get_element(element)
        f = c_function('structure_new_atom',
                       args = (ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p),
                       ret = ctypes.py_object)
        return f(self._c_pointer, atom_name.encode('utf-8'), element._c_pointer)

    def new_bond(self, atom1, atom2):
        '''Create a new :class:`.Bond` joining two :class:`Atom` objects.'''
        f = c_function('structure_new_bond',
                       args = (ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p),
                       ret = ctypes.py_object)
        return f(self._c_pointer, atom1._c_pointer, atom2._c_pointer)

    def new_coordset(self, index=None, size=None):
        '''Create a new empty coordset.  In almost all circumstances one would use the
           add_coordset(s) method instead (to add fully populated coordsets), but in some
           cases when building a Structure from scratch this method is needed.

           'index' defaults to one more than highest existing index (or 1 if none existing);
           'size' is for efficiency when creating the first coordinate set of a new Structure,
           and is otherwise unnecessary to specify
        '''
        if index is None:
            f = c_function('structure_new_coordset_default', args = (ctypes.c_void_p,))
            f()
        else:
            if size is None:
                f = c_function('structure_new_coordset_index',
                    args = (ctypes.c_void_p, ctypes.c_int))
                f(index)
            else:
                f = c_function('structure_new_coordset_index_size',
                    args = (ctypes.c_void_p, ctypes.c_int, ctypes.c_int))
                f(index, size)

    def new_residue(self, residue_name, chain_id, pos, insert=' '):
        '''Create a new :class:`.Residue`.'''
        f = c_function('structure_new_residue',
                       args = (ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int, ctypes.c_char),
                       ret = ctypes.py_object)
        return f(self._c_pointer, residue_name.encode('utf-8'), chain_id.encode('utf-8'), pos, insert.encode('utf-8'))

    PMS_ALWAYS_CONNECTS, PMS_NEVER_CONNECTS, PMS_TRACE_CONNECTS = range(3)
    def polymers(self, missing_structure_treatment = PMS_ALWAYS_CONNECTS,
            consider_chains_ids = True):
        '''Return a list of (:class:`.Residues`, Residue.polymer_type) tuples, one tuple
        per polymer.
        'missing_structure_treatment' controls whether a single polymer can span any missing
        structure, no missing structure, or only missing structure that is part of a chain trace.
        'consider_chain_ids', if true, will break polymers when chain IDs change, regardless of
        other considerations.'''
        f = c_function('structure_polymers',
                       args = (ctypes.c_void_p, ctypes.c_int, ctypes.c_int),
                       ret = ctypes.py_object)
        polymers = f(self._c_pointer, missing_structure_treatment, consider_chains_ids)
        from .molarray import Residues
        return [(Residues(res_array), ptype) for res_array, ptype in polymers]

    def pseudobond_group(self, name, *, create_type = "normal"):
        '''Get or create a :class:`.PseudobondGroup` belonging to this structure.'''
        if isinstance(create_type, int):
            create_arg = create_type
        elif create_type is None:
            create_arg = 0
        elif create_type == "normal":
            create_arg = 1
        else:  # per-coordset
            create_arg = 2
        f = c_function('structure_pseudobond_group',
                       args = (ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int),
                       ret = ctypes.py_object)
        return f(self._c_pointer, name.encode('utf-8'), create_arg)

    def _delete_pseudobond_group(self, pbg):
        f = c_function('structure_delete_pseudobond_group',
                       args = (ctypes.c_void_p, ctypes.c_void_p), ret = None)
        f(self._c_pointer, pbg._c_pointer)

    def reorder_residues(self, new_order):
        '''Reorder the residues.  Obviously, 'new_order' has to have exactly the same
           residues as the structure currently has.
        '''
        f = c_function('structure_reorder_residues', args = (ctypes.c_void_p, ctypes.py_object))
        f(self._c_pointer, [r._c_pointer for r in new_order])

    @classmethod
    def restore_snapshot(cls, session, data):
        g = StructureData(logger=session.logger)
        g.set_state_from_snapshot(session, data)
        return g

    def rings(self, cross_residues=False, all_size_threshold=0):
        '''Return :class:`.Rings` collection of rings found in this Structure.

        If 'cross_residues' is False, then rings that cross residue boundaries are not
        included.  If 'all_size_threshold' is zero, then return only minimal rings, of
        any size.  If it is greater than zero, then return all rings not larger than the
        given value.

        The returned rings are quite emphemeral, and shouldn't be cached or otherwise
        retained for long term use.  They may only live until the next call to rings()
        [from anywhere, including C++].
        '''
        f = c_function('structure_rings', args = (ctypes.c_void_p, ctypes.c_bool, ctypes.c_int),
                ret = ctypes.py_object)
        return _rings(f(self._c_pointer, cross_residues, all_size_threshold))

    def set_state_from_snapshot(self, session, data):
        '''Restore from session info'''
        self._ses_call("restore_setup")
        f = c_function('structure_session_restore',
                args = (ctypes.c_void_p, ctypes.c_int,
                        ctypes.py_object, ctypes.py_object, ctypes.py_object))
        f(self._c_pointer, data['version'], tuple(data['ints']), tuple(data['floats']), tuple(data['misc']))
        self._ses_end_handler = session.triggers.add_handler("end restore session",
            self._ses_restore_teardown)

    def save_state(self, session, flags):
        '''Gather session info; return version number'''
        # the save setup/teardown handled in Structure/AtomicStructure class, so that
        # the trigger handlers can be deregistered when the object is deleted
        f = c_function('structure_session_info',
                    args = (ctypes.c_void_p, ctypes.py_object, ctypes.py_object,
                        ctypes.py_object),
                    ret = ctypes.c_int)
        data = {'ints': [],
                'floats': [],
                'misc': []}
        data['version'] = f(self._c_pointer, data['ints'], data['floats'], data['misc'])
        # data is all simple Python primitives, let session saving know that...
        from chimerax.core.state import FinalizedState
        return FinalizedState(data)

    def session_atom_to_id(self, ptr):
        '''Map Atom pointer to session ID'''
        f = c_function('structure_session_atom_to_id',
                    args = (ctypes.c_void_p, ctypes.c_void_p), ret = size_t)
        return f(self._c_pointer, ptr)

    def session_bond_to_id(self, ptr):
        '''Map Bond pointer to session ID'''
        f = c_function('structure_session_bond_to_id',
                    args = (ctypes.c_void_p, ctypes.c_void_p), ret = size_t)
        return f(self._c_pointer, ptr)

    def session_chain_to_id(self, ptr):
        '''Map Chain pointer to session ID'''
        f = c_function('structure_session_chain_to_id',
                    args = (ctypes.c_void_p, ctypes.c_void_p), ret = size_t)
        return f(self._c_pointer, ptr)

    def session_residue_to_id(self, ptr):
        '''Map Residue pointer to session ID'''
        f = c_function('structure_session_residue_to_id',
                    args = (ctypes.c_void_p, ctypes.c_void_p), ret = size_t)
        return f(self._c_pointer, ptr)

    def session_id_to_atom(self, i):
        '''Map sessionID to Atom pointer'''
        f = c_function('structure_session_id_to_atom',
                    args = (ctypes.c_void_p, ctypes.c_size_t), ret = ctypes.c_void_p)
        return f(self._c_pointer, i)

    def session_id_to_bond(self, i):
        '''Map sessionID to Bond pointer'''
        f = c_function('structure_session_id_to_bond',
                    args = (ctypes.c_void_p, ctypes.c_size_t), ret = ctypes.c_void_p)
        return f(self._c_pointer, i)

    def session_id_to_chain(self, i):
        '''Map sessionID to Chain pointer'''
        f = c_function('structure_session_id_to_chain',
                    args = (ctypes.c_void_p, ctypes.c_size_t), ret = ctypes.c_void_p)
        return f(self._c_pointer, i)

    def session_id_to_residue(self, i):
        '''Map sessionID to Residue pointer'''
        f = c_function('structure_session_id_to_residue',
                    args = (ctypes.c_void_p, ctypes.c_size_t), ret = ctypes.c_void_p)
        return f(self._c_pointer, i)

    def set_color(self, rgba):
        '''Set color of atoms, bonds, and residues'''
        f = c_function('set_structure_color',
                    args = (ctypes.c_void_p, ctypes.c_void_p))
        return f(self._c_pointer, pointer(rgba))

    def use_default_atom_radii(self):
        '''If some atoms' radii has previously been explicitly set, this call will
        revert to using the default radii'''
        f = c_function('structure_use_default_atom_radii', args = (ctypes.c_void_p,))
        f(self._c_pointer)

    def _cpp_notify_position(self, pos):
        f = c_function('structure_set_position', args = (ctypes.c_void_p, ctypes.c_void_p))
        f(self._c_pointer, pointer(pos.matrix))

    def _ses_call(self, func_qual):
        f = c_function('structure_session_' + func_qual, args=(ctypes.c_void_p,))
        f(self._c_pointer)

    def _ses_restore_teardown(self, *args):
        self._ses_call("restore_teardown")
        self._ses_end_handler = None
        from chimerax.core.triggerset import DEREGISTER
        return DEREGISTER

    def _start_change_tracking(self, change_tracker):
        f = c_function('structure_start_change_tracking',
                args = (ctypes.c_void_p, ctypes.c_void_p))
        f(self._c_pointer, change_tracker._c_pointer)

    # Graphics changed flags used by rendering code.  Private.
    _SHAPE_CHANGE = 0x1
    _COLOR_CHANGE = 0x2
    _SELECT_CHANGE = 0x4
    _RIBBON_CHANGE = 0x8
    _ADDDEL_CHANGE = 0x10
    _DISPLAY_CHANGE = 0x20
    _ALL_CHANGE = 0x2f
    _graphics_changed = c_property('structure_graphics_change', int32)

# -----------------------------------------------------------------------------
#
class CoordSet(State):
    '''
    The coordinates for one frame of a Structure

    To create a Bond use the :class:`.AtomicStructure` new_coordset() method.
    '''
    def __init__(self, cs_pointer):
        set_c_pointer(self, cs_pointer)

    # cpp_pointer and deleted are "base class" methods, though for performance reasons
    # we are placing them directly in each class rather than using a base class,
    # and for readability by most programmers we avoid using metaclasses
    @property
    def cpp_pointer(self):
        '''Value that can be passed to C++ layer to be used as pointer (Python int)'''
        return self._c_pointer.value

    @property
    def deleted(self):
        '''Has the C++ side been deleted?'''
        return not hasattr(self, '_c_pointer')

    id = c_property('coordset_id', int32, read_only = True, doc="ID number of coordset")
    structure = c_property('coordset_structure', pyobject, read_only=True,
        doc=":class:`.AtomicStructure` the coordset belongs to")

    @property
    def session(self):
        "Session that this CoordSet is in"
        return self.structure.session

    # used by custom-attr registration code
    @property
    def has_custom_attrs(self):
        return has_custom_attrs(CoordSet, self)

    def take_snapshot(self, session, flags):
        data = {'structure': self.structure, 'cs_id': self.id,
                'custom attrs': get_custom_attrs(CoordSet, self)}
        return data

    @staticmethod
    def restore_snapshot(session, data):
        cs = data['structure'].coordset(data['cs_id'])
        set_custom_attrs(cs, data)
        return cs

# -----------------------------------------------------------------------------
#
class ChangeTracker:
    '''Per-session singleton change tracker keeps track of all
    atomic data changes'''

    def __init__(self):
        f = c_function('change_tracker_create', args = (), ret = ctypes.c_void_p)
        set_c_pointer(self, f())

    # cpp_pointer and deleted are "base class" methods, though for performance reasons
    # we are placing them directly in each class rather than using a base class,
    # and for readability by most programmers we avoid using metaclasses
    @property
    def cpp_pointer(self):
        '''Value that can be passed to C++ layer to be used as pointer (Python int)'''
        return self._c_pointer.value

    @property
    def deleted(self):
        '''Has the C++ side been deleted?'''
        return not hasattr(self, '_c_pointer')

    def add_modified(self, modded, reason):
        f = c_function('change_tracker_add_modified',
            args = (ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_char_p))
        from .molarray import Collection
        if isinstance(modded, Collection):
            class_num = self._class_to_int(modded.object_class)
            for ptr in modded.pointers:
                f(self._c_pointer, class_num, ptr, reason.encode('utf-8'))
        else:
            f(self._c_pointer, self._inst_to_int(modded), modded._c_pointer,
                reason.encode('utf-8'))
    @property
    def changed(self):
        f = c_function('change_tracker_changed', args = (ctypes.c_void_p,), ret = ctypes.c_bool)
        return f(self._c_pointer)

    @property
    def changes(self):
        f = c_function('change_tracker_changes', args = (ctypes.c_void_p,),
            ret = ctypes.py_object)
        global_data, per_structure_data = f(self._c_pointer)
        class Changes:
            def __init__(self, created, modified, reasons, total_deleted):
                self.created = created
                self.modified = modified
                self.reasons = reasons
                self.total_deleted = total_deleted
        def process_changes(data):
            final_changes = {}
            for k, v in data.items():
                created_ptrs, mod_ptrs, reasons, tot_del = v
                temp_ns = {}
                # can't effectively use locals() as the third argument as per the
                # Python 3 documentation for exec() and locals()
                exec("from .molarray import {}s as collection".format(k), globals(), temp_ns)
                collection = temp_ns['collection']
                fc_key = k[:-4] if k.endswith("Data") else k
                final_changes[fc_key] = Changes(collection(created_ptrs),
                    collection(mod_ptrs), reasons, tot_del)
            return final_changes
        global_changes = process_changes(global_data)
        per_structure_changes = {}
        for s_ptr, structure_data in per_structure_data.items():
            per_structure_changes[_atomic_structure(s_ptr)] = process_changes(structure_data)
        return global_changes, per_structure_changes

    def clear(self):
        f = c_function('change_tracker_clear', args = (ctypes.c_void_p,))
        f(self._c_pointer)

    def _class_to_int(self, klass):
        # has to tightly coordinate wih change_track_add_modified
        #
        # used with Collections, so can use exact equality test
        if klass.__name__ == "Atom":
            return 0
        if klass.__name__ == "Bond":
            return 1
        if klass.__name__ == "Pseudobond":
            return 2
        if klass.__name__ == "Residue":
            return 3
        if klass.__name__ == "Chain":
            return 4
        if klass.__name__ == "Structure":
            return 5
        if klass.__name__ == "AtomicStructure":
            return 5
        if klass.__name__ == "PseudobondGroup":
            return 6
        if klass.__name__ == "CoordSet":
            return 7
        raise AssertionError("Unknown class for change tracking: %s" % klass.__name__)

    def _inst_to_int(self, inst):
        # has to tightly coordinate wih change_track_add_modified
        #
        # used with instances, so may be a derived subclass
        if isinstance(inst, Atom):
            return 0
        if isinstance(inst, Bond):
            return 1
        if isinstance(inst, Pseudobond):
            return 2
        if isinstance(inst, Residue):
            return 3
        if isinstance(inst, Chain):
            return 4
        if isinstance(inst, StructureData):
            return 5
        if isinstance(inst, PseudobondGroupData):
            return 6
        if isinstance(inst, CoordSet):
            return 7
        raise AssertionError("Unknown class for change tracking: %s" % inst.__class__.__name__)

from .cymol import Element
"""
# -----------------------------------------------------------------------------
#
class Element:
    '''A chemical element having a name, number, mass, and other physical properties.'''

    NUM_SUPPORTED_ELEMENTS = c_function('element_NUM_SUPPORTED_ELEMENTS', args = (),
        ret = size_t)()

    def __init__(self, element_pointer):
        if not isinstance(element_pointer, int) or element_pointer < 256:
            raise ValueError("Do not use Element constructor directly;"
                " use Element.get_element method to get an Element instance")
        set_c_pointer(self, element_pointer)

    # cpp_pointer and deleted are "base class" methods, though for performance reasons
    # we are placing them directly in each class rather than using a base class,
    # and for readability by most programmers we avoid using metaclasses
    @property
    def cpp_pointer(self):
        '''Value that can be passed to C++ layer to be used as pointer (Python int)'''
        return self._c_pointer.value

    @property
    def deleted(self):
        '''Has the C++ side been deleted?'''
        return not hasattr(self, '_c_pointer')

    name = c_property('element_name', string, read_only = True)
    '''Element name, for example C for carbon. Read only.'''
    names = c_function('element_names', ret = ctypes.py_object)()
    '''Set of known element names'''
    number = c_property('element_number', uint8, read_only = True)
    '''Element atomic number, for example 6 for carbon. Read only.'''
    mass = c_property('element_mass', float32, read_only = True)
    '''Element atomic mass,
    taken from http://en.wikipedia.org/wiki/List_of_elements_by_atomic_weight.
    Read only.'''
    is_alkali_metal = c_property('element_is_alkali_metal', npy_bool, read_only = True)
    '''Is atom an alkali metal. Read only.'''
    is_halogen = c_property('element_is_halogen', npy_bool, read_only = True)
    '''Is atom a halogen. Read only.'''
    is_metal = c_property('element_is_metal', npy_bool, read_only = True)
    '''Is atom a metal. Read only.'''
    is_noble_gas = c_property('element_is_noble_gas', npy_bool, read_only = True)
    '''Is atom a noble_gas. Read only.'''
    valence = c_property('element_valence', uint8, read_only = True)
    '''Element valence number, for example 7 for chlorine. Read only.'''

    def __str__(self):
        return self.name

    @staticmethod
    def bond_length(e1, e2):
        '''Standard single-bond length between two elements

        Arguments can be element instances, atomic numbers, or element names'''
        if not isinstance(e1, Element):
            e1 = Element.get_element(e1)
        if not isinstance(e2, Element):
            e2 = Element.get_element(e2)
        f = c_function('element_bond_length', args = (ctypes.c_void_p, ctypes.c_void_p),
            ret = ctypes.c_float)
        return f(e1._c_pointer, e2._c_pointer)

    @staticmethod
    def bond_radius(e):
        '''Standard single-bond 'radius'
        (the amount this element would contribute to bond length)

        Argument can be an element instance, atomic number, or element name'''
        if not isinstance(e, Element):
            e = Element.get_element(e)
        f = c_function('element_bond_radius', args = (ctypes.c_void_p,), ret = ctypes.c_float)
        return f(e._c_pointer)

    @staticmethod
    def get_element(name_or_number):
        '''Get the Element that corresponds to an atomic name or number'''
        if type(name_or_number) == type(1):
            f = c_function('element_number_get_element', args = (ctypes.c_int,), ret = ctypes.c_void_p)
            f_arg = name_or_number
        elif type(name_or_number) == type(""):
            f = c_function('element_name_get_element', args = (ctypes.c_char_p,), ret = ctypes.c_void_p)
            f_arg = name_or_number.encode('utf-8')
        else:
            raise ValueError("'get_element' arg must be string or int")
        return _element(f(f_arg))
"""

# -----------------------------------------------------------------------------
#
from collections import namedtuple
ExtrudeValue = namedtuple("ExtrudeValue", ["vertices", "normals",
                                           "triangles", "colors",
                                           "front_band", "back_band"])

class RibbonXSection:
    '''
    A cross section that can extrude ribbons when given the
    required control points, tangents, normals and colors.
    '''
    def __init__(self, coords=None, coords2=None, normals=None, normals2=None,
                 faceted=False, tess=None, xs_pointer=None):
        if xs_pointer is None:
            f = c_function('rxsection_new',
                           args = (ctypes.py_object,        # coords
                                   ctypes.py_object,        # coords2
                                   ctypes.py_object,        # normals
                                   ctypes.py_object,        # normals2
                                   ctypes.c_bool,           # faceted
                                   ctypes.py_object),       # tess
                                   ret = ctypes.c_void_p)   # pointer to C++ instance
            xs_pointer = f(coords, coords2, normals, normals2, faceted, tess)
        set_c_pointer(self, xs_pointer)

    # cpp_pointer and deleted are "base class" methods, though for performance reasons
    # we are placing them directly in each class rather than using a base class,
    # and for readability by most programmers we avoid using metaclasses
    @property
    def cpp_pointer(self):
        '''Value that can be passed to C++ layer to be used as pointer (Python int)'''
        return self._c_pointer.value

    @property
    def deleted(self):
        '''Has the C++ side been deleted?'''
        return not hasattr(self, '_c_pointer')

    def extrude(self, centers, tangents, normals, color,
                cap_front, cap_back, offset):
        '''Return the points, normals and triangles for a ribbon.'''
        f = c_function('rxsection_extrude',
                       args = (ctypes.c_void_p,     # self
                               ctypes.py_object,    # centers
                               ctypes.py_object,    # tangents
                               ctypes.py_object,    # normals
                               ctypes.py_object,    # color
                               ctypes.c_bool,       # cap_front
                               ctypes.c_bool,       # cap_back
                               ctypes.c_int),       # offset
                       ret = ctypes.py_object)      # tuple
        t = f(self._c_pointer, centers, tangents, normals, color,
              cap_front, cap_back, offset)
        if t is not None:
            t = ExtrudeValue(*t)
        return t

    def blend(self, back_band, front_band):
        '''Return the triangles blending front and back halves of ribbon.'''
        f = c_function('rxsection_blend',
                       args = (ctypes.c_void_p,     # self
                               ctypes.py_object,    # back_band
                               ctypes.py_object),    # front_band
                       ret = ctypes.py_object)      # tuple
        t = f(self._c_pointer, back_band, front_band)
        return t

    def scale(self, scale):
        '''Return new cross section scaled by 2-tuple scale.'''
        f = c_function('rxsection_scale',
                       args = (ctypes.c_void_p,     # self
                               ctypes.c_float,      # x scale
                               ctypes.c_float),     # y scale
                       ret = ctypes.c_void_p)       # pointer to C++ instance
        p = f(self._c_pointer, scale[0], scale[1])
        return RibbonXSection(xs_pointer=p)

    def arrow(self, scales):
        '''Return new arrow cross section scaled by 2x2-tuple scale.'''
        f = c_function('rxsection_arrow',
                       args = (ctypes.c_void_p,     # self
                               ctypes.c_float,      # wide x scale
                               ctypes.c_float,      # wide y scale
                               ctypes.c_float,      # narrow x scale
                               ctypes.c_float),     # narrow y scale
                       ret = ctypes.c_void_p)       # pointer to C++ instance
        p = f(self._c_pointer, scales[0][0], scales[0][1], scales[1][0], scales[1][1])
        return RibbonXSection(xs_pointer=p)

# -----------------------------------------------------------------------------
#

# SeqMatchMaps are returned by C++ functions, but unlike most other classes in this file,
# they have no persistence in the C++ layer
class SeqMatchMap(State):
    """Class to track the matching between an alignment sequence and a structure sequence

       The match map can be indexed by either an integer (ungapped) sequence position,
       or by a Residue, which will return a Residue or a sequence position, respectively.
       The pos_to_res and res_to_pos attributes return dictionaries of the corresponding
       mapping.
    """
    def __init__(self, session, align_seq, struct_seq):
        self._pos_to_res = {}
        self._res_to_pos = {}
        self._align_seq = align_seq
        self._struct_seq = struct_seq
        self.session = session
        from . import get_triggers
        self._handler = get_triggers(session).add_handler("changes", self._atomic_changes)

    def __bool__(self):
        return bool(self._pos_to_res)

    def __contains__(self, i):
        if isinstance(i, int):
            return i in self._pos_to_res
        return i in self._res_to_pos

    def __getitem__(self, i):
        if isinstance(i, int):
            return self._pos_to_res[i]
        return self._res_to_pos[i]

    @property
    def align_seq(self):
        return self._align_seq

    def match(self, res, pos):
        self._pos_to_res[pos] = res
        self._res_to_pos[res] = pos
        if self._align_seq.circular:
            self._pos_to_res[pos + len(self._align_seq.ungapped())/2] = res

    @property
    def pos_to_res(self):
        return self._pos_to_res

    @property
    def res_to_pos(self):
        return self._res_to_pos

    @staticmethod
    def restore_snapshot(session, data):
        inst = SeqMatchMap(session, data['align seq'], data['struct seq'])
        inst._pos_to_res = data['pos to res']
        inst._res_to_pos = data['res to pos']
        return inst

    @property
    def struct_seq(self):
        return self._struct_seq

    def take_snapshot(self, session, flags):
        '''Gather session info; return version number'''
        data = {
            'align seq': self._align_seq,
            'pos to res': self._pos_to_res,
            'res to pos': self._res_to_pos,
            'struct seq': self._struct_seq,
            'version': 1
        }
        return data

    def _atomic_changes(self, trig_name, changes):
        if changes.num_deleted_residues() > 0:
            for r, i in list(self._res_to_pos.items()):
                if r.deleted:
                    del self._res_to_pos[r]
                    del self._pos_to_res[i]
                    if self._align_seq.circular:
                        del self._pos_to_res[i + len(self._align_seq.ungapped())/2]

    def __del__(self):
        self._pos_to_res.clear()
        self._res_to_pos.clear()
        from . import get_triggers
        get_triggers(self.session).remove_handler(self._handler)

# -----------------------------------------------------------------------------
#

# tell the C++ layer about class objects whose Python objects can be instantiated directly
# from C++ with just a pointer, and put functions in those classes for getting the instance
# from the pointer (needed by Collections)
from .pbgroup import PseudobondGroup
#for class_obj in [Atom, Bond, CoordSet, Element, PseudobondGroup, Pseudobond, Residue, Ring]:
for class_obj in [Bond, CoordSet, PseudobondGroup, Pseudobond, Residue, Ring]:
    cname = class_obj.__name__.lower()
    func_name = "set_" + cname + "_pyclass"
    f = c_function(func_name, args = (ctypes.py_object,))
    f(class_obj)

    if class_obj == PseudobondGroup:
        # put these funcs in PseudobondGroupData not PseudobondGroup
        class_obj = PseudobondGroupData
    func_name = cname + "_py_inst"
    class_obj.c_ptr_to_py_inst = lambda ptr, *, fname=func_name: c_function(fname,
        args = (ctypes.c_void_p,), ret = ctypes.py_object)(ctypes.c_void_p(int(ptr)))
    func_name = cname + "_existing_py_inst"
    class_obj.c_ptr_to_existing_py_inst = lambda ptr, *, fname=func_name: c_function(fname,
        args = (ctypes.c_void_p,), ret = ctypes.py_object)(ctypes.c_void_p(int(ptr)))

# Chain/StructureSeq/Sequence classes could theoretically be handled the same as the
# above classes, but the fact that classes are not first-class objects in C++ makes
# this extremely difficult, so therefore they are treated as not directly instantiable 
# from C++ and therefore require a different "<class>_py_inst" function...
for class_obj in [Sequence, StructureSeq, Chain]:
    cname = class_obj.__name__.lower()
    func_name = cname + "_existing_py_inst"
    class_obj.c_ptr_to_py_inst = lambda ptr, *, klass=class_obj, fname=func_name: c_function(fname,
        args = (ctypes.c_void_p,), ret = ctypes.py_object)(ctypes.c_void_p(int(ptr))) or klass(ptr)
    class_obj.c_ptr_to_existing_py_inst = lambda ptr, *, fname=func_name: c_function(fname,
        args = (ctypes.c_void_p,), ret = ctypes.py_object)(ctypes.c_void_p(int(ptr)))

# Structure/AtomicStructure cannot be instantiated with just a pointer, and therefore
# differs slightly from both the above...
StructureData.c_ptr_to_py_inst = lambda ptr: c_function("structure_py_inst",
    args = (ctypes.c_void_p,), ret = ctypes.py_object)(ctypes.c_void_p(int(ptr)))
StructureData.c_ptr_to_existing_py_inst = lambda ptr: c_function("structure_existing_py_inst",
    args = (ctypes.c_void_p,), ret = ctypes.py_object)(ctypes.c_void_p(int(ptr)))
