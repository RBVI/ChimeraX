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

from chimerax.core.state import State, StateManager
from numpy import uint8, int32, uint32, float64, float32, byte
npy_bool = bool
from .molc import CFunctions, string, cptr, pyobject, set_c_pointer, pointer, size_t
import ctypes
from . import ctypes_support as convert

# -------------------------------------------------------------------------------
# Access functions from libmolc C library.
#
import chimerax.arrays # Load libarrrays shared library before importing libmolc.
_atomic_c_functions = CFunctions('libmolc')
c_property = _atomic_c_functions.c_property
cvec_property = _atomic_c_functions.cvec_property
c_function = _atomic_c_functions.c_function
c_array_function = _atomic_c_functions.c_array_function

def python_instances_of_class(inst_class, *, open_only=True):
    f = c_function('python_instances_of_class', args = (ctypes.py_object,), ret = ctypes.py_object)
    instances = f(inst_class)
    if not open_only:
        return instances
    if issubclass(inst_class, PseudobondGroupData):
        filt = lambda x: (not x.structure) or x.structure.id
    elif hasattr(inst_class, 'structure'):
        filt = lambda x: x.structure.id
    elif issubclass(inst_class, StructureData):
        filt = lambda x: x.id
    elif issubclass(inst_class, Pseudobond):
        filt = lambda x: (not x.group.structure) or x.group.structure.id
    elif issubclass(inst_class, (PseudobondManager, Sequence)):
        filt = lambda x: True
    else:
        raise ValueError("Don't know how to determine open instances of class %s" % inst_class.__name__)
    return [x for x in instances if filt(x)]

# delay .cymol import until 'CFunctions' call above establishes lib path
from .cymol import CyAtom
class Atom(CyAtom, State):
    '''An atom in a (chemical) structure'''

    # So that attr-registration API can provide return-type info; provide that data here
    # [because Cython properties use immutable getset_descriptor slots, and the final address of a
    # property isn't obtainable until the end of the class definition, using this inelegant solution]
    _attr_reg_info = [
        ('alt_loc', (str,)), ('bfactor', (float,)), ('display', (bool,)), ('idatm_type', (str,)),
        ('is_side_connector', (bool,)), ('is_side_chain', (bool,)), ('is_side_only', (bool,)),
        ('name', (str,)), ('num_alt_locs', (int,)), ('num_bonds', (int,)), ('num_explicit_bonds', (int,)),
        ('occupancy', (float,)), ('radius', (float,)), ('selected', (bool,)), ('visible', (bool,)),
    ]

    # possibly long-term hack for interoperation with ctypes;
    # has to be here instead of CyAtom because super().__delattr__ doesn't work there
    def __delattr__(self, name):
        if name == "_c_pointer" or name == "_c_pointer_ref":
            self._deleted = True
        else:
            super().__delattr__(name)

    def take_snapshot(self, session, flags):
        data = {'structure': self.structure,
                'ses_id': self.structure.session_atom_to_id(self._c_pointer),
                'custom attrs': self.custom_attrs}
        return data

    @staticmethod
    def restore_snapshot(session, data):
        a = Atom.c_ptr_to_py_inst(data['structure'].session_id_to_atom(data['ses_id']))
        a.set_custom_attrs(data)
        return a
Atom.set_py_class(Atom)

# -----------------------------------------------------------------------------
#
class Bond(State):
    '''
    Bond connecting two atoms.

    To create a Bond use chimerax.atomic.struct_edit.add_bond()
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

    @property
    def atomspec(self):
        a1, a2 = self.atoms
        return a1.atomspec + a2.atomspec

    atoms = c_property('bond_atoms', cptr, 2, astype = convert.atom_pair, read_only = True,
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
    in_cycle = c_property('bond_in_cycle', npy_bool, read_only = True,
        doc = "Supported API. Is the bond in a cycles of bonds?  Boolean value.")
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

    def rings(self, cross_residue=False, all_size_threshold=0):
        '''Return :class:`.Rings` collection of rings this Bond is involved in.

        If 'cross_residue' is False, then rings that cross residue boundaries are not
        included.  If 'all_size_threshold' is zero, then return only minimal rings, of
        any size.  If it is greater than zero, then return all rings not larger than the
        given value.

        The returned rings are quite emphemeral, and shouldn't be cached or otherwise
        retained for long term use.  They may only live until the next call to rings()
        [from anywhere, including C++].
        '''
        f = c_function('bond_rings', args = (ctypes.c_void_p, ctypes.c_bool, ctypes.c_int),
                ret = ctypes.py_object)
        return convert.rings(f(self._c_pointer, cross_residue, all_size_threshold))

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
        return convert.atoms(f(self._c_pointer, side_atom._c_pointer))

    @property
    def smaller_side(self):
        '''Returns the bond atom on the side of the bond with fewer total atoms attached'''
        f = c_function('bond_smaller_side', args = (ctypes.c_void_p,), ret = ctypes.py_object)
        return f(self._c_pointer)

    @property
    def polymeric_start_atom(self):
        f = c_function('bond_polymeric_start_atom', args = (ctypes.c_void_p,), ret = ctypes.py_object)
        return f(self._c_pointer)

    def string(self, *, style=None, minimal=False, reversed=False):
        "Supported API.  Get text representation of Bond (also used by __str__ for printing)"
        a1, a2 = self.atoms
        if reversed:
            a1, a2 = a2, a1
        bond_sep = " \N{Left Right Arrow} "
        return a1.string(style=style, minimal=minimal) + bond_sep + a2.string(style=style, relative_to=a1)

    def take_snapshot(self, session, flags):
        data = {'structure': self.structure,
                'ses_id': self.structure.session_bond_to_id(self._c_pointer),
                'custom attrs': self.custom_attrs}
        return data

    @staticmethod
    def restore_snapshot(session, data):
        b = Bond.c_ptr_to_py_inst(data['structure'].session_id_to_bond(data['ses_id']))
        b.set_custom_attrs(data)
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

    atoms = c_property('pseudobond_atoms', cptr, 2, astype = convert.atom_pair, read_only = True,
        doc = "Supported API. Two-tuple of :py:class:`Atom` objects that are the bond end points.")
    color = c_property('pseudobond_color', uint8, 4,
        doc = "Supported API. Color RGBA length 4 sequence/array. Values in range 0-255")
    display = c_property('pseudobond_display', npy_bool, doc =
        "Whether to display the bond if both atoms are shown. "
        "Can be overriden by the hide attribute.")
    group = c_property('pseudobond_group', cptr, astype = convert.pseudobond_group, read_only = True,
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
        # tinyarray doesn't have .sum()
        return sqrt(sum(v*v))

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

    def take_snapshot(self, session, flags):
        data = {'group': self.group,
                'ses_id': self._ses_id,
                'custom attrs': self.custom_attrs}
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
            pb.set_custom_attrs(data)
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
    change_tracker = c_property('pseudobond_group_change_tracker', pyobject, read_only = True,
        doc = "The :class:`.ChangeTracker` currently in use by this pseudobond group. Read only.")
    color = c_property('pseudobond_group_color', uint8, 4,
        doc="Supported API. Sets the color attribute of current pseudobonds and new pseudobonds")
    group_type = c_property('pseudobond_group_group_type', uint8, read_only = True, doc=
        "Supported API. PseudobondGroup.GROUP_TYPE_NORMAL is a normal group,"
        "PseudobondGroup.GROUP_TYPE_COORD_SET is a per-coord-set pseudobond group")
    halfbond = c_property('pseudobond_group_halfbond', npy_bool,
        doc = "Sets the halfbond attribute of current pseudobonds and new pseudobonds")
    num_pseudobonds = c_property('pseudobond_group_num_pseudobonds', size_t, read_only = True,
        doc = "Supported API. Number of pseudobonds in group. Read only.")
    pseudobonds = c_property('pseudobond_group_pseudobonds', cptr, 'num_pseudobonds',
        astype = convert.pseudobonds, read_only = True,
        doc = "Supported API. Group pseudobonds as a :class:`.Pseudobonds` collection. Read only.")
    radius = c_property('pseudobond_group_radius', float32,
        doc = "Supported API. Sets the radius attribute of current pseudobonds and new pseudobonds")
    structure = c_property('pseudobond_group_structure', pyobject,
        read_only = True, doc ="Structure that pseudobond group is owned by.  "
        "Supported API. Returns None if called on a group managed by the global pseudobond manager")

    def change_name(self, name):
        f = c_function('pseudobond_group_change_category',
            args = (ctypes.c_void_p, ctypes.c_char_p))
        try:
            f(self._c_pointer, name.encode('utf-8'))
        except TypeError:
            from chimerax.core.errors import UserError
            raise UserError("Another pseudobond group is already named '%s'" % name)

    def clear(self):
        "Supported API. Delete all pseudobonds in group"
        f = c_function('pseudobond_group_clear', args = (ctypes.c_void_p,))
        f(self._c_pointer)

    def delete_pseudobond(self, pb):
        "Supported API. Delete a specific pseudobond from a group"
        f = c_function('pseudobond_group_delete_pseudobond',
            args = (ctypes.c_void_p, ctypes.c_void_p))
        f(self._c_pointer, pb._c_pointer)

    def get_num_pseudobonds(self, cs_id):
        '''Supported API. Get the number of pseudobonds for a particular coordinate set.
        Use the 'num_pseudobonds' property to get the number of pseudobonds for the current
        coordinate set.
        '''
        f = c_function('pseudobond_group_get_num_pseudobonds',
                       args = (ctypes.c_void_p, ctypes.c_int,), ret = ctypes.c_size_t)
        return f(self._c_pointer, cs_id)

    def get_pseudobonds(self, cs_id):
        '''Supported API. Get the pseudobonds for a particular coordinate set. Use the 'pseudobonds'
        property to get the pseudobonds for the current coordinate set.
        '''
        from numpy import empty
        ai = empty((self.get_num_pseudobonds(cs_id),), cptr)
        f = c_function('pseudobond_group_get_pseudobonds',
                       args = (ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p),
                       ret = ctypes.c_void_p)
        f(self._c_pointer, cs_id, pointer(ai))
        return convert.pseudobonds(ai)

    def new_pseudobond(self, atom1, atom2, cs_id = None):
        '''Supported API. Create a new pseudobond between the specified :class:`Atom` objects.
        If the pseudobond group supports per-coordset pseudobonds, you may
        specify a coordinate set ID (defaults to the current coordinate set).
        '''
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

    def new_pseudobonds(self, atoms1, atoms2):
        "Create new pseudobonds between the specified :class:`Atoms` atoms. "
        f = c_function('pseudobond_group_new_pseudobonds',
                       args = (ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int),
                       ret = ctypes.py_object)
        ptrs = f(self._c_pointer, atoms1._c_pointers, atoms2._c_pointers, len(atoms1))
        from .molarray import Pseudobonds
        return Pseudobonds(ptrs)

    # Graphics changed flags used by rendering code.  Private.
    _SHAPE_CHANGE = 0x1
    _COLOR_CHANGE = 0x2
    _SELECT_CHANGE = 0x4
    _RIBBON_CHANGE = 0x8
    _ADDDEL_CHANGE = 0x10
    _DISPLAY_CHANGE = 0x20
    _RING_CHANGE = 0x40
    _ALL_CHANGE = 0x6f  # not _ADDDEL_CHANGE
    _graphics_changed = c_property('pseudobond_group_graphics_change', int32)


# -----------------------------------------------------------------------------
#
class PseudobondManager(StateManager):
    '''Per-session singleton pseudobond manager keeps track of all :class:`.PseudobondGroupData` objects.'''

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
        "Supported API. Get an existing :class:`.PseudobondGroup` or create a new one with the given name."
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
                f = c_function('set_pseudobondgroup_py_instance',
                    args = (ctypes.c_void_p, ctypes.py_object))
                f(pbg_ptr, obj)
            obj_map[cat] = obj
        return obj_map

    def include_state(self):
        return bool(self.group_map)

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
                'custom attrs': self.custom_attrs}
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
        pbm.set_custom_attrs(data)
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
from .cymol import CyResidue
class Residue(CyResidue, State):
    '''
    A group of atoms such as an amino acid or nucleic acid. Every atom in
    an :class:`.AtomicStructure` belongs to a residue, including solvent and ions.

    To create a Residue use the :class:`.AtomicStructure` new_residue() method.
    '''

    # So that attr-registration API can provide return-type info; provide that data here
    # [because Cython properties use immutable getset_descriptor slots, and the final address of a
    # property isn't obtainable until the end of the class definition, using this inelegant solution]
    _attr_reg_info = [
        ('chi1', (float, None)), ('chi2', (float, None)), ('chi3', (float, None)), ('chi4', (float, None)),
        ('is_helix', (bool,)), ('is_strand', (bool,)), ('name', (str,)), ('num_atoms', (int,)),
        ('number', (int,)), ('omega', (float, None)), ('phi', (float, None)), ('psi', (float, None)),
    ]

    # possibly long-term hack for interoperation with ctypes;
    # has to be here instead of CyResidue because super().__delattr__ doesn't work there
    def __delattr__(self, name):
        if name == "_c_pointer" or name == "_c_pointer_ref":
            self._deleted = True
        else:
            super().__delattr__(name)

    def take_snapshot(self, session, flags):
        data = {'structure': self.structure,
                'ses_id': self.structure.session_residue_to_id(self._c_pointer),
                'custom attrs': self.custom_attrs}
        return data

    @staticmethod
    def restore_snapshot(session, data):
        r = Residue.c_ptr_to_py_inst(data['structure'].session_id_to_residue(data['ses_id']))
        r.set_custom_attrs(data)
        return r

    # C++ class variables are problematic for Cython (particularly a map of maps # where the key type
    # of the nested map is a varidic template!); so expose class variables via ctypes
    def ideal_chirality(self, atom_name):
        """Return the ideal chirality (N = none; R = right-handed (rectus); S = left-handed (sinister)
            for the given atom name in this residue.  The chirality is only known if the mmCIF chemical
            component for this residue has been read."""
        f = c_function('residue_ideal_chirality', args = (ctypes.c_char_p, ctypes.c_char_p),
            ret = ctypes.py_object)
        return f(self.name.encode('utf-8'), atom_name.encode('utf-8'))

    aa_min_backbone_names = c_function('residue_aa_min_backbone_names', args = (), ret = ctypes.py_object)()
    aa_max_backbone_names = c_function('residue_aa_max_backbone_names', args = (), ret = ctypes.py_object)()
    aa_side_connector_names = c_function('residue_aa_side_connector_names', args = (), ret = ctypes.py_object)()
    aa_min_ordered_backbone_names = c_function('residue_aa_min_ordered_backbone_names', args = (), ret = ctypes.py_object)()
    na_min_backbone_names = c_function('residue_na_min_backbone_names', args = (), ret = ctypes.py_object)()
    na_max_backbone_names = c_function('residue_na_max_backbone_names', args = (), ret = ctypes.py_object)()
    na_side_connector_names = c_function('residue_na_side_connector_names', args = (), ret = ctypes.py_object)()
    na_min_ordered_backbone_names = c_function('residue_na_min_ordered_backbone_names', args = (), ret = ctypes.py_object)()
    def clear_hide_bits(self, bit_mask, atoms_only=False):
        "Clear Residue's atoms' and bonds' hide bits in bit mask"
        f = c_array_function('residue_clear_hide_bits', args=(uint32, npy_bool), per_object=False)
        b_ref = ctypes.byref(self._c_pointer)
        f(b_ref, 1, bit_mask, atoms_only)

Residue.set_py_class(Residue)


# -----------------------------------------------------------------------------
#
class Ring:
    '''
    A ring in the structure.
    '''

    def __init__(self, ring_pointer):
        set_c_pointer(self, ring_pointer)
        # Rings in the C++ are "ephemeral"; they get destroyed whenever a new set of rings is computed
        # using different criteria, therefore for better reliability pre-fetch all the ring properties
        # when the ring is constructed
        self.__size = c_property('ring_size', size_t, read_only=True).fget(self)
        self.__atoms = c_property('ring_atoms', cptr, 'size', astype = convert.atoms, read_only = True
            ).fget(self)
        self.__bonds = c_property('ring_bonds', cptr, 'size', astype = convert.bonds, read_only = True
            ).fget(self)
        self.__ordered_atoms = c_property('ring_ordered_atoms', cptr, 'size', astype = convert.atoms,
            read_only=True).fget(self)
        self.__ordered_bonds = c_property('ring_ordered_bonds', cptr, 'size', astype = convert.bonds,
            read_only=True).fget(self)

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

    @property
    def aromatic(self):
        "Supported API. Whether the ring is aromatic. Boolean value. Read only"
        # Unlike the other attrs, don't prefetch since the call may then call idatm_type(), which
        # could cause the Ring to be destroyed!
        for a in self.__atoms:
            if a.element.name == "C" and a.idatm_type != "Car":
                return False
        return True

    @property
    def atoms(self):
        "Supported API. :class:`.Atoms` collection containing the atoms of the ring, in no particular order"
        " (see :meth:`.Ring.ordered_atoms`)"
        return self.__atoms

    @property
    def bonds(self):
        '''Supported API. :class:`.Bonds` collection containing the bonds of the ring,
        in no particular order (see :meth:`.Ring.ordered_bonds`)
        '''
        return self.__bonds

    @property
    def ordered_atoms(self):
        ":class:`.Atoms` collection containing the atoms of the ring, in ring order."
        return self.__ordered_atoms

    @property
    def ordered_bonds(self):
        ":class:`.Bonds` collection containing the bonds of the ring, in ring order."
        return self.__ordered_bonds

    @property
    def size(self):
        "Supported API. Number of atoms (and bonds) in the ring"
        return self.__size

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
    amino3to1 = protein3to1
    rname3to1 = lambda rn: c_function('sequence_rname3to1', args = (ctypes.c_char_p,),
        ret = ctypes.c_char)(rn.encode('utf-8')).decode('utf-8')

    protein1to3 = { 'A':'ALA', 'B':'ASX', 'C':'CYS', 'D':'ASP', 'E':'GLU', 'F':'PHE',
        'G':'GLY', 'H':'HIS', 'I':'ILE', 'K':'LYS', 'L':'LEU', 'M':'MET', 'N':'ASN',
        'O':'HYP', 'P':'PRO', 'Q':'GLN', 'R':'ARG', 'S':'SER', 'T':'THR', 'V':'VAL',
        'W':'TRP', 'Y':'TYR', 'Z':'GLX' }

    # the following colors for use by alignment/sequence viewers
    default_helix_fill_color = (1.0, 1.0, 0.8)
    default_helix_outline_color = tuple([chan/255.0 for chan in (218, 165, 32)]) # goldenrod
    default_strand_fill_color = (0.88, 1.0, 1.0) # light cyan
    default_strand_outline_color = tuple([0.75*chan for chan in default_strand_fill_color])

    chimerax_exiting = False

    def __init__(self, seq_pointer=None, *, name="sequence", characters=""):
        self.attrs = {} # miscellaneous attributes
        self.markups = {} # per-residue (strings or lists)
        self.numbering_start = None
        self._features = {}
        self.accession_id = {}
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
        # since this Sequence has been created in the Python layer, don't call
        # set_sequence_py_instance, since that will add a reference and the
        # Sequence will not be properly garbage collected

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
        "Supported API. A string representing the contents of the sequence")
    circular = c_property('sequence_circular', npy_bool, doc="Indicates the sequence involves"
        " a circular permutation; the sequence characters have been doubled, and residue"
        " correspondences of the first half implicitly exist in the second half as well."
        " Typically used in alignments to line up with sequences that aren't permuted.")
    name = c_property('sequence_name', string, doc="Supported API. The sequence name")

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
        if Sequence.chimerax_exiting:
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
    def feature_data_sources(self):
        from .seq_support import get_manager
        mgr = get_manager()
        return mgr.data_sources

    def features(self, *, data_source="all", fetch=True):
        from .seq_support import get_manager
        mgr = get_manager()
        if data_source == "all":
            if fetch:
                for ds in mgr.data_sources:
                    if ds not in self._features:
                        try:
                            self._features[ds] = mgr.get_features(self.characters, ds)
                        except mgr.DataSourceFailure:
                            pass
            return self._features
        if data_source not in self._features:
            if fetch:
                self._features[data_source] = mgr.get_features(self.characters, data_source)
            else:
                return {}
        return self._features[data_source]

    @property
    def full_name(self):
        return self.name

    def gapped_to_ungapped(self, index):
        """Supported API.  Given an index into the sequence,
           returns the corresponding index into the sequence as if gaps had been removed."""
        f = c_function('sequence_gapped_to_ungapped', args = (ctypes.c_void_p, ctypes.c_int),
            ret = ctypes.c_int)
        g2u = f(self._c_pointer, index)
        if g2u < 0:
            return None
        return g2u

    def __getitem__(self, key):
        return self.characters[key]

    def __hash__(self):
        return id(self)

    @staticmethod
    def is_gap_character(c):
        if len(c) != 1:
            raise ValueError("Argument to is_gap_character must be single-character string, not %s"
                % repr(c))
        return c_function('sequence_is_gap_character', args = (ctypes.c_char_p,),
            ret = ctypes.c_bool)(c.encode('utf-8'))

    def __len__(self):
        """Supported API. Sequence length"""
        f = c_function('sequence_len', args = (ctypes.c_void_p,), ret = ctypes.c_size_t)
        return f(self._c_pointer)

    @staticmethod
    def restore_snapshot(session, data):
        seq = Sequence()
        seq.set_state_from_snapshot(session, data)
        return seq

    def search(self, pattern, case_sensitive = False):
        """Search sequence for an egrep-style pattern.  Return a list of (index, length) tuples.
           The search ignores gap characters but returns values for the full sequence, including gaps."""
        f = c_function('sequence_search', args = (ctypes.c_void_p, ctypes.c_char_p, ctypes.c_bool),
            ret = ctypes.py_object)
        return f(self._c_pointer, pattern.encode('utf-8'), case_sensitive)

    def set_features(self, data_source, features):
        self._features[data_source] = features

    def __setitem__(self, key, val):
        chars = self.characters
        if isinstance(key, slice):
            start = key.start if key.start is not None else 0
            stop = key.stop if key.stop is not None else len(chars)
            self.characters = chars[:start] + val + chars[stop:]
        else:
            self.characters = chars[:key] + val + chars[key+1:]

    # no __str__, since it's unclear whether it should be self.name or self.characters

    def set_state_from_snapshot(self, session, data):
        self.name = data['name']
        self.characters = data['characters']
        self.attrs = data.get('attrs', {})
        self.markups = data.get('markups', {})
        self.numbering_start = data.get('numbering_start', None)
        self._features = data.get('features', {})
        self.accession_id = data.get('accession_id', {})
        self.set_custom_attrs(data)

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
            'custom attrs': self.custom_attrs, 'features': self._features,
            'accession_id': self.accession_id }
        return data

    def ungapped(self):
        """Supported API. String of sequence without gap characters"""
        f = c_function('sequence_ungapped', args = (ctypes.c_void_p,), ret = ctypes.py_object)
        return f(self._c_pointer)

    def ungapped_to_gapped(self, index):
        """Supported API.  Given an index into the sequence with gaps removed,
           returns the corresponding index into the full sequence."""
        f = c_function('sequence_ungapped_to_gapped', args = (ctypes.c_void_p, ctypes.c_int),
            ret = ctypes.c_int)
        return f(self._c_pointer, index)

    def _cpp_rename(self, old_name):
        # called from C++ layer when 'name' attr changed
        self._fire_trigger('rename', (self, old_name))

    def _fire_trigger(self, trig_name, arg):
        # If no one is listening to the trigger, don't create a delayed firing of the trigger
        # ... because ...
        # this class has a __del__ method that can execute multiple times because the
        # __del__ method in some cases can create a self reference.  If the only reference
        # back to this class is the delayed trigger handler below, then as the set of
        # trigger handlers is cleared, the __del__ can execute multiple times and the
        # dict/set-clearing code doesn't like that and can crash
        if not self.triggers.trigger_handlers(trig_name):
            return

        # when C++ layer notifies us directly of change, delay firing trigger until
        # next 'changes' trigger to ensure that entire C++ layer is in a consistent state
        def delayed(*args, trigs=self.triggers, trig_name=trig_name, trig_arg=arg):
            trigs.activate_trigger(trig_name, trig_arg)
            from chimerax.core.triggerset import DEREGISTER
            return DEREGISTER
        from chimerax.atomic import get_triggers
        atomic_trigs = get_triggers()
        atomic_trigs.add_handler('changes', delayed)


    @atexit.register
    def _exiting():
        Sequence.chimerax_exiting = True

# -----------------------------------------------------------------------------
#
class StructureSeq(Sequence):
    '''
    A sequence that has associated structure residues.

    Unlike the Chain subclass, StructureSeq will not change in size once created,
    though associated residues may change to None if those residues are deleted/closed.
    '''

    def __init__(self, sseq_pointer=None, *, chain_id=None, structure=None, polymer_type=Residue.PT_NONE):
        if sseq_pointer is None:
            sseq_pointer = c_function('sseq_new',
                args = (ctypes.c_char_p, ctypes.c_void_p, ctypes.c_int), ret = ctypes.c_void_p)(
                    chain_id.encode('utf-8'), structure._c_pointer, polymer_type)
        super().__init__(sseq_pointer)
        self.triggers.add_trigger('delete')
        self.triggers.add_trigger('characters changed')
        self.triggers.add_trigger('residues changed')
        from weakref import ref
        def proxy_handler(*args, rs=ref(self)):
            s = rs()
            if s:
                s._changes_cb(*args)
        from . import get_triggers
        self.changes_handler = get_triggers().add_handler('changes', proxy_handler)

    def __del__(self):
        if not self.chimerax_exiting:
            self.changes_handler.remove()

    def __lt__(self, other):
        # for sorting (objects of the same type)
        if self.structure != other.structure:
            return self.structure < other.structure
        if self.chain_id != other.chain_id:
            return self.chain_id < other.chain_id
        if self is other: # optimization to avoid comparing residue lists if possible
            return False
        # only happens for different chains in the same structure but with the same ID
        s_exist = self.existing_residues
        if not s_exist: return True
        o_exist = other.existing_residues
        if not o_exist: return False
        return s_exist[0] < o_exist[0]

    chain_id = c_property('sseq_chain_id', string)
    '''Chain identifier. Read only string.'''
    # characters read-only in StructureSeq/Chain (use bulk_set)
    characters = c_property('sequence_characters', string, doc=
        "Supported API. A string representing the contents of the sequence. Read only.")
    description = c_property('sseq_description', string, doc="description derived from PDB/mmCIF"
        " info and set by AtomicStructure constructor")
    existing_residues = c_property('sseq_residues', cptr, 'num_residues', astype = convert.non_null_residues, read_only = True)
    '''Supported API. :class:`.Residues` collection containing the residues of this sequence with existing structure, in order. Read only.'''
    from_seqres = c_property('sseq_from_seqres', npy_bool, doc = "Was the full sequence "
        " determined from SEQRES (or equivalent) records in the input file")
    num_existing_residues = c_property('sseq_num_existing_residues', size_t, read_only = True)
    '''Supported API. Number of residues in this sequence with existing structure. Read only.'''
    num_residues = c_property('sseq_num_residues', size_t, read_only = True)
    '''Supported API. Number of residues belonging to this sequence, including those without structure. Read only.'''
    polymer_type = c_property('sseq_polymer_type', uint8, read_only = True)
    '''Polymer type of this sequence. Same values as Residue.polymer_type, except should not return PT_NONE.'''
    residues = c_property('sseq_residues', cptr, 'num_residues', astype = convert.residues_or_nones,
        read_only = True, doc = "Supported API. List containing the residues of this sequence in order. "
        "Residues with no structure will be None. Read only.")
    structure = c_property('sseq_structure', pyobject, read_only = True)
    '''Supported API. :class:`.AtomicStructure` that this structure sequence comes from. Read only.'''

    # allow append/extend for now, since NeedlemanWunsch uses it

    def bulk_set(self, residues, characters, *, fire_triggers=True):
        '''Set all residues/characters of StructureSeq.  "characters" is a string or a list of characters.'''
        ptrs = [r._c_pointer.value if r else 0 for r in residues]
        if type(characters) == list:
            characters = "".join(characters)
        f = c_function('sseq_bulk_set', args = (ctypes.c_void_p, ctypes.py_object, ctypes.c_char_p))
        f(self._c_pointer, ptrs, characters.encode('utf-8'))
        if fire_triggers:
            self._fire_trigger('characters changed', self)
            self._fire_trigger('residues changed', self)

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
                if rem.startswith(part):
                    rem = rem[len(part):]
                    continue
            break
        if rem and not rem.isspace():
            name_part = " " + rem.strip()
        else:
            name_part = ""
        return "%s (#%s)%s" % (self.structure.name, self.structure.id_string, name_part)

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
        '''Supported API. Returns a dict that maps from :class:`.Residue` to an ungapped sequence position'''
        f = c_function('sseq_res_map', args = (ctypes.c_void_p,), ret = ctypes.py_object)
        ptr_map = f(self._c_pointer)
        obj_map = {}
        for res_ptr, pos in ptr_map.items():
            res = convert.residue(res_ptr)
            obj_map[res] = pos
        return obj_map

    def residue_at(self, index):
        '''Supported API. Return the Residue/None at the (ungapped) position 'index'.
        More efficient than self.residues[index] since the entire residues
        list isn't built/destroyed.
        '''
        f = c_function('sseq_residue_at', args = (ctypes.c_void_p, ctypes.c_size_t),
            ret = ctypes.c_void_p)
        return convert.residue_or_none(f(self._c_pointer, index))

    def residue_before(self, r):
        '''Return the residue at index one less than the given residue, or None if no such residue exists.'''
        pos = self.res_map[r]
        return None if pos == 0 else self.residue_at(pos-1)

    def residue_after(self, r):
        '''Return the residue at index one more than the given residue, or None if no such residue exists.'''
        pos = self.res_map[r]
        return None if pos+1 >= len(self) else self.residue_at(pos+1)

    @staticmethod
    def restore_snapshot(session, data):
        sseq = StructureSeq(chain_id=data['chain_id'], structure=data['structure'])
        Sequence.set_state_from_snapshot(sseq, session, data['Sequence'])
        sseq.description = data['description']
        sseq.bulk_set(data['residues'], sseq.characters, fire_triggers=False)
        sseq.description = data.get('description', None)
        sseq.set_custom_attrs(data)
        return sseq

    @property
    def session(self):
        "Supported API. Session that this StructureSeq is in"
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
            'custom attrs': self.custom_attrs
        }
        return data

    def _changes_cb(self, trig_name, changes):
        if "name changed" in changes.residue_reasons():
            updated_chars = []
            some_changed = False
            for res, cur_char in zip(self.residues, self.characters):
                if res:
                    uc = Sequence.rname3to1(res.name)
                    updated_chars.append(uc)
                    if uc != cur_char:
                        some_changed = True
                else:
                    updated_chars.append(cur_char)
            if some_changed:
                self.bulk_set(self.residues, ''.join(updated_chars), fire_triggers=False)
                self.from_seqres = False
                self._fire_trigger('characters changed', self)

    def _cpp_seq_demotion(self):
        # called from C++ layer when this should be demoted to Sequence
        numbering_start = self.numbering_start
        self._fire_trigger('delete', self)
        self.changes_handler.remove()
        self.__class__ = Sequence
        self.numbering_start = numbering_start

    def _cpp_structure_seq_demotion(self):
        # called from C++ layer when a Chain should be demoted to a StructureSeq
        self.__class__ = StructureSeq

    def _cpp_modified(self):
        # called from C++ layer when the residue list changes
        self._fire_trigger('residues changed', self)

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

def try_assoc(seq, sseq, assoc_params, *, max_errors = 6):
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
    mmap = SeqMatchMap(seq, sseq)
    for r, i in res_to_pos.items():
        mmap.match(convert.residue(r), i)
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

    @property
    def atomspec(self):
        return self.string(style="command")

    # also used by Residue
    @staticmethod
    def chain_id_to_atom_spec(chain_id): 
        if chain_id:
            if chain_id.isspace():
                id_text = "?"
            elif chain_id.isalnum():
                id_text = chain_id
            else:
                # use single quotes on the inside so that they can be used in 
                # cxcmd HTML contexts
                id_text = "/chain_id='%s'" % chain_id
        else:
            id_text = "?"
        return '/' + id_text

    @property
    def identity(self):
        """'Fake' attribute to allow for //identity="/A" tests"""
        class IdentityTester():
            def __init__(self, chain):
                self.chain = chain

            def __eq__(self, chain_spec):
                return self.chain.characters in self._get_test_set(chain_spec)

            def __ne__(self, chain_spec):
                return self.chain.characters not in self._get_test_set(chain_spec)

            def __str__(self):
                # raise TypeError so that the attribute-testing code can catch it and do the right thing
                raise TypeError("fake attribute")

            def _get_test_set(self, chain_spec):
                from chimerax.atomic import UniqueChainsArg
                from chimerax.core.commands import AnnotationError
                from chimerax.core.errors import UserError
                try:
                    chains, text, rest = UniqueChainsArg.parse(chain_spec, self.chain.structure.session)
                except AnnotationError:
                    raise UserError("Cannot parse chain specifier '%s' for identity attribute test"
                        % chain_spec)
                if rest:
                    raise UserError("Extraneous text after chain specifer in identity attribute test")
                return set([chain.characters for chain in chains])

        return IdentityTester(self)

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
        chain.set_custom_attrs(data)
        return chain

    def string(self, style=None, include_structure=None):
        chain_str = self.chain_id_to_atom_spec(self.chain_id)
        from .structure import Structure
        if include_structure is not False and (
        include_structure is True
        or len([s for s in self.structure.session.models.list() if isinstance(s, Structure)]) > 1
        or not chain_str):
            struct_string = self.structure.string(style=style)
        else:
            struct_string = ""
        return struct_string + chain_str

    def take_snapshot(self, session, flags):
        data = {
            'description': self.description,
            'ses_id': self.structure.session_chain_to_id(self._c_pointer),
            'structure': self.structure,
            'custom attrs': self.custom_attrs
        }
        return data

import string
chain_id_characters = string.ascii_uppercase + string.ascii_lowercase + '1234567890'
_cid_index = { c:i for i,c in enumerate(chain_id_characters) }
def next_chain_id(cid):
    if not cid or cid.isspace():
        return chain_id_characters[0]
    for col in range(len(cid)-1, -1, -1):
        try:
            next_index = _cid_index[cid[col]] + 1
        except KeyError:
            raise ValueError("Illegal chain ID character: %s" % repr(cid[col]))
        if next_index < len(chain_id_characters):
            return cid[:col] + chain_id_characters[next_index] + (chain_id_characters[0] * (len(cid)-col-1))
    return chain_id_characters[0] * (len(cid)+1)

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
    _ss_suppress_count = 0

    def __init__(self, mol_pointer=None, *, logger=None):
        if mol_pointer is None:
            # Create a new structure
            from .structure import AtomicStructure
            new_func = 'atomic_structure_new' \
                if isinstance(self, AtomicStructure) else 'structure_new'
            mol_pointer = c_function(new_func, args = (ctypes.py_object,),
                ret = ctypes.c_void_p)(logger)
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

    @staticmethod
    def begin_destructor_batching(trig_name, trig_data):
        c_function('structure_begin_destructor_batching', args = ())()

    @staticmethod
    def end_destructor_batching(trig_name, trig_data):
        c_function('structure_end_destructor_batching', args = ())()

    active_coordset_change_notify = c_property('structure_active_coordset_change_notify', npy_bool,
    doc = '''Whether notifications are issued when the active coordset is changed.  Should only be
        set to true when temporarily changing the active coordset in a Python script. Boolean''')
    active_coordset = c_property('structure_active_coordset', cptr, astype = convert.coordset,
        read_only = True, doc="Supported API. Currently active :class:`CoordSet`. Read only.")
    active_coordset_id = c_property('structure_active_coordset_id', int32,
        doc = "Supported API. Index of the active coordinate set.")
    alt_loc_change_notify = c_property('structure_alt_loc_change_notify', npy_bool, doc=
        '''Whether notifications are issued when altlocs are changed.  Should only be
        set to false when temporarily changing alt locs in a Python script. Boolean''')
    ss_change_notify = c_property('structure_ss_change_notify', npy_bool, doc=
        '''Whether notifications are issued when secondardy structure is changed.  Should only be
        set to false when temporarily changing secondary structure in a Python script. Boolean''')
    atoms = c_property('structure_atoms', cptr, 'num_atoms', astype = convert.atoms, read_only = True,
        doc = "Supported API. :class:`.Atoms` collection containing all atoms of the structure.")
    ball_scale = c_property('structure_ball_scale', float32,
        doc = "Scales sphere radius in ball-and-stick style.")
    bonds = c_property('structure_bonds', cptr, 'num_bonds', astype = convert.bonds, read_only = True,
        doc = ":class:`.Bonds` collection containing all bonds of the structure.")
    chains = c_property('structure_chains', cptr, 'num_chains', astype = convert.chains, read_only = True,
        doc = "Supported API. :class:`.Chains` collection containing all chains of the structure.")
    change_tracker = c_property('structure_change_tracker', pyobject, read_only = True,
        doc = "The :class:`.ChangeTracker` currently in use by this structure. Read only.")
    coordset_ids = c_property('structure_coordset_ids', int32, 'num_coordsets', read_only = True,
        doc = "Supported API. Return array of ids of all coordinate sets.")
    coordset_size = c_property('structure_coordset_size', int32, read_only = True,
        doc = "Supported API. Return the size of the active coordinate set array.")
    display = c_property('structure_display', npy_bool, doc =
        "Don't call this directly.  Use Model's 'display' attribute instead.  Only exposed so that "
        "Model's 'display' attribute can call it so that 'display changed' shows up in triggers.")
    idatm_failed = c_property('structure_idatm_failed', npy_bool, read_only = True,
        doc = "Supported API. Whether the IDATM computation failed for this structure. Boolean")
    idatm_valid = c_property('structure_idatm_valid', npy_bool,
        doc = "Supported API. Whether atoms have valid IDATM types set. Boolean")
    lower_case_chains = c_property('structure_lower_case_chains', npy_bool,
        doc = "Supported API. Structure has lower case chain ids. Boolean")
    num_atoms = c_property('structure_num_atoms', size_t, read_only = True,
        doc = "Supported API. Number of atoms in structure. Read only.")
    num_atoms_visible = c_property('structure_num_atoms_visible', size_t, read_only = True,
        doc = "Number of visible atoms in structure. Read only.")
    num_bonds = c_property('structure_num_bonds', size_t, read_only = True,
        doc = "Supported API. Number of bonds in structure. Read only.")
    num_bonds_visible = c_property('structure_num_bonds_visible', size_t, read_only = True,
        doc = "Number of visible bonds in structure. Read only.")
    num_coordsets = c_property('structure_num_coordsets', size_t, read_only = True,
        doc = "Supported API. Number of coordinate sets in structure. Read only.")
    num_chains = c_property('structure_num_chains', size_t, read_only = True,
        doc = "Supported API. Number of chains in structure. Read only.")
    num_ribbon_residues = c_property('structure_num_ribbon_residues', size_t, read_only = True,
        doc = "Supported API. Number of residues in structure shown as ribbon. Read only.")
    num_residues = c_property('structure_num_residues', size_t, read_only = True,
        doc = "Supported API. Number of residues in structure. Read only.")
    residues = c_property('structure_residues', cptr, 'num_residues', astype = convert.residues,
        read_only = True, doc = "Supported API. :class:`.Residues` collection containing the"
        " residues of this structure. Read only.")
    pbg_map = c_property('structure_pbg_map', pyobject, astype = convert.pseudobond_group_map,
        read_only = True, doc = "Supported API. Dictionary mapping name to"
        " :class:`.PseudobondGroup` for pseudobond groups belonging to this structure. Read only.")
    metadata = c_property('metadata', pyobject, read_only = True,
        doc = "Supported API. Dictionary with metadata. Read only.")
    def set_metadata_entry(self, key, values):
        """Set metadata dictionary entry"""
        f = c_array_function('set_metadata_entry', args=(pyobject, pyobject), per_object=False)
        s_ref = ctypes.byref(self._c_pointer)
        f(s_ref, 1, key, values)
    pdb_version = c_property('pdb_version', int32, doc = "If this structure came from a PDB file,"
        " the major PDB version number of that file (2 or 3). Read only.")
    res_numbering = c_property('structure_res_numbering', int32,
        doc = "Numbering scheme for residues.  One of Residue.RN_AUTHOR/RN_CANONICAL/RN_UNIPROT")
    ribbon_tether_scale = c_property('structure_ribbon_tether_scale', float32,
        doc = "Ribbon tether thickness scale factor"
        " (1.0 = match displayed atom radius, 0=invisible).")
    ribbon_tether_shape = c_property('structure_ribbon_tether_shape', int32,
        doc = "Ribbon tether shape. Integer value.")
    TETHER_CONE = 0
    '''Tether is cone with point at ribbon.'''
    TETHER_REVERSE_CONE = 1
    '''Tether is cone with point at atom.'''
    TETHER_CYLINDER = 2
    '''Tether is cylinder.'''
    ribbon_show_spine = c_property('structure_ribbon_show_spine', npy_bool,
        doc = "Display ribbon spine. Boolean.")
    ribbon_orientation = c_property('structure_ribbon_orientation', int32,
        doc = "Ribbon orientation. Integer value.")
    RIBBON_ORIENT_GUIDES = 1
    '''Ribbon orientation from guide atoms.'''
    RIBBON_ORIENT_ATOMS = 2
    '''Ribbon orientation from interpolated atoms.'''
    RIBBON_ORIENT_CURVATURE = 3
    '''Ribbon orientation perpendicular to ribbon curvature.'''
    RIBBON_ORIENT_PEPTIDE = 4
    '''Ribbon orientation perpendicular to peptide planes.'''
    ribbon_display_count = c_property('structure_ribbon_display_count', int32, read_only = True,
        doc = "Return number of residues with ribbon display set. Integer.")
    ribbon_tether_sides = c_property('structure_ribbon_tether_sides', int32,
        doc = "Number of sides for ribbon tether. Integer value.")
    ribbon_tether_opacity = c_property('structure_ribbon_tether_opacity', float32,
        doc = "Ribbon tether opacity scale factor (relative to the atom).")
    ribbon_mode_helix = c_property('structure_ribbon_mode_helix', int32,
        doc = "Ribbon mode for helices. Integer value.")
    ribbon_mode_strand = c_property('structure_ribbon_mode_strand', int32,
        doc = "Ribbon mode for strands. Integer value.")
    RIBBON_MODE_DEFAULT = 0
    '''Default ribbon mode showing secondary structure with ribbons.'''
    RIBBON_MODE_ARC = 1
    '''Ribbon mode showing secondary structure as an arc (tube or plank).'''
    RIBBON_MODE_WRAP = 2
    '''Ribbon mode showing helix as ribbon wrapped around tube.'''
    ring_display_count = c_property('structure_ring_display_count', int32, read_only = True,
        doc = "Return number of residues with ring display set. Integer.")

    ss_assigned = c_property('structure_ss_assigned', npy_bool, doc =
        "Has secondary structure been assigned, either by data in original structure file "
        "or by some algorithm (e.g. dssp command)")

    from contextlib import contextmanager
    @contextmanager
    def suppress_ss_change_notifications(self):
        """Suppress secondard structure change notifications while the code body runs.
           Restore the original secondard structure of this atom when done."""
        orig_ss_types = self.residues.ss_types
        orig_ss_ids = self.residues.ss_ids
        if self._ss_suppress_count == 0:
            self.ss_change_notify = False
        self._ss_suppress_count += 1
        try:
            yield
        finally:
            self.residues.ss_types = orig_ss_types
            self.residues.ss_ids = orig_ss_ids
            self._ss_suppress_count -= 1
            if self._ss_suppress_count == 0:
                self.ss_change_notify = True

    def _combine(self, s, chain_id_map, ref_xform):
        f = c_function('structure_combine', args = (ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.py_object))
        if s.scene_position == ref_xform:
            pos_ptr = 0
        else:
            pos_ptr = pointer((ref_xform.inverse() * s.scene_position).matrix)
        f(s._c_pointer, self._c_pointer, pos_ptr, chain_id_map)

    def _copy(self):
        f = c_function('structure_copy', args = (ctypes.c_void_p,), ret = ctypes.c_void_p)
        p = f(self._c_pointer)
        return p

    def bonded_groups(self, *, consider_missing_structure=True):
        '''Find bonded groups of atoms.  Returns a list of Atoms collections'''
        f = c_function('structure_bonded_groups', args = (ctypes.c_void_p, ctypes.c_bool),
            ret = ctypes.py_object)
        from .molarray import Atoms
        import numpy
        return [Atoms(numpy.array(x, numpy.uintp)) for x in f(self._c_pointer, consider_missing_structure)]

    def chain_trace_atoms(self):
        '''
        Find pairs of atoms that should be connected in a chain trace.
        Returns None or a 2-tuple of two Atoms instances where corresponding atoms
        should be connected.  A chain trace connects two adjacent CA atoms if both
        atoms are shown but the intervening C and N atoms are not shown, *and* no ribbon
        depiction connects the residues.  Adjacent means that there is a bond between the
        two residues.  So for instance CA-only structures has no bond between the residues
        and those do not show a chain trace connection, instead they show a "missing structure"
        connection.  For nucleic acid chains adjacent displayed P atoms with undisplayed
        intervening O3' and O5' atoms are part of a chain trace.
        '''
        f = c_function('structure_chain_trace_atoms', args = (ctypes.c_void_p,), ret = ctypes.py_object)
        ap = f(self._c_pointer)
        if ap is None:
            return None
        else:
            from .molarray import Atoms
            return (Atoms(ap[0]), Atoms(ap[1]))

    def change_chain_ids(self, chains, chain_ids, *, non_polymeric=True):
        '''Change the chain IDs of the given chains to the corresponding chain ID.  The final ID
           must not conflict with other unchanged chains of the structure.  If 'non_polymeric' is
           True, then non-polymeric residues with the same chain ID as any of the given change
           will also have their chain ID changed in the same way.
        '''
        f = c_function('structure_change_chain_ids',
            args = (ctypes.c_void_p, ctypes.py_object, ctypes.py_object, ctypes.c_bool))
        f(self._c_pointer, [c._c_pointer.value for c in chains], chain_ids, non_polymeric)

    def combine_sym_atoms(self):
        '''Combine "symmetry" atoms, which for this purpose is atoms with the same element type
           on the exact same 3D position'''
        f = c_function('structure_combine_sym_atoms', args = (ctypes.c_void_p,))(self._c_pointer)

    def add_coordset(self, id, xyz):
        '''Supported API. Add a coordinate set with the given id.'''
        if xyz.dtype != float64:
            raise ValueError('add_coordset(): array must be float64, got %s' % xyz.dtype.name)
        f = c_function('structure_add_coordset',
                       args = (ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t))
        f(self._c_pointer, id, pointer(xyz), len(xyz))

    def add_coordsets(self, xyzs, replace = True):
        '''Add coordinate sets.  If 'replace' is True, clear out existing coordinate sets first'''
        if len(xyzs.shape) != 3:
            raise ValueError('add_coordsets(): array must be (frames)x(atoms)x3-dimensional')
        if not xyzs.flags.c_contiguous:
            # molc.cpp code doesn't know about strides...
            xyzs = xyzs.copy()
        cs_size = self.coordset_size
        if cs_size > 0:
            dim_check = cs_size
            check_text = "previous coordinate sets"
            do_check = True
        else:
            dim_check = self.num_atoms
            check_text = "number of atoms"
            do_check = dim_check > 0
        if do_check and xyzs.shape[1] != dim_check:
            raise ValueError('add_coordsets(): second dimension of coordinate array'
                ' must be same as %s' % check_text)
        if xyzs.shape[2] != 3:
            raise ValueError('add_coordsets(): third dimension of coordinate array'
                ' must be 3 (xyz)')
        if xyzs.dtype != float64:
            raise ValueError('add_coordsets(): array must be float64, got %s' % xyzs.dtype.name)
        f = c_function('structure_add_coordsets',
                       args = (ctypes.c_void_p, ctypes.c_bool, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t))
        f(self._c_pointer, replace, pointer(xyzs), *xyzs.shape[:2])

    def remove_coordsets(self):
        '''Remove all coordinate sets.'''
        f = c_function('structure_remove_coordsets', args = (ctypes.c_void_p,))
        f(self._c_pointer)

    def coordset(self, cs_id):
        '''Supported API. Return the CoordSet for the given coordset ID'''
        f = c_function('structure_py_obj_coordset', args = (ctypes.c_void_p, ctypes.c_int),
            ret = ctypes.py_object)
        return f(self._c_pointer, cs_id)

    def connect_structure(self, *, bond_length_tolerance=0.4, metal_coordination_distance=3.6):
        '''Generate bonds and relevant pseudobonds (missing structure; metal coordination)
        for structure.  Typically used for structures where only the atom positions and not
        the connectivity is known.
        
        'bond_length_tolerance' is how much longer the inter-atom distance can exceed the
        ideal bond length and still have a bond created between the atoms.
        
        'metal_coordination_distance' is the maximum distance between a metal and a possibly
        coordinating atom that will generate a metal-coordination pseudobond.
        '''
        from chimerax.connect_structure._cs import connect_structure as connect_struct
        connect_struct(self.cpp_pointer, bond_length_tolerance, metal_coordination_distance)

    def delete_alt_locs(self):
        '''Incorporate current alt locs as "regular" atoms and remove other alt locs'''
        f = c_function('structure_delete_alt_locs', args = (ctypes.c_void_p,))(self._c_pointer)

    def delete_atom(self, atom):
        '''Supported API. Delete the specified Atom.'''
        f = c_function('structure_delete_atom', args = (ctypes.c_void_p, ctypes.c_void_p))
        f(self._c_pointer, atom._c_pointer)

    def delete_bond(self, bond):
        '''Supported API. Delete the specified Bond.'''
        f = c_function('structure_delete_bond', args = (ctypes.c_void_p, ctypes.c_void_p))
        f(self._c_pointer, bond._c_pointer)

    def delete_residue(self, residue):
        '''Supported API.  Delete the specified Residue.
           *Rarely* needed, since deleting atoms will delete empty residues automatically.  Can be needed
           when moving atoms from one residue to another, leaving an empty residue that needs deletion.
        '''
        f = c_function('structure_delete_residue', args = (ctypes.c_void_p, ctypes.c_void_p))
        f(self._c_pointer, residue._c_pointer)

    def find_residue(self, chain_id, pos, insert=' '):
        """Supported API.  Find a residue in the structure.  Returns None if none match."""
        f = c_function('structure_find_residue',
               args = (ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int, ctypes.c_char),
               ret = ctypes.py_object)
        return f(self._c_pointer, chain_id.encode('utf-8'), pos, insert.encode('utf-8'))

    @property
    def frag_sel(self):
        # special purpose function for the "connected fragment" selection level;
        # returns a mask of connected fragment atoms involving currently selected atoms
        f = c_function('structure_frag_sel', args = (ctypes.c_void_p,), ret = ctypes.py_object)
        return f(self._c_pointer)

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
        '''Supported API. Create a new :class:`.Atom` object. It must be added to a
        :class:`.Residue` object belonging to this structure before being used.
        'element' can be a string (atomic symbol), an integer (atomic number),
        or an Element instance.  It is advisible to add the atom to its residue
        as soon as possible since errors that occur in between can crash ChimeraX.
        Also, there are functions in chimerax.atomic.struct_edit for adding atoms
        that are considerably less tedious and error-prone than using new_atom()
        and related calls.'''
        if not isinstance(element, Element):
            element = Element.get_element(element)
        f = c_function('structure_new_atom',
                       args = (ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p),
                       ret = ctypes.py_object)
        return f(self._c_pointer, atom_name.encode('utf-8'), element._c_pointer)

    def new_bond(self, atom1, atom2):
        '''Supported API. Create a new :class:`.Bond` joining two :class:`Atom` objects.
        In most cases one should use chimerax.atomic.struct_edit.add_bond() instead, which
        does a lot of maintenance of data structures that new_bond() alone does not.
        '''
        f = c_function('structure_new_bond',
                       args = (ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p),
                       ret = ctypes.py_object)
        return f(self._c_pointer, atom1._c_pointer, atom2._c_pointer)

    def new_coordset(self, index=None, size=None):
        '''Supported API. Create a new empty coordset.  In almost all circumstances one would
            use the add_coordset(s) method instead (to add fully populated coordsets), but in
            some cases when building a Structure from scratch this method is needed.

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

    def new_residue(self, residue_name, chain_id, pos, insert=None, *, precedes=None):
        ''' Supported API. Create a new :class:`.Residue`.
            If 'precedes' is None, new residue will be appended to residue list, otherwise the
            new residue will be inserted before the 'precedes' resdidue.
        '''
        if not insert:
            insert = ' '
        if not chain_id:
            chain_id = ' '
        f = c_function('structure_new_residue',
                       args = (ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int, ctypes.c_char, ctypes.c_void_p),
                       ret = ctypes.py_object)
        return f(self._c_pointer, residue_name.encode('utf-8'), chain_id.encode('utf-8'), pos, insert.encode('utf-8'), ctypes.c_void_p(0) if precedes is None else precedes._c_pointer)

    @property
    def nonstandard_residue_names(self):
        '''"ligand-y" residue names in this structure'''
        f = c_function('structure_nonstandard_residue_names',
                       args = (ctypes.c_void_p,), ret = ctypes.py_object)
        return f(self._c_pointer)

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
        '''Supported API. Get or create a :class:`.PseudobondGroup` belonging to this structure.
           The 'create_type' parameter controls if and how the pseudobond is created, as per:

           0 (also: None)
             If no such group exists, none is created and None is returned

           1 (also: "normal")
             A "normal" pseudobond group will be created if necessary, one where the pseudobonds
             apply to all coordinate sets

           2 (also: "per coordset")
             A "per coordset" pseudobond group will be created if necessary, one where different
             coordsets can have different pseudobonds
        '''
        if isinstance(create_type, int):
            create_arg = create_type
        elif create_type is None:
            create_arg = 0
        elif create_type == "normal":
            create_arg = 1
        else:  # per coordset
            create_arg = 2
        f = c_function('structure_pseudobond_group',
                       args = (ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int),
                       ret = ctypes.py_object)
        # if the group is being created, the C++ layer will call the Python constructor, which
        # in turn will add the group to the open models.  Depending on what trigger handlers
        # do, this could result in a loop of the C++ layer trying to create Python instances,
        # so suppress the trigger until the C++ call returns.
        from chimerax.core.models import ADD_MODELS
        with self.session.triggers.block_trigger(ADD_MODELS):
            return f(self._c_pointer, name.encode('utf-8'), create_arg)

    def _delete_pseudobond_group(self, pbg):
        f = c_function('structure_delete_pseudobond_group',
                       args = (ctypes.c_void_p, ctypes.c_void_p), ret = None)
        f(self._c_pointer, pbg._c_pointer)

    def renumber_residues(self, renumbered, start):
        '''Renumber the given residues ('renumbered'), starting from the integer 'start'.
           Residues must be in the same chain and the resulting numbering must not conflict
           with other residues in the same chain (unless those residues have non-blank insertion
           codes).  The renumbering will set insertion codes to blanks.  The renumbering does NOT
           reorder the residues (which determines sequence order).  Use reorder_residues() for that.
        '''
        f = c_function('structure_renumber_residues',
            args = (ctypes.c_void_p, ctypes.py_object, ctypes.c_int))
        f(self._c_pointer, [r._c_pointer.value for r in renumbered], start)

    def reorder_residues(self, new_order):
        '''Reorder the residues.  Obviously, 'new_order' has to have exactly the same
           residues as the structure currently has.
        '''
        f = c_function('structure_reorder_residues', args = (ctypes.c_void_p, ctypes.py_object))
        f(self._c_pointer, [r._c_pointer.value for r in new_order])

    def res_numbering_valid(self, res_numbering):
        '''Is a particular residue-numbering scheme (author, UniProt) valid for this structure?'''
        f = c_function('structure_res_numbering_valid', args = (ctypes.c_void_p, ctypes.c_int),
            ret = ctypes.c_bool)
        return f(self._c_pointer, res_numbering)

    @classmethod
    def restore_snapshot(cls, session, data):
        g = StructureData(logger=session.logger)
        g.set_state_from_snapshot(session, data)
        return g

    def ribbon_orients(self, residues=None):
        '''Return array of orientation values for given residues.'''
        if residues is None:
            residues = self.residues
        f = c_function('structure_ribbon_orient', args = (ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t), ret = ctypes.py_object)
        return f(self._c_pointer, residues._c_pointers, len(residues))

    def rings(self, cross_residue=False, all_size_threshold=0):
        '''Return :class:`.Rings` collection of rings found in this Structure.

        If 'cross_residue' is False, then rings that cross residue boundaries are not
        included.  If 'all_size_threshold' is zero, then return only minimal rings, of
        any size.  If it is greater than zero, then return all rings not larger than the
        given value.

        The returned rings are quite emphemeral, and shouldn't be cached or otherwise
        retained for long term use.  They may only live until the next call to rings()
        [from anywhere, including C++].
        '''
        f = c_function('structure_rings', args = (ctypes.c_void_p, ctypes.c_bool, ctypes.c_int),
                ret = ctypes.py_object)
        return convert.rings(f(self._c_pointer, cross_residue, all_size_threshold))

    def set_state_from_snapshot(self, session, data):
        '''Restore from session info'''
        self._ses_call("restore_setup")
        f = c_function('structure_session_restore',
                args = (ctypes.c_void_p, ctypes.c_int,
                        ctypes.py_object, ctypes.py_object, ctypes.py_object))
        try:
            f(self._c_pointer, data['version'], tuple(data['ints']), tuple(data['floats']),
                tuple(data['misc']))
        except TypeError as e:
            if "Don't know how to restore new session data" in str(e):
                from chimerax.core.session import RestoreError
                raise RestoreError(str(e))
            raise
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

    def set_res_numbering_valid(self, res_numbering, valid=True):
        '''Indicate whether a particular residue-numbering scheme (author, UniProt) is valid for this structure'''
        f = c_function('set_structure_res_numbering_valid',
                    args = (ctypes.c_void_p, ctypes.c_int, ctypes.c_bool))
        f(self._c_pointer, res_numbering, valid)

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
    _RING_CHANGE = 0x40
    _ALL_CHANGE = 0x7f		# Mask including all change bits
    _graphics_changed = c_property('structure_graphics_change', int32)

# -----------------------------------------------------------------------------
#
class CoordSet(State):
    '''
    The coordinates for one frame of a Structure

    To create a CoordSet use the :class:`.AtomicStructure` new_coordset() method.
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

    @property
    def xyzs(self):
        "Numpy array of coordinates"
        f = c_function('coordset_xyzs', args = (ctypes.c_void_p,), ret = ctypes.py_object)
        return f(self._c_pointer)

    def take_snapshot(self, session, flags):
        data = {'structure': self.structure, 'cs_id': self.id,
                'custom attrs': self.custom_attrs}
        return data

    @staticmethod
    def restore_snapshot(session, data):
        cs = data['structure'].coordset(data['cs_id'])
        cs.set_custom_attrs(data)
        return cs

# -----------------------------------------------------------------------------
#
class ChangeTracker:
    '''Per-session singleton change tracker keeps track of all
    atomic data changes'''

    def __init__(self, ct_pointer=None):
        if ct_pointer is None:
            f = c_function('change_tracker_create', args = (), ret = ctypes.c_void_p)
            set_c_pointer(self, f())
        else:
            set_c_pointer(self, ct_pointer)
        f = c_function('set_changetracker_py_instance', args = (ctypes.c_void_p, ctypes.py_object))
        f(self._c_pointer, self)
        self.tracked_classes = frozenset([Atom, Bond, Pseudobond, Residue, Chain, StructureData,
            PseudobondGroupData, CoordSet])


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
        # So to avoid having all code test whether a structure is open or not,
        # have this call use the structure's own change tracker rather than self
        f = c_function('change_tracker_add_modified',
            args = (ctypes.c_int, ctypes.c_void_p, ctypes.c_char_p))
        from .molarray import Collection
        from collections.abc import Iterable
        if isinstance(modded, Collection):
            class_num = self._class_to_int(modded.object_class)
            for ptr in modded.pointers:
                f(class_num, int(ptr), reason.encode('utf-8'))
        else:
            try:
                iterable_test = iter(modded)
            except TypeError:
                f(self._inst_to_int(modded), modded._c_pointer, reason.encode('utf-8'))
            else:
                for item in modded:
                    f(self._inst_to_int(item), item._c_pointer, reason.encode('utf-8'))

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
            from . import molarray
            for k, v in data.items():
                created_ptrs, mod_ptrs, reasons, tot_del = v
                collection = getattr(molarray, k + 's')
                fc_key = k[:-4] if k.endswith("Data") else k
                final_changes[fc_key] = Changes(collection(created_ptrs),
                    collection(mod_ptrs), reasons, tot_del)
            return final_changes
        global_changes = process_changes(global_data)
        per_structure_changes = {}
        for s_ptr, structure_data in per_structure_data.items():
            per_structure_changes[convert.atomic_structure(s_ptr)] = process_changes(structure_data)
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
    def __init__(self, align_seq, struct_seq):
        self._pos_to_res = {}
        self._res_to_pos = {}
        self._align_seq = align_seq
        self._struct_seq = struct_seq
        from . import get_triggers
        self._handler = get_triggers().add_handler("changes", self._atomic_changes)
        from chimerax.core.triggerset import TriggerSet
        self.triggers = TriggerSet()
        self.triggers.add_trigger('modified')

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

    def __len__(self):
        return len(self._pos_to_res)

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
        inst = SeqMatchMap(data['align seq'], data['struct seq'])
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
            modified = False
            for r, i in list(self._res_to_pos.items()):
                if r.deleted:
                    modified = True
                    del self._res_to_pos[r]
                    del self._pos_to_res[i]
                    if self._align_seq.circular:
                        del self._pos_to_res[i + len(self._align_seq.ungapped())/2]
            if modified:
                self.triggers.activate_trigger('modified', self)

    def __del__(self):
        self._pos_to_res.clear()
        self._res_to_pos.clear()
        from . import get_triggers
        get_triggers().remove_handler(self._handler)

# -----------------------------------------------------------------------------
#

# tell the C++ layer about class objects whose Python objects can be instantiated directly
# from C++ with just a pointer, and put functions in those classes for getting the instance
# from the pointer (needed by Collections)
from .pbgroup import PseudobondGroup
#for class_obj in [Atom, Bond, CoordSet, Element, PseudobondGroup, Pseudobond, Residue, Ring]:
for class_obj in [Bond, ChangeTracker, CoordSet, PseudobondGroup, Pseudobond, Ring]:
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
