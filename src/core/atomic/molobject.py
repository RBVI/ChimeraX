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

from numpy import uint8, int32, float64, float32, byte, bool as npy_bool
from .molc import string, cptr, pyobject, c_property, set_c_pointer, c_function, c_array_function, ctype_type_to_numpy, pointer
import ctypes
size_t = ctype_type_to_numpy[ctypes.c_size_t]   # numpy dtype for size_t

# -------------------------------------------------------------------------------
# These routines convert C++ pointers to Python objects and are used for defining
# the object properties.
#
def _atoms(p):
    from .molarray import Atoms
    return Atoms(p)
def _atom_pair(p):
    return (object_map(p[0],Atom), object_map(p[1],Atom))
def _atom_or_none(p):
    return object_map(p, Atom) if p else None
def _bonds(p):
    from .molarray import Bonds
    return Bonds(p)
def _chain(p):
    if not p: return None
    return object_map(p, Chain)
def _element(p):
    return object_map(p, Element)
def _pseudobonds(p):
    from .molarray import Pseudobonds
    return Pseudobonds(p)
def _residue(p):
    return object_map(p, Residue)
def _residues(p):
    from .molarray import Residues
    return Residues(p)
def _non_null_residues(p):
    from .molarray import Residues
    return Residues(p[p!=0])
def _residues_or_nones(p):
    return [_residue(rptr) if rptr else None for rptr in p]
def _chains(p):
    from .molarray import Chains
    return Chains(p)
def _atomic_structure(p):
    if not p: return None
    return object_map(p, StructureData)
def _pseudobond_group(p):
    from .pbgroup import PseudobondGroup
    return object_map(p, PseudobondGroup)
def _pseudobond_group_map(pbgc_map):
    pbg_map = dict((name, _pseudobond_group(pbg)) for name, pbg in pbgc_map.items())
    return pbg_map

# -----------------------------------------------------------------------------
#
class Atom:
    '''
    An atom includes physical and graphical properties such as an element name,
    coordinates in space, and color and radius for rendering.

    To create an Atom use the :class:`.AtomicStructure` new_atom() method.
    '''

    SPHERE_STYLE, BALL_STYLE, STICK_STYLE = range(3)

    HIDE_RIBBON = 0x1
    BBE_MIN, BBE_RIBBON, BBE_MAX = range(3)

    def __init__(self, c_pointer):
        set_c_pointer(self, c_pointer)

    def __str__(self, atom_only = False):
        from ..core_settings import settings
        cmd_style = settings.atomspec_contents == "command-line specifier"
        if cmd_style:
            atom_str = '@' + self.name
        else:
            atom_str = self.name
        if atom_only:
            return atom_str
        if cmd_style:
            return '%s%s' % (str(self.residue), atom_str)
        return '%s %s' % (str(self.residue), atom_str)

    alt_loc = c_property('atom_alt_loc', byte, doc='Alternate location indicator')
    bfactor = c_property('atom_bfactor', float32, doc = "B-factor, floating point value.")
    bonds = c_property('atom_bonds', cptr, "num_bonds", astype=_bonds, read_only=True,
        doc="Bonds connected to this atom as an array of :py:class:`Bonds` objects. Read only.")
    chain_id = c_property('atom_chain_id', string, read_only = True,
        doc = "Protein Data Bank chain identifier. Limited to 4 characters. Read only string.")
    color = c_property('atom_color', uint8, 4, doc="Color RGBA length 4 numpy uint8 array.")
    coord = c_property('atom_coord', float64, 3,
        doc="Coordinates as a numpy length 3 array, 64-bit float values.")
    display = c_property('atom_display', npy_bool,
        doc="Whether to display the atom. Boolean value.")
    draw_mode = c_property('atom_draw_mode', uint8,
        doc="Controls how the atom is depicted.\n\n|  Possible values:\n"
        "SPHERE_STYLE\n"
        "    Use full atom radius\n"
        "BALL_STYLE\n"
        "    Use reduced atom radius, but larger than bond radius\n"
        "STICK_STYLE\n"
        "    Match bond radius")
    element = c_property('atom_element', cptr, astype = _element, read_only = True,
        doc =  ":class:`Element` corresponding to the chemical element for the atom.")
    element_name = c_property('atom_element_name', string, read_only = True,
        doc = "Chemical element name. Read only.")
    element_number = c_property('atom_element_number', uint8, read_only = True,
        doc = "Chemical element number. Read only.")
    hide = c_property('atom_hide', int32,
        doc="Whether atom is hidden (overrides display).  Integer bitmask."
        "\n\n|  Possible values:\n"
        "HIDE_RIBBON\n"
        "    Hide mask for backbone atoms in ribbon.")
    in_chain = c_property('atom_in_chain', npy_bool, read_only = True,
        doc = "Whether this atom belongs to a polymer. Read only.")
    is_ribose = c_property('atom_is_ribose', npy_bool, read_only = True,
        doc = "Whether this atom is part of an nucleic acid ribose moiety. Read only.")
    is_sidechain = c_property('atom_is_sidechain', npy_bool, read_only = True,
        doc = "Whether this atom is part of an amino/nucleic acid sidechain. Read only.")
    name = c_property('atom_name', string, doc = "Atom name. Maximum length 4 characters.")
    neighbors = c_property('atom_neighbors', cptr, "num_bonds", astype=_atoms, read_only=True,
        doc=":class:`.Atom`\\ s connnected to this atom directly by one bond. Read only.")
    num_bonds = c_property("atom_num_bonds", size_t, read_only=True,
        doc="Number of bonds connected to this atom. Read only.")
    occupancy = c_property('atom_occupancy', float32, doc = "Occupancy, floating point value.")
    radius = c_property('atom_radius', float32, doc="Radius of atom.")
    residue = c_property('atom_residue', cptr, astype = _residue, read_only = True,
        doc = ":class:`Residue` the atom belongs to.")
    selected = c_property('atom_selected', npy_bool, doc="Whether the atom is selected.")
    structure = c_property('atom_structure', cptr, astype=_atomic_structure, read_only=True,
        doc=":class:`.AtomicStructure` the atom belongs to")
    structure_category = c_property('atom_structure_category', string, read_only=True,
        doc = "Whether atom is ligand, ion, etc.")
    visible = c_property('atom_visible', npy_bool, read_only=True,
        doc="Whether atom is displayed and not hidden.")

    def set_alt_loc(self, loc, create):
        if isinstance(loc, str):
            loc = loc.encode('utf-8')
        f = c_function('atom_set_alt_loc', args=(ctypes.c_void_p, ctypes.c_char, ctypes.c_bool, ctypes.c_bool))
        f(self._c_pointer, loc, create, False)

    def has_alt_loc(self, loc):
        if isinstance(loc, str):
            loc = loc.encode('utf-8')
        #value_type = npy_bool
        #vtype = numpy_type_to_ctype[value_type]
        vtype = ctypes.c_uint8
        v = vtype()
        v_ref = ctypes.byref(v)
        f = c_array_function('atom_has_alt_loc', args=(byte,), ret=npy_bool, per_object=False)
        a_ref = ctypes.byref(self._c_pointer)
        f(a_ref, 1, loc, v_ref)
        return v.value

    def delete(self):
        '''Delete this Atom from it's Structure'''
        f = c_function('atom_delete', args = (ctypes.c_void_p, ctypes.c_size_t))
        c = f(self._c_pointer_ref, 1)

    def connects_to(self, atom):
        '''Whether this atom is directly bonded to a specified atom.'''
        f = c_function('atom_connects_to', args = (ctypes.c_void_p, ctypes.c_void_p),
               ret = ctypes.c_bool)
        c = f(self._c_pointer, atom._c_pointer)
        return c

    def is_backbone(self, bb_extent=BBE_MAX):
        '''Whether this Atom is considered backbone, given the 'extent' criteria.

        |  Possible 'extent' values are:
        BBE_MIN
            Only the atoms needed to connect the residue chain (and their hydrogens)
        BBE_MAX
            All non-sidechain atoms
        BBE_RIBBON
            The backbone atoms that a ribbon depiction hides
        '''
        f = c_function('atom_is_backbone', args = (ctypes.c_void_p, ctypes.c_int),
                ret = ctypes.c_bool)
        return f(self._c_pointer, bb_type)

    @property
    def scene_coord(self):
        '''
        Atom center coordinates in the global scene coordinate system.
        This accounts for the :class:`Drawing` positions for the hierarchy
        of models this atom belongs to.
        '''
        return self.structure.scene_position * self.coord

    def take_snapshot(self, session, flags):
        data = {'structure': self.structure,
                'ses_id': self.structure.session_atom_to_id(self._c_pointer)}
        return data

    @staticmethod
    def restore_snapshot(session, data):
        return object_map(data['structure'].session_id_to_atom(data['ses_id']), Atom)

# -----------------------------------------------------------------------------
#
class Bond:
    '''
    Bond connecting two atoms.

    To create a Bond use the :class:`.AtomicStructure` new_bond() method.
    '''
    def __init__(self, bond_pointer):
        set_c_pointer(self, bond_pointer)

    def __str__(self):
        a1, a2 = self.atoms
        bond_sep = " \N{Left Right Arrow} "
        if a1.residue == a2.residue:
            return str(a1) + bond_sep + a2.__str__(atom_only=True)
        if a1.structure == a2.structure:
            # tautology for bonds, but this func is conscripted by pseudobonds, so test...
            res_str = a2.residue.__str__(residue_only=True)
            atom_str = a2.__str__(atom_only=True)
            joiner = "" if res_str.startswith(":") else " "
            return str(a1) + bond_sep + res_str + joiner + atom_str
        return str(a1) + bond_sep + str(a2)

    atoms = c_property('bond_atoms', cptr, 2, astype = _atom_pair, read_only = True)
    '''Two-tuple of :py:class:`Atom` objects that are the bond end points.'''
    color = c_property('bond_color', uint8, 4)
    '''Color RGBA length 4 numpy uint8 array.'''
    display = c_property('bond_display', npy_bool)
    '''
    Whether to display the bond if both atoms are shown.
    Can be overriden by the hide attribute.
    '''
    halfbond = c_property('bond_halfbond', npy_bool)
    '''
    Whether to color the each half of the bond nearest an end atom to match that atom
    color, or use a single color and the bond color attribute.  Boolean value.
    '''
    radius = c_property('bond_radius', float32)
    '''Displayed cylinder radius for the bond.'''
    HIDE_RIBBON = 0x1
    '''Hide mask for backbone bonds in ribbon.'''
    hide = c_property('bond_hide', int32)
    '''Whether bond is hidden (overrides display).  Integer bitmask.'''
    shown = c_property('bond_shown', npy_bool, read_only = True)
    '''Whether bond is visible and both atoms are shown and at least one is not Sphere style. Read only.'''
    structure = c_property('bond_structure', cptr, astype = _atomic_structure, read_only = True)
    ''':class:`.AtomicStructure` the bond belongs to.'''
    visible = c_property('bond_visible', npy_bool, read_only = True)
    '''Whether bond is display and not hidden. Read only.'''
    length = c_property('bond_length', float32, read_only = True)
    '''Bond length. Read only.'''
    
    def other_atom(self, atom):
        '''Return the :class:`Atom` at the other end of this bond opposite
        the specified atom.'''
        f = c_function('bond_other_atom', args = (ctypes.c_void_p, ctypes.c_void_p), ret = ctypes.c_void_p)
        c = f(self._c_pointer, atom._c_pointer)
        return object_map(c, Atom)

    def take_snapshot(self, session, flags):
        data = {'structure': self.structure,
                'ses_id': self.structure.session_bond_to_id(self._c_pointer)}
        return data

    @staticmethod
    def restore_snapshot(session, data):
        return object_map(data['structure'].session_id_to_bond(data['ses_id']), Bond)

# -----------------------------------------------------------------------------
#
class Pseudobond:
    '''
    A Pseudobond is a graphical line between atoms for example depicting a distance
    or a gap in an amino acid chain, often shown as a dotted or dashed line.
    Pseudobonds can join atoms belonging to different :class:`.AtomicStructure`\\ s
    which is not possible with a :class:`Bond`\\ .

    To create a Pseudobond use the :class:`PseudobondGroup` new_pseudobond() method.
    '''
    def __init__(self, pbond_pointer):
        set_c_pointer(self, pbond_pointer)

    __str__ = Bond.__str__

    atoms = c_property('pseudobond_atoms', cptr, 2, astype = _atom_pair, read_only = True)
    '''Two-tuple of :py:class:`Atom` objects that are the bond end points.'''
    color = c_property('pseudobond_color', uint8, 4)
    '''Color RGBA length 4 numpy uint8 array.'''
    display = c_property('pseudobond_display', npy_bool)
    '''
    Whether to display the bond if both atoms are shown.
    Can be overriden by the hide attribute.
    '''
    group = c_property('pseudobond_group', cptr, astype = _pseudobond_group, read_only = True)
    ''':py:class:`.pbgroup.PseudobondGroup` that this pseudobond belongs to'''
    halfbond = c_property('pseudobond_halfbond', npy_bool)
    '''
    Whether to color the each half of the bond nearest an end atom to match that atom
    color, or use a single color and the bond color attribute.  Boolean value.
    '''
    radius = c_property('pseudobond_radius', float32)
    '''Displayed cylinder radius for the bond.'''
    shown = c_property('pseudobond_shown', npy_bool, read_only = True)
    '''Whether bond is visible and both atoms are shown. Read only.'''

    def delete(self):
        '''Delete this pseudobond from it's group'''
        f = c_function('pseudobond_delete', args = (ctypes.c_void_p, ctypes.c_size_t))
        c = f(self._c_pointer_ref, 1)

    @property
    def length(self):
        '''Distance between centers of two bond end point atoms.'''
        a1, a2 = self.atoms
        v = a1.scene_coord - a2.scene_coord
        from math import sqrt
        return sqrt((v*v).sum())
    
    def other_atom(self, atom):
        '''Return the :class:`Atom` at the other end of this bond opposite
        the specified atom.'''
        a1,a2 = self.atoms
        return a2 if atom is a1 else a1

    _ses_id = c_property('pseudobond_get_session_id', int32, read_only = True,
        doc="Used by session save/restore internals")

    def take_snapshot(self, session, flags):
        return [self.group, self._ses_id]

    @staticmethod
    def restore_snapshot(session, data):
        group, id = data
        f = c_function('pseudobond_group_resolve_session_id',
            args = [ctypes.c_void_p, ctypes.c_int], ret = ctypes.c_void_p)
        return object_map(f(group._c_pointer, id), Pseudobond)

# -----------------------------------------------------------------------------
#
class PseudobondGroupData:
    '''
    A group of pseudobonds typically used for one purpose such as display
    of distances or missing segments.  The category attribute names the group,
    for example "distances" or "missing segments".

    This base class of :class:`.PseudobondGroup` represents the C++ data while
    the derived class handles rendering the pseudobonds.

    To create a PseudobondGroup use the :class:`PseudobondManager` get_group() method.
    '''

    def __init__(self, pbg_pointer):
        set_c_pointer(self, pbg_pointer)

    category = c_property('pseudobond_group_category', string, read_only = True,
        doc = "Name of the pseudobond group.  Read only string.")
    num_pseudobonds = c_property('pseudobond_group_num_pseudobonds', size_t, read_only = True,
        doc = "Number of pseudobonds in group. Read only.")
    structure = c_property('pseudobond_group_structure', cptr, astype = _atomic_structure,
        read_only = True, doc ="Structure pseudobond group is owned by.  Returns None if called"
        "on a group managed by the global pseudobond manager")
    pseudobonds = c_property('pseudobond_group_pseudobonds', cptr, 'num_pseudobonds',
        astype = _pseudobonds, read_only = True,
        doc = "Group pseudobonds as a :class:`.Pseudobonds` collection. Read only.")

    def new_pseudobond(self, atom1, atom2):
        '''Create a new pseudobond between the specified :class:`Atom` objects.'''
        f = c_function('pseudobond_group_new_pseudobond',
                       args = (ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p),
                       ret = ctypes.c_void_p)
        pb = f(self._c_pointer, atom1._c_pointer, atom2._c_pointer)
        return object_map(pb, Pseudobond)

    # Graphics changed flags used by rendering code.  Private.
    _SHAPE_CHANGE = 0x1
    _COLOR_CHANGE = 0x2
    _SELECT_CHANGE = 0x4
    _RIBBON_CHANGE = 0x8
    _ALL_CHANGE = 0xf
    _graphics_changed = c_property('pseudobond_group_graphics_change', int32)


# -----------------------------------------------------------------------------
#
from ..state import State
class PseudobondManager(State):
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

    def delete_group(self, pbg):
        f = c_function('pseudobond_global_manager_delete_group',
                       args = (ctypes.c_void_p, ctypes.c_void_p), ret = None)
        f(self._c_pointer, pbg._c_pointer)

    def get_group(self, category, create = True):
        '''Get an existing :class:`.PseudobondGroup` or create a new one given a category name.'''
        f = c_function('pseudobond_global_manager_get_group',
                       args = (ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int),
                       ret = ctypes.c_void_p)
        pbg = f(self._c_pointer, category.encode('utf-8'), create)
        if not pbg:
            return None
        from .pbgroup import PseudobondGroup
        return object_map(pbg,
            lambda ptr, ses=self.session: PseudobondGroup(ptr, session=ses))

    def group_map(self):
        '''Returns a dict that maps from :class:`.PseudobondGroup` category to group'''
        f = c_function('pseudobond_global_manager_group_map',
                       args = (ctypes.c_void_p,),
                       ret = ctypes.py_object)
        ptr_map = f(self._c_pointer)
        obj_map = {}
        for cat, pbg_ptr in ptr_map.items():
            obj = object_map(pbg_ptr,
                lambda ptr, ses=self.session: PseudobondGroup(ptr, session=ses))
            obj_map[cat] = obj
        return obj_map

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
            obj_map[ses_id] = object_map(ptr, None)
        data = {'version': version,
                'mgr data':retvals,
                'structure mapping': obj_map}
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
        return pbm

    def reset_state(self, session):
        f = c_function('pseudobond_global_manager_clear', args = (ctypes.c_void_p,))
        f(self._c_pointer)

    def _ses_call(self, func_qual):
        f = c_function('pseudobond_global_manager_session_' + func_qual, args=(ctypes.c_void_p,))
        f(self._c_pointer)


# -----------------------------------------------------------------------------
#
class Residue:
    '''
    A group of atoms such as an amino acid or nucleic acid. Every atom in
    an :class:`.AtomicStructure` belongs to a residue, including solvent and ions.

    To create a Residue use the :class:`.AtomicStructure` new_residue() method.
    '''

    SS_HELIX = 0
    SS_SHEET = SS_STRAND = 1

    def __init__(self, residue_pointer):
        set_c_pointer(self, residue_pointer)

    def __str__(self, residue_only = False):
        from ..core_settings import settings
        cmd_style = settings.atomspec_contents == "command-line specifier"
        ic = self.insertion_code
        if cmd_style:
            res_str = ":" + str(self.number) + ic
        else:
            res_str = self.name + " " + str(self.number) + ic
        if residue_only:
            return res_str
        chain_str = '/' + self.chain_id
        from .structure import Structure
        if len([s for s in self.structure.session.models.list() if isinstance(s, Structure)]) > 1:
            struct_string = str(self.structure)
        else:
            struct_string = ""
        from ..core_settings import settings
        if cmd_style:
            return struct_string + chain_str + res_str
        return '%s%s %s' % (struct_string, chain_str, res_str)

    atoms = c_property('residue_atoms', cptr, 'num_atoms', astype = _atoms, read_only = True)
    ''':class:`.Atoms` collection containing all atoms of the residue.'''
    chain = c_property('residue_chain', cptr, astype = _chain, read_only = True)
    ''':class:`.Chain` that this residue belongs to, if any. Read only.'''
    chain_id = c_property('residue_chain_id', string, read_only = True)
    '''Protein Data Bank chain identifier. Limited to 4 characters. Read only string.'''
    @property
    def description(self):
        '''Description of residue (if available) from HETNAM/HETSYN records or equivalent'''
        return getattr(self.structure, '_hetnam_descriptions', {}).get(self.name, None)
    insertion_code = c_property('residue_insertion_code', string)
    '''Protein Data Bank residue insertion code. 1 character or empty string.'''
    is_helix = c_property('residue_is_helix', npy_bool, doc=
        "Whether this residue belongs to a protein alpha helix. Boolean value. "
        "If set to True, also sets is_sheet to False. "
        "Use set_secondary_structure() if this behavior is undesired.")
    is_sheet = c_property('residue_is_sheet', npy_bool, doc=
        "Whether this residue belongs to a protein beta sheet. Boolean value. "
        "If set to True, also sets is_helix to False. "
        "Use set_secondary_structure() if this behavior is undesired.")
    PT_NONE = 0
    '''Residue polymer type = none.'''
    PT_AMINO = 1
    '''Residue polymer type = amino acid.'''
    PT_NUCLEIC = 2
    '''Residue polymer type = nucleotide.'''
    polymer_type = c_property('residue_polymer_type', int32, read_only = True)
    '''Polymer type of residue. Integer value.'''
    name = c_property('residue_name', string, read_only = True)
    '''Residue name. Maximum length 4 characters. Read only.'''
    num_atoms = c_property('residue_num_atoms', size_t, read_only = True)
    '''Number of atoms belonging to the residue. Read only.'''
    number = c_property('residue_number', int32, read_only = True)
    '''Integer sequence position number as defined in the input data file. Read only.'''
    principal_atom = c_property('residue_principal_atom', cptr, astype = _atom_or_none, read_only=True)
    '''The 'chain trace' :class:`.Atom`\\ , if any.

    Normally returns the C4' from a nucleic acid since that is always present,
    but in the case of a P-only trace it returns the P.'''
    ribbon_display = c_property('residue_ribbon_display', npy_bool)
    '''Whether to display the residue as a ribbon/pipe/plank. Boolean value.'''
    ribbon_hide_backbone = c_property('residue_ribbon_hide_backbone', npy_bool)
    '''Whether a ribbon automatically hides the residue backbone atoms. Boolean value.'''
    ribbon_color = c_property('residue_ribbon_color', uint8, 4)
    '''Ribbon color RGBA length 4 numpy uint8 array.'''
    ribbon_style = c_property('residue_ribbon_style', int32)
    '''Whether the residue is displayed as a ribbon or a pipe/plank. Integer value.'''
    RIBBON = 0
    '''Ribbon style = ribbon.'''
    PIPE = 1
    '''Ribbon style = pipe/plank.'''
    ribbon_adjust = c_property('residue_ribbon_adjust', float32)
    '''Smoothness adjustment factor (no adjustment = 0 <= factor <= 1 = idealized).'''
    ss_id = c_property('residue_ss_id', int32)
    '''Secondary structure id number. Integer value.'''
    structure = c_property('residue_structure', cptr, astype = _atomic_structure, read_only = True)
    ''':class:`.AtomicStructure` that this residue belongs to. Read only.'''

    # TODO: Currently no C++ method to get Chain

    def add_atom(self, atom):
        '''Add the specified :class:`.Atom` to this residue.
        An atom can only belong to one residue, and all atoms
        must belong to a residue.'''
        f = c_function('residue_add_atom', args = (ctypes.c_void_p, ctypes.c_void_p))
        f(self._c_pointer, atom._c_pointer)

    def find_atom(self, atom_name):
        '''Return the atom with the given name, or None if not found.\n'''
        '''If multiple atoms in the residue have that name, an arbitrary one that matches will'''
        ''' be returned.'''
        f = c_function('residue_find_atom', args = (ctypes.c_void_p, ctypes.c_char_p),
            ret = ctypes.c_void_p)
        return _atom_or_none(f(self._c_pointer, atom_name.encode('utf-8')))

    def set_alt_loc(self, loc):
        if isinstance(loc, str):
            loc = loc.encode('utf-8')
        f = c_array_function('residue_set_alt_loc', args=(byte,), per_object=False)
        r_ref = ctypes.byref(self._c_pointer)
        f(r_ref, 1, loc)

    def set_secondary_structure(self, ss_type, value):
        '''Set helix/sheet to True/False
        Unlike is_helix/is_sheet attrs, this function only sets the value requested,
        it will not unset any other types as a side effect.
        'ss_type' should be one of Residue.SS_HELIX or RESIDUE.SS_SHEET'''
        if ss_type == Residue.SS_HELIX:
            f = c_array_function('residue_set_ss_helix', args=(npy_bool,), per_object=False)
        else:
            f = c_array_function('residue_set_ss_sheet', args=(npy_bool,), per_object=False)
        f(self._c_pointer_ref, 1, value)

    def take_snapshot(self, session, flags):
        data = {'structure': self.structure,
                'ses_id': self.structure.session_residue_to_id(self._c_pointer)}
        return data

    @staticmethod
    def restore_snapshot(session, data):
        return object_map(data['structure'].session_id_to_residue(data['ses_id']), Residue)

import atexit
# -----------------------------------------------------------------------------
#
class Sequence:
    '''
    A polymeric sequence.  Offers string-like interface.
    '''

    SS_HELIX = 'H'
    SS_OTHER = 'O'
    SS_STRAND = 'S'

    nucleic3to1 = c_function('sequence_nucleic3to1', args = (ctypes.c_char_p,), ret = byte)
    protein3to1 = c_function('sequence_protein3to1', args = (ctypes.c_char_p,), ret = byte)
    rname3to1 = c_function('sequence_rname3to1', args = (ctypes.c_char_p,), ret = byte)

    chimera_exiting = False

    def __init__(self, seq_pointer=None, *, name="sequence", characters=""):
        self.attrs = {} # miscellaneous attributes
        self.markups = {} # per-residue (strings or lists)
        self.numbering_start = None
        set_pyobj_f = c_function('sequence_set_pyobj', args = (ctypes.c_void_p, ctypes.py_object))
        if seq_pointer:
            set_c_pointer(self, seq_pointer)
            set_pyobj_f(self._c_pointer, self)
            return # name/characters already exists; don't set
        seq_pointer = c_function('sequence_new',
            args = (ctypes.c_char_p, ctypes.c_char_p), ret = ctypes.c_void_p)(
                name.encode('utf-8'), characters.encode('utf-8'))
        set_c_pointer(self, seq_pointer)
        set_pyobj_f(self._c_pointer, self)

    characters = c_property('sequence_characters', string, doc=
        "A string representing the contents of the sequence")
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
        set_pyobj_f = c_function('sequence_set_pyobj', args = (ctypes.c_void_p, ctypes.py_object))
        set_pyobj_f(self._c_pointer, None) # will destroy C++ object unless it's an active Chain

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
        return f(self._c_pointer, index)

    def __getitem__(self, key):
        return self.characters[key]

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
        seq.name = data['name']
        self.characters = data['characters']
        seq.attrs = data.get('attrs', {})
        seq.markups = data.get('markups', {})
        seq.numbering_start = data.get('numbering_start', None)

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
            'markups': self.markups }
        return data

    def ungapped(self):
        """String of sequence without gap characters"""
        f = c_function('sequence_ungapped', args = (ctypes.c_void_p,), ret = ctypes.py_object)
        return f(self._c_pointer)

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
        # description derived from PDB/mmCIF info and set by AtomicStructure constructor
        self.description = None

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

    residues = c_property('sseq_residues', cptr, 'num_residues', astype = _residues_or_nones,
        read_only = True, doc = "List containing the residues of this sequence in order. "
        "Residues with no structure will be None. Read only.")
    num_residues = c_property('sseq_num_residues', size_t, read_only = True)
    '''Number of residues belonging to this sequence, including those without structure. Read only.'''
    structure = c_property('sseq_structure', cptr, astype = _atomic_structure, read_only = True)
    ''':class:`.AtomicStructure` that this structure sequence comes from. Read only.'''

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
        return "%s (%s)$s" % (self.structure.name, self.structure, name_part)

    @property
    def has_protein(self):
        for r in self.residues:
            if r and Sequence.protein3to1(r.name.encode('utf8')) != 'X':
                return True
        return False

    @staticmethod
    def restore_snapshot(session, data):
        sseq = StructureSequence(chain_id=data['chain_id'], structure=data['structure'])
        Sequence.set_state_from_snapshot(sseq, session, data['Sequence'])
        sseq.description = data['description']
        self.bulk_set(data['residues'], sseq.characters)
        sseq.description = data.get('description', None)
        return sseq

    @staticmethod
    def restore_snapshot(session, data):
        chain = object_map(data['structure'].session_id_to_chain(data['ses_id']), Chain)
        chain.description = data.get('description', None)
        return chain

    def ss_type(self, loc, loc_is_ungapped=False):
        if not loc_is_ungapped:
            loc = self.gapped_to_ungapped(loc)
        if loc is None:
            return None
        r = self.residues[loc]
        if r is None:
            return None
        if r.is_helix:
            return self.SS_HELIX
        if r.is_sheet:
            return self.SS_STRAND
        return self.SS_OTHER

    def take_snapshot(self, session, flags):
        data = {
            'Sequence': Sequence.take_snapshot(self),
            'chain_id': self.chain_id,
            'description': self.description,
            'residues': self.residues,
            'structure': self.structure
        }
        return data

# -----------------------------------------------------------------------------
#
class Chain(StructureSeq):
    '''
    A single polymer chain such as a protein, DNA or RNA strand.
    A chain has a sequence associated with it.  A chain may have breaks.
    Chain objects are not always equivalent to Protein Databank chains.

    '''

    def extend(self, chars):
        # disallow extend
        raise AssertionError("extend() called on Chain object")

    @staticmethod
    def restore_snapshot(session, data):
        chain = object_map(data['structure'].session_id_to_chain(data['ses_id']), Chain)
        chain.description = data.get('description', None)
        return chain

    def take_snapshot(self, session, flags):
        data = {
            'description': self.description,
            'ses_id': self.structure.session_chain_to_id(self._c_pointer),
            'structure': self.structure
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
    def __init__(self, mol_pointer=None, *, logger=None):
        if mol_pointer is None:
            # Create a new graph
            from .structure import AtomicStructure
            new_func = 'atomic_structure_new' if isinstance(self, AtomicStructure) else 'structure_new'
            mol_pointer = c_function(new_func, args = (ctypes.py_object,), ret = ctypes.c_void_p)(logger)
        set_c_pointer(self, mol_pointer)

    def delete(self):
        '''Deletes the C++ data for this atomic structure.'''
        c_function('structure_delete', args = (ctypes.c_void_p,))(self._c_pointer)

    atoms = c_property('structure_atoms', cptr, 'num_atoms', astype = _atoms, read_only = True)
    ''':class:`.Atoms` collection containing all atoms of the structure.'''
    bonds = c_property('structure_bonds', cptr, 'num_bonds', astype = _bonds, read_only = True)
    ''':class:`.Bonds` collection containing all bonds of the structure.'''
    chains = c_property('structure_chains', cptr, 'num_chains', astype = _chains, read_only = True)
    ''':class:`.Chains` collection containing all chains of the structure.'''
    name = c_property('structure_name', string)
    '''Structure name, a string.'''
    num_atoms = c_property('structure_num_atoms', size_t, read_only = True)
    '''Number of atoms in structure. Read only.'''
    num_atoms_visible = c_property('structure_num_atoms_visible', size_t, read_only = True)
    '''Number of visible atoms in structure. Read only.'''
    num_bonds = c_property('structure_num_bonds', size_t, read_only = True)
    '''Number of bonds in structure. Read only.'''
    num_coord_sets = c_property('structure_num_coord_sets', size_t, read_only = True)
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
    ss_assigned = c_property('structure_ss_assigned', npy_bool, doc =
        "Has secondary structure been assigned, either by data in original structure file "
        "or by some algorithm (e.g. dssp command)")

    def _copy(self):
        f = c_function('structure_copy', args = (ctypes.c_void_p,), ret = ctypes.c_void_p)
        p = f(self._c_pointer)
        return p
        
    def new_atom(self, atom_name, element_name):
        '''Create a new :class:`.Atom` object. It must be added to a :class:`.Residue` object
        belonging to this structure before being used.'''
        f = c_function('structure_new_atom',
                       args = (ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p),
                       ret = ctypes.c_void_p)
        ap = f(self._c_pointer, atom_name.encode('utf-8'), element_name.encode('utf-8'))
        return object_map(ap, Atom)

    def new_bond(self, atom1, atom2):
        '''Create a new :class:`.Bond` joining two :class:`Atom` objects.'''
        f = c_function('structure_new_bond',
                       args = (ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p),
                       ret = ctypes.c_void_p)
        bp = f(self._c_pointer, atom1._c_pointer, atom2._c_pointer)
        return object_map(bp, Bond)

    def new_residue(self, residue_name, chain_id, pos, insert=' '):
        '''Create a new :class:`.Residue`.'''
        f = c_function('structure_new_residue',
                       args = (ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int, ctypes.c_char),
                       ret = ctypes.c_void_p)
        rp = f(self._c_pointer, residue_name.encode('utf-8'), chain_id.encode('utf-8'), pos, insert.encode('utf-8'))
        return object_map(rp, Residue)

    def polymers(self, consider_missing_structure = True, consider_chains_ids = True):
        '''Return a tuple of :class:`.Residues` objects each containing residues for one polymer.
        Arguments control whether a single polymer can span missing residues or differing chain identifiers.'''
        f = c_function('structure_polymers',
                       args = (ctypes.c_void_p, ctypes.c_int, ctypes.c_int),
                       ret = ctypes.py_object)
        resarrays = f(self._c_pointer, consider_missing_structure, consider_chains_ids)
        from .molarray import Residues
        return tuple(Residues(ra) for ra in resarrays)

    def pseudobond_group(self, name, create_type = "normal"):
        '''Get or create a :class:`.PseudobondGroup` belonging to this structure.'''
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
        from .pbgroup import PseudobondGroup
        return object_map(pbg, PseudobondGroup)

    def delete_pseudobond_group(self, pbg):
        f = c_function('structure_delete_pseudobond_group',
                       args = (ctypes.c_void_p, ctypes.c_void_p), ret = None)
        f(self._c_pointer, pbg._c_pointer)

    @classmethod
    def restore_snapshot(cls, session, data):
        g = StructureData(logger=session.logger)
        g.set_state_from_snapshot(session, data)
        return g

    def set_state_from_snapshot(self, session, data):
        '''Restore from session info'''
        self._ses_call("restore_setup")
        f = c_function('structure_session_restore',
                args = (ctypes.c_void_p, ctypes.c_int,
                        ctypes.py_object, ctypes.py_object, ctypes.py_object))
        f(self._c_pointer, data['version'], data['ints'], data['floats'], data['misc'])
        session.triggers.add_handler("end restore session", self._ses_restore_teardown)

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

    def take_snapshot(self, session, flags):
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
        from ..state import FinalizedState
        return FinalizedState(data)

    def _ses_call(self, func_qual):
        f = c_function('structure_session_' + func_qual, args=(ctypes.c_void_p,))
        f(self._c_pointer)

    def _ses_restore_teardown(self, *args):
        self._ses_call("restore_teardown")
        from ..triggerset import DEREGISTER
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
    _ALL_CHANGE = 0xf
    _graphics_changed = c_property('structure_graphics_change', int32)

class ChangeTracker:
    '''Per-session singleton change tracker keeps track of all
    atomic data changes'''

    def __init__(self):
        f = c_function('change_tracker_create', args = (), ret = ctypes.c_void_p)
        set_c_pointer(self, f())

    def add_modified(self, modded, reason):
        f = c_function('change_tracker_add_modified',
            args = (ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_char_p))
        from .molarray import Collection
        if isinstance(modded, Collection):
            class_num = self._class_to_int(modded.object_class)
            for ptr in modded.pointers:
                f(self._c_pointer, class_num, ptr, reason.encode('utf-8'))
        else:
            f(self._c_pointer, self._class_to_int(modded.__class__), modded._c_pointer,
                reason.encode('utf-8'))
    @property
    def changed(self):
        f = c_function('change_tracker_changed', args = (ctypes.c_void_p,), ret = npy_bool)
        return f(self._c_pointer)

    @property
    def changes(self):
        f = c_function('change_tracker_changes', args = (ctypes.c_void_p,),
            ret = ctypes.py_object)
        data = f(self._c_pointer)
        class Changes:
            def __init__(self, created, modified, reasons, total_deleted):
                self.created = created
                self.modified = modified
                self.reasons = reasons
                self.total_deleted = total_deleted
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

    def clear(self):
        f = c_function('change_tracker_clear', args = (ctypes.c_void_p,))
        f(self._c_pointer)

    def _class_to_int(self, klass):
        # has to tightly coordinate wih change_track_add_modified
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
        raise AssertionError("Unknown class for change tracking")

# -----------------------------------------------------------------------------
#
class Element:
    '''A chemical element having a name, number, mass, and other physical properties.'''
    def __init__(self, element_pointer):
        set_c_pointer(self, element_pointer)

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

    def get_element(name_or_number):
        '''Get the Element that corresponds to an atomic name or number'''
        if type(name_or_number) == type(1):
            f = c_function('element_number_get_element', args = (ctypes.c_int,), ret = ctypes.c_void_p)
        elif type(name_or_number) == type(""):
            f = c_function('element_name_get_element', args = (ctypes.c_char_p,), ret = ctypes.c_void_p)
        else:
            raise ValueError("'get_element' arg must be string or int")
        return _element(f(name_or_number))

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

    def delete(self):
        '''Deletes the C++ data for this atomic structure.'''
        c_function('rxsection_delete', args = (ctypes.c_void_p,))(self._c_pointer)

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
    # When a C++ object such as an Atom is deleted the pointer is removed
    # from the object map if it exists and the Python object has its _c_pointer
    # attribute deleted.
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
