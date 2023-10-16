# distutils: language=c++
# cython: language_level=3, boundscheck=False, auto_pickle=False
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


cimport cydecl
import collections
from tinyarray import array, zeros
from cython.operator import dereference
from ctypes import c_void_p, byref
cimport cython
from libc.stdint cimport uintptr_t

IF UNAME_SYSNAME == "Windows":
    ctypedef long long ptr_type
ELSE:
    ctypedef long ptr_type

cdef const char * _translate_struct_cat(cydecl.StructCat cat):
    if cat == cydecl.StructCat.Main:
        return "main"
    if cat == cydecl.StructCat.Solvent:
        return "solvent"
    if cat == cydecl.StructCat.Ligand:
        return "ligand"
    if cat == cydecl.StructCat.Ions:
        return "ions"
    if cat == cydecl.StructCat.Unassigned:
        return "other"
    raise ValueError("Unknown structure category")

cdef class CyAtom:
    '''Base class for Atom, and is present only for performance reasons.'''
    cdef cydecl.Atom *cpp_atom

    SPHERE_STYLE, BALL_STYLE, STICK_STYLE = range(3)
    HIDE_RIBBON = 0x1
    HIDE_ISOLDE = 0x2
    HIDE_NUCLEOTIDE = 0x4
    BBE_MIN, BBE_RIBBON, BBE_MAX = range(3)

    _idatm_tuple = collections.namedtuple('idatm', ['geometry', 'substituents', 'description'])
    _idatm_tuple.geometry.__doc__ = "arrangement of bonds; 0: no bonds; 1: one bond;" \
        " 2: linear; 3: planar; 4: tetrahedral"
    _idatm_tuple.substituents.__doc__ = "number of bond partners"
    _idatm_tuple.description.__doc__ = "text description of atom type"
    _non_const_map = cydecl.Atom.get_idatm_info_map()
    idatm_info_map = { idatm_type.decode():
        _idatm_tuple(info['geometry'], info['substituents'], info['description'].decode())
        for idatm_type, info in _non_const_map.items()
    }
    _alt_loc_suppress_count = 0

    def __cinit__(self, ptr_type ptr_val, *args, **kw):
        self.cpp_atom = <cydecl.Atom *>ptr_val

    def __init__(self, ptr_val):
        self._deleted = False

    @property
    def cpp_pointer(self):
        if self._deleted: raise RuntimeError("Atom already deleted")
        return int(<ptr_type>self.cpp_atom)
    @property
    def _c_pointer(self):
        return c_void_p(self.cpp_pointer)
    @property
    def _c_pointer_ref(self):
        return byref(self._c_pointer)

    def __hash__(self):
        return id(self)

    def __lt__(self, other):
        # for sorting (objects of the same type)
        if self.residue == other.residue:
            return self.name < other.name \
                if self.name != other.name else self.serial_number < other.serial_number
        return self.residue < other.residue

    def __str__(self):
        "Supported API.  Allow Atoms to be used directly in print() statements"
        return self.string()

    # properties...

    @property
    def alt_loc(self):
        "Supported API. Alternate location indicator"
        if self._deleted: raise RuntimeError("Atom already deleted")
        return chr(self.cpp_atom.alt_loc())

    @alt_loc.setter
    def alt_loc(self, loc):
        "For switching between existing alt locs; use 'set_alt_loc' method for creating alt locs"
        if len(loc) != 1:
            raise ValueError("Alt loc must be single character, not '%s'" % loc)
        if self._deleted: raise RuntimeError("Atom already deleted")
        self.cpp_atom.set_alt_loc(ord(loc[0]))

    @property
    def alt_locs(self):
        if self._deleted: raise RuntimeError("Atom already deleted")
        alt_locs = self.cpp_atom.alt_locs()
        return [chr(loc) for loc in alt_locs]

    @property
    def aniso_u(self):
        "Supported API. Anisotropic temperature factors, returns 3x3 array of float or None.  Read only."
        if self._deleted: raise RuntimeError("Atom already deleted")
        c_arr = self.cpp_atom.aniso_u()
        if c_arr:
            arr = dereference(c_arr)
            a00 = arr[0]
            a01 = arr[1]
            a02 = arr[2]
            a11 = arr[3]
            a12 = arr[4]
            a22 = arr[5]
            return array([[a00, a01, a02], [a01, a11, a12], [a02, a12, a22]])
        return None

    @property
    def aniso_u6(self):
        '''Get anisotropic temperature factors as a 6 element float array
           containing (u11, u22, u33, u12, u13, u23) [i.e. PDB/mmCIF order] or None.
        '''
        if self._deleted: raise RuntimeError("Atom already deleted")
        c_arr = self.cpp_atom.aniso_u()
        if c_arr:
            # The C++ layer holds the values in row major order, so...
            arr = dereference(c_arr)
            a00 = arr[0]
            a01 = arr[1]
            a02 = arr[2]
            a11 = arr[3]
            a12 = arr[4]
            a22 = arr[5]
            return array([a00, a11, a22, a01, a02, a12])
        return None

    @aniso_u6.setter
    def aniso_u6(self, u6):
        '''Set anisotropic temperature factors as a 6 element float array
           representing the unique elements of the symmetrix matrix
           containing (u11, u22, u33, u12, u13, u23). If 'u6' arg is None,
           then clear any aniso_u values."
        '''
        if self._deleted: raise RuntimeError("Atom already deleted")
        if u6 is None:
            self.cpp_atom.clear_aniso_u()
            return

        if len(u6) != 6:
            raise ValueError("aniso_u6 array isn't length 6")
        # Note C++ layer holds the values in row major order
        self.cpp_atom.set_aniso_u(u6[0], u6[3], u6[4], u6[1], u6[5], u6[2])

    @property
    def atomspec(self):
        return self.string(style="command")

    @property
    def bfactor(self):
        "Supported API. B-factor, floating point value."
        if self._deleted: raise RuntimeError("Atom already deleted")
        return self.cpp_atom.bfactor()

    @bfactor.setter
    def bfactor(self, bf):
        if self._deleted: raise RuntimeError("Atom already deleted")
        self.cpp_atom.set_bfactor(bf)

    @property
    def bonds(self):
        "Supported API. Bonds connected to this atom as a list of :py:class:`Bond` objects. Read only."
        # work around non-const-correct code by using temporary...
        if self._deleted: raise RuntimeError("Atom already deleted")
        bonds = self.cpp_atom.bonds()
        from . import Bond
        return [Bond.c_ptr_to_py_inst(<ptr_type>b) for b in bonds]

    @property
    def color(self):
        "Supported API. Color RGBA length 4 sequence/array. Values in range 0-255"
        if self._deleted: raise RuntimeError("Atom already deleted")
        color = self.cpp_atom.color()
        # colors frequently get sliced to lop off the alpha, and tinyarray
        # doesn't support slicing, so return a tuple
        return (color.r, color.g, color.b, color.a)

    @color.setter
    @cython.boundscheck(False)  # turn off bounds checking
    @cython.wraparound(False)  # turn off negative index wrapping
    def color(self, rgba):
        if len(rgba) != 4:
            raise ValueError("Atom.color = rgba: 'rgba' must be length 4")
        if self._deleted: raise RuntimeError("Atom already deleted")
        self.cpp_atom.set_color(rgba[0], rgba[1], rgba[2], rgba[3])

    @property
    def coord(self):
        '''Supported API. Coordinates from the current coordinate set (or alt loc) as a length 3
           sequence/array, float values.  See get_coord method for other coordsets / alt locs.
           See scene_coord for coordinates after rotations and translations.
        '''
        if self._deleted: raise RuntimeError("Atom already deleted")
        crd = self.cpp_atom.coord()
        return array((crd[0], crd[1], crd[2]))

    @property
    def pb_coord(self):
        '''Pseudobond coordinates.  If atom is not visible and is part of a residue
           displayed as a cartoon, return coordinates on the cartoon.  Otherwise,
           return the actual atomic coordinates.
        '''
        if not self.visible and self.residue.ribbon_display:
            c = self.ribbon_coord
            if c is not None:
                return c
        return self.coord

    @property
    def pb_scene_coord(self):
        return self.structure.scene_position * self.pb_coord

    @coord.setter
    @cython.boundscheck(False)  # turn off bounds checking
    @cython.wraparound(False)  # turn off negative index wrapping
    def coord(self, xyz):
        if len(xyz) != 3:
            raise ValueError("set_coord(xyz): 'xyz' must be length 3")
        if self._deleted: raise RuntimeError("Atom already deleted")
        self.cpp_atom.set_coord(cydecl.cycoord.Point(xyz[0], xyz[1], xyz[2]))

    @property
    def coord_index(self):
        "Supported API. Coordinate index of atom in coordinate set."
        if self._deleted: raise RuntimeError("Atom already deleted")
        return self.cpp_atom.coord_index()

    @coord_index.setter
    def coord_index(self, index):
        if self._deleted: raise RuntimeError("Atom already deleted")
        self.cpp_atom.set_coord_index(index)

    @property
    def default_radius(self):
        "Supported API. Default atom radius. Read only."
        if self._deleted: raise RuntimeError("Atom already deleted")
        return self.cpp_atom.default_radius()

    @property
    def deleted(self):
        "Supported API. Has the C++ side been deleted?"
        return self._deleted

    @property
    def display(self):
        "Supported API. Whether to display the atom. Boolean value."
        if self._deleted: raise RuntimeError("Atom already deleted")
        return self.cpp_atom.display()

    @display.setter
    def display(self, cydecl.bool disp):
        if self._deleted: raise RuntimeError("Atom already deleted")
        self.cpp_atom.set_display(disp)

    @property
    def display_radius(self):
        if self._deleted: raise RuntimeError("Atom already deleted")
        dm = self.draw_mode
        if dm == CyAtom.SPHERE_STYLE:
            return self.radius
        if dm == CyAtom.BALL_STYLE:
            return self.radius * self.structure.ball_scale
        if dm == CyAtom.STICK_STYLE:
            return self.cpp_atom.maximum_bond_radius(self.structure.bond_radius)
        raise ValueError("Unknown draw mode")

    @property
    def draw_mode(self):
        '''Supported API. Controls how the atom is depicted.\n\nPossible values:\n\n
        SPHERE_STYLE\n
            Use full atom radius\n\n
        BALL_STYLE\n
            Use reduced atom radius, but larger than bond radius\n\n
        STICK_STYLE\n
            Match bond radius
        '''
        if self._deleted: raise RuntimeError("Atom already deleted")
        return self.cpp_atom.draw_mode()

    @draw_mode.setter
    def draw_mode(self, int dm):
        if self._deleted: raise RuntimeError("Atom already deleted")
        self.cpp_atom.set_draw_mode(<cydecl.DrawMode>dm)

    @property
    def effective_coord(self):
        '''Return the atom's ribbon_coord if the residue is displayed as a ribbon
           and it has a ribbon coordinate, otherwise return the current coordinate.
        '''
        if self._deleted: raise RuntimeError("Atom already deleted")
        crd = self.cpp_atom.effective_coord()
        return array((crd[0], crd[1], crd[2]))

    @property
    def element(self):
        "Supported API. :class:`Element` corresponding to the atom's chemical element"
        if self._deleted: raise RuntimeError("Atom already deleted")
        return self.cpp_atom.element().py_instance(True)

    @element.setter
    def element(self, val):
        "Supported API. set atom's chemical element"
        if self._deleted: raise RuntimeError("Atom already deleted")
        cdef Element e
        if type(val) == Element:
            e = val
        elif type(val) in (int, str):
            e = Element.get_element(val)
        else:
            raise ValueError("Cannot set Element from %s" % repr(val))
        self.cpp_atom.set_element(dereference(e.cpp_element))

    @property
    def hide(self):
        '''Supported API. Whether atom is hidden (overrides display).
        Integer bitmask.\n\nPossible values:\n\n
        HIDE_RIBBON\n
            Hide mask for backbone atoms in ribbon.\n
        HIDE_ISOLDE\n
            Hide mask for backbone atoms for ISOLDE.\n
        HIDE_NUCLEOTIDE\n
            Hide mask for sidechain atoms in nucleotides.\n
        '''
        if self._deleted: raise RuntimeError("Atom already deleted")
        return self.cpp_atom.hide()

    @hide.setter
    def hide(self, int hide_bits):
        if self._deleted: raise RuntimeError("Atom already deleted")
        self.cpp_atom.set_hide(hide_bits)

    @property
    def idatm_type(self):
        '''Supported API. Atom's <a href="help:user/atomtypes.html">IDATM type</a>'''
        if self._deleted: raise RuntimeError("Atom already deleted")
        return self.cpp_atom.idatm_type().decode()

    @idatm_type.setter
    def idatm_type(self, idatm_type):
        string_type = "" if idatm_type is None else idatm_type
        if self._deleted: raise RuntimeError("Atom already deleted")
        self.cpp_atom.set_idatm_type(string_type.encode())

    @property
    def is_ribose(self):
        "Whether this atom is part of an nucleic acid ribose moiety. Read only."
        if self._deleted: raise RuntimeError("Atom already deleted")
        return self.cpp_atom.is_ribose()

    @property
    def is_side_connector(self):
        "Whether this atom is connects the side chain to the backbone, e.g. CA/ribose. Read only."
        if self._deleted: raise RuntimeError("Atom already deleted")
        return self.cpp_atom.is_side_connector()

    @property
    def is_side_chain(self):
        '''Whether this atom is part of an amino/nucleic acid sidechain.
           Includes atoms needed to connect to backbone (CA/ribose).
           is_side_only property excludes those. Read only.
        '''
        if self._deleted: raise RuntimeError("Atom already deleted")
        return self.cpp_atom.is_side_chain(False)

    @property
    def is_side_only(self):
        '''Whether this atom is part of an amino/nucleic acid sidechain.
           Does not include atoms needed to connect to backbone (CA/ribose).
           is_side_chain property includes those.  Read only.
        '''
        if self._deleted: raise RuntimeError("Atom already deleted")
        return self.cpp_atom.is_side_chain(True)

    @property
    def name(self):
        "Supported API. Atom name. Maximum length 4 characters."
        if self._deleted: raise RuntimeError("Atom already deleted")
        return self.cpp_atom.name().decode()

    @name.setter
    def name(self, new_name):
        if self._deleted: raise RuntimeError("Atom already deleted")
        self.cpp_atom.set_name(new_name.encode())

    @property
    def neighbors(self):
        "Supported API. :class:`.Atom`\\ s connected to this atom directly by one bond. Read only."
        # work around Cython not always generating const-correct code
        if self._deleted: raise RuntimeError("Atom already deleted")
        tmp = <cydecl.vector[cydecl.Atom*]>self.cpp_atom.neighbors()
        return [nb.py_instance(True) for nb in tmp]

    @property
    def num_alt_locs(self):
        "Number of alternate locations for this atom. Read only."
        if self._deleted: raise RuntimeError("Atom already deleted")
        return self.cpp_atom.alt_locs().size()

    @property
    def num_bonds(self):
        "Supported API. Number of bonds connected to this atom. Read only."
        if self._deleted: raise RuntimeError("Atom already deleted")
        return self.cpp_atom.bonds().size()

    @property
    def num_explicit_bonds(self):
        "Supported API. Number of bonds and missing-structure pseudobonds connected to this atom. Read only."
        if self._deleted: raise RuntimeError("Atom already deleted")
        return self.cpp_atom.num_explicit_bonds()

    @property
    def occupancy(self):
        "Supported API. Occupancy, floating point value."
        if self._deleted: raise RuntimeError("Atom already deleted")
        return self.cpp_atom.occupancy()

    @occupancy.setter
    def occupancy(self, new_occ):
        if self._deleted: raise RuntimeError("Atom already deleted")
        self.cpp_atom.set_occupancy(new_occ)

    @property
    def radius(self):
        "Supported API. Radius of atom."
        if self._deleted: raise RuntimeError("Atom already deleted")
        return self.cpp_atom.radius()

    @radius.setter
    def radius(self, new_rad):
        if self._deleted: raise RuntimeError("Atom already deleted")
        self.cpp_atom.set_radius(new_rad)

    @property
    def residue(self):
        "Supported API. :class:`Residue` the atom belongs to. Read only."
        if self._deleted: raise RuntimeError("Atom already deleted")
        return self.cpp_atom.residue().py_instance(True)

    @property
    def ribbon_coord(self):
        '''Atom ribbon coordinate in the structure coordinate system
           for displaying pseudobonds or tethers to the ribbon when
           the atom is hidden.  Value is None for non-backbone atoms.
        '''
        if self._deleted: raise RuntimeError("Atom already deleted")
        crd = self.cpp_atom.ribbon_coord()
        if crd:
            c = dereference(crd)
            return array((c[0], c[1], c[2]))
        return None

    @ribbon_coord.setter
    def ribbon_coord(self, xyz):
        "Set the ribbon coordinate.  Can be None."
        if self._deleted: raise RuntimeError("Atom already deleted")
        if xyz:
            self.cpp_atom.set_ribbon_coord(cydecl.cycoord.Point(xyz[0], xyz[1], xyz[2]))
        else:
            self.cpp_atom.clear_ribbon_coord()

    @property
    def scene_coord(self):
        '''Supported API. Atom center coordinates in the global scene coordinate system.
           This accounts for the :class:`Drawing` positions for the hierarchy
           of models this atom belongs to.
        '''
        if self._deleted: raise RuntimeError("Atom already deleted")
        crd = self.cpp_atom.scene_coord()
        return array((crd[0], crd[1], crd[2]))

    @scene_coord.setter
    def scene_coord(self, xyz):
        self.coord = self.structure.scene_position.inverse() * xyz

    @property
    def selected(self):
        "Supported API. Whether the atom is selected."
        if self._deleted: raise RuntimeError("Atom already deleted")
        return self.cpp_atom.selected()

    @selected.setter
    def selected(self, new_sel):
        if self._deleted: raise RuntimeError("Atom already deleted")
        self.cpp_atom.set_selected(new_sel)

    @property
    def serial_number(self):
        "Supported API. Atom serial number from input file."
        if self._deleted: raise RuntimeError("Atom already deleted")
        return self.cpp_atom.serial_number()

    @serial_number.setter
    def serial_number(self, new_sn):
        if self._deleted: raise RuntimeError("Atom already deleted")
        self.cpp_atom.set_serial_number(new_sn)

    @property
    def session(self):
        "Session that this Atom is in"
        return self.structure.session

    @property
    def structure(self):
        "Supported API. :class:`.AtomicStructure` the atom belongs to"
        if self._deleted: raise RuntimeError("Atom already deleted")
        return self.cpp_atom.structure().py_instance(True)

    @property
    def structure_category(self):
        "Supported API. Whether atom is ligand, ion, etc. Read only."
        if self._deleted: raise RuntimeError("Atom already deleted")
        return _translate_struct_cat(self.cpp_atom.structure_category()).decode()

    from contextlib import contextmanager
    @contextmanager
    def suppress_alt_loc_change_notifications(self):
        """Suppress alt loc change notifications while the code body runs.
           Restore the original alt loc of this atom when done."""
        orig_alt_loc = self.alt_loc
        if self._alt_loc_suppress_count == 0:
            self.structure.alt_loc_change_notify = False
        self._alt_loc_suppress_count += 1
        try:
            yield
        finally:
            self.alt_loc = orig_alt_loc
            self._alt_loc_suppress_count -= 1
            if self._alt_loc_suppress_count == 0:
                self.structure.alt_loc_change_notify = True

    @property
    def visible(self):
        "Supported API. Whether atom is displayed and not hidden."
        if self._deleted: raise RuntimeError("Atom already deleted")
        return self.cpp_atom.visible()

    # instance methods...

    def clear_hide_bits(self, bit_mask):
        '''Set the hide bits 'off' that are 'on' in "bitmask"
           and leave others unchanged. Opposite of set_hide_bits()
        '''
        if self._deleted: raise RuntimeError("Atom already deleted")
        self.cpp_atom.clear_hide_bits(bit_mask)

    def connects_to(self, CyAtom atom):
        "Supported API. Whether this atom is directly bonded to a specified atom."
        if self._deleted: raise RuntimeError("Atom already deleted")
        return self.cpp_atom.connects_to(atom.cpp_atom)

    def delete(self):
        "Supported API. Delete this Atom from it's Structure"
        if self._deleted: raise RuntimeError("Atom already deleted")
        self.cpp_atom.structure().delete_atom(self.cpp_atom)

    def delete_alt_loc(self, loc):
        ''''Raw' editing routine with very little consistency checking.
           Using Residue.delete_alt_loc() is recommended in most cases.
        '''
        if len(loc) != 1:
            raise ValueError("Alt loc must be single character, not '%s'" % loc)
        if self._deleted: raise RuntimeError("Atom already deleted")
        self.cpp_atom.delete_alt_loc(ord(loc[0]))

    def get_alt_loc_coord(self, loc):
        '''Supported API.  Like the 'coord' property, but uses the given alt loc
           (character) rather than the current alt loc.  Space character gets the
           non-alt-loc coord."
        '''
        if self._deleted: raise RuntimeError("Atom already deleted")
        if loc == ' ':
            return self.coord
        if self.has_alt_loc(loc):
            crd = self.cpp_atom.coord(ord(loc[0]))
            return array((crd[0], crd[1], crd[2]))
        raise ValueError("Atom %s has no alt loc %s" % (self, loc))

    def get_alt_loc_scene_coord(self, loc):
        '''Supported API.  Like the 'scene_coord' property, but uses the given alt loc
           (character) rather than the current alt loc. Space character gets the
           non-alt-loc scene coord.
        '''
        if self._deleted: raise RuntimeError("Atom already deleted")
        if loc == ' ':
            return self.scene_coord
        return self.structure.scene_position * self.get_alt_loc_coord(loc)

    def get_coordset_coord(self, cs_id):
        '''Supported API.  Like the 'coord' property, but uses the given coordset ID
           (integer) rather than the current coordset.
        '''
        if self._deleted: raise RuntimeError("Atom already deleted")
        cdef cydecl.CoordSet* cs = self.cpp_atom.structure().find_coord_set(cs_id)
        if not cs:
            raise ValueError("No such coordset ID: %d" % cs_id)
        crd = self.cpp_atom.coord(cs)
        return array((crd[0], crd[1], crd[2]))

    def get_coordset_scene_coord(self, cs_id):
        '''Supported API.  Like the 'scene_coord' property, but uses the given coordset ID
           (integer) rather than the current coordset.
        '''
        return self.structure.scene_position * self.get_coordset_coord(cs_id)

    def has_alt_loc(self, loc):
        "Supported API. Does this Atom have an alt loc with the given letter?"
        if len(loc) != 1:
            raise ValueError("Alt loc must be single character, not '%s'" % loc)
        if self._deleted: raise RuntimeError("Atom already deleted")
        return self.cpp_atom.has_alt_loc(ord(loc[0]))

    def is_backbone(self, bb_extent=CyAtom.BBE_MAX):
        '''Supported API. Whether this Atom is considered backbone, given the 'extent' criteria.n\n
        Possible 'extent' values are:\n\n
        BBE_MIN\n
            Only the atoms needed to connect the residue chain (and their hydrogens)
        BBE_MAX\n
            All non-sidechain atoms
        BBE_RIBBON\n
            The backbone atoms that a ribbon depiction hides
        '''
        if self._deleted: raise RuntimeError("Atom already deleted")
        return self.cpp_atom.is_backbone(<cydecl.BackboneExtent>bb_extent)

    def is_missing_heavy_template_neighbors(self, *, chain_start = False, chain_end = False,
            no_template_okay=False):
        if self._deleted: raise RuntimeError("Atom already deleted")
        return self.cpp_atom.is_missing_heavy_template_neighbors(chain_start, chain_end, no_template_okay)

    def rings(self, cross_residues=False, all_size_threshold=0):
        '''Return :class:`.Rings` collection of rings this Atom participates in.

        If 'cross_residues' is False, then rings that cross residue boundaries are not
        included.  If 'all_size_threshold' is zero, then return only minimal rings, of
        any size.  If it is greater than zero, then return all rings not larger than the
        given value.

        The returned rings are quite emphemeral, and shouldn't be cached or otherwise
        retained for long term use.  They may only live until the next call to rings()
        [from anywhere, including C++].
        '''
        # work around non-const-correct code by using temporary...
        if self._deleted: raise RuntimeError("Atom already deleted")
        ring_ptrs = self.cpp_atom.rings(cross_residues, all_size_threshold)
        from chimerax.atomic.molarray import Rings
        import numpy
        return Rings(numpy.array([<ptr_type>r for r in ring_ptrs], dtype=numpy.uintp))

    def set_alt_loc(self, loc, create):
        '''Normally used to create alt locs.
           The 'alt_loc' property is used to switch between existing alt locs.
        '''
        if self._deleted: raise RuntimeError("Atom already deleted")
        self.cpp_atom.set_alt_loc(ord(loc[0]), create, False)

    @cython.boundscheck(False)  # turn off bounds checking
    @cython.wraparound(False)  # turn off negative index wrapping
    def set_coord(self, xyz, int cs_id):
        cdef int size = xyz.shape[0]
        if size != 3:
            raise ValueError('setcoord(xyz, cs_id): "xyz" must by numpy array of dimension 1x3')
        if self._deleted: raise RuntimeError("Atom already deleted")
        cdef cydecl.CoordSet* cs = self.cpp_atom.structure().find_coord_set(cs_id)
        if not cs:
            raise ValueError("No such coordset ID: %d" % cs_id)
        if self._deleted: raise RuntimeError("Atom already deleted")
        self.cpp_atom.set_coord(cydecl.cycoord.Point(xyz[0], xyz[1], xyz[2]), cs)

    def set_hide_bits(self, bit_mask):
        '''Set the hide bits 'on' that are 'on' in "bitmask"
           and leave others unchanged. Opposite of clear_hide_bits()
        '''
        if self._deleted: raise RuntimeError("Atom already deleted")
        self.cpp_atom.set_hide_bits(bit_mask)

    def set_implicit_idatm_type(self, idatm_type):
        if self._deleted: raise RuntimeError("Atom already deleted")
        if idatm_type is None:
            raise ValueError("Cannot set implicit IDATM type to None")
        self.cpp_atom.set_implicit_idatm_type(idatm_type.encode())

    def side_atoms(self, CyAtom skip_atom=None, CyAtom cycle_atom=None):
        '''All the atoms connected to this atom on this side of 'skip_atom' (if given).
           Missing-structure pseudobonds are treated as connecting their atoms for the purpose of
           computing the connected atoms.  Connectivity will never trace through skip_atom, but if
           'cycle_atom' (which can be the same as skip_atom) is reached then a cycle/ring is assumed
           to exist and ValueError is thrown.
        '''
        if self._deleted: raise RuntimeError("Atom already deleted")
        sn_ptr = NULL if skip_atom is None else skip_atom.cpp_atom
        ca_ptr = NULL if cycle_atom is None else cycle_atom.cpp_atom
        # have to use a temporary to workaround the generated code otherwise taking the address
        # of a temporary variable (the return value)
        try:
            tmp = <cydecl.vector[cydecl.Atom*]>self.cpp_atom.side_atoms(<cydecl.Atom*>sn_ptr,
                <cydecl.Atom*>ca_ptr)
        except RuntimeError as e:
            # Cython raises RuntimeError for std::logic_error.
            # Raise ValueError instead to be consistent with molc.cpp
            raise ValueError(str(e))
        from chimerax.atomic import Atoms
        import numpy
        return Atoms(numpy.array([<ptr_type>r for r in tmp], dtype=numpy.uintp))

    def string(self, *, atom_only=False, style=None, relative_to=None, omit_structure=None, minimal=False):
        '''Supported API.  Get text representation of Atom
           (also used by __str__ for printing); if omit_structure is None, the the structure
           will be omitted if only one structure is open
        '''
        if style == None:
            from .settings import settings
            style = settings.atomspec_contents
        if relative_to:
            if self.residue == relative_to.residue:
                return self.string(atom_only=True, style=style)
            if self.structure == relative_to.structure:
                # tautology for bonds, but this func is conscripted by pseudobonds, so test...
                if style.startswith('serial'):
                    return self.string(atom_only=True, style=style)
                if self.residue.chain_id == relative_to.residue.chain_id:
                    chain_str = ""
                else:
                    from chimerax.atomic import Chain
                    chain_str = Chain.chain_id_to_atom_spec(self.residue.chain_id) + (
                        ' ' if style.startswith("simple") else "")
                res_str = "" if self.residue == relative_to.residue \
                    else self.residue.string(residue_only=True, style=style)
                atom_str = self.string(atom_only=True, style=style)
                joiner = "" if atom_str.startswith("@") else " "
                return chain_str + res_str + joiner + atom_str
        if style.startswith("simple"):
            atom_str = self.name
            if self.num_alt_locs > 0:
                atom_str += " (alt loc %s)" % self.alt_loc
        elif style.startswith("command"):
            # have to get fancy if the atom name isn't unique in the residue
            atoms = self.residue.atoms
            if len(atoms.filter(atoms.names == self.name)) > 1:
                atom_str = '@@serial_number=' + str(self.serial_number)
            elif self.name.endswith('-'):
                atom_str = '@@name="' + self.name + '"'
            else:
                atom_str = '@' + self.name
        else:
            atom_str = str(self.serial_number)
        if atom_only:
            return atom_str
        if minimal and self.structure.num_residues == 1:
            if omit_structure is None:
                from .structure import Structure
                omit_structure = len([s for s in self.structure.session.models.list()
                    if isinstance(s, Structure)]) == 1
            if omit_structure:
                return atom_str
            return self.structure.string(style=style) + (" " if style.startswith("simple") else "") + atom_str
        if not style.startswith('simple'):
            return '%s%s' % (self.residue.string(
                style=style, omit_structure=omit_structure, minimal=minimal), atom_str)
        return '%s %s' % (self.residue.string(style=style, omit_structure=omit_structure, minimal=minimal),
            atom_str)

    def use_default_radius(self):
        '''Supported API.  If an atom's radius has previously been explicitly set,
           this call will revert it to using the default radius
        '''
        if self._deleted: raise RuntimeError("Atom already deleted")
        self.cpp_atom.use_default_radius()

    # static methods...

    @staticmethod
    def c_ptr_to_existing_py_inst(ptr_type ptr_val):
        return (<cydecl.Atom *>ptr_val).py_instance(False)

    @staticmethod
    def c_ptr_to_py_inst(ptr_type ptr_val):
        return (<cydecl.Atom *>ptr_val).py_instance(True)

    @staticmethod
    def set_py_class(klass):
        cydecl.Atom.set_py_class(klass)

cdef class Element:
    '''A chemical element having a name, number, mass, and other physical properties.'''
    cdef cydecl.cyelem.Element *cpp_element

    NUM_SUPPORTED_ELEMENTS = cydecl.cyelem.Element.AS.NUM_SUPPORTED_ELEMENTS

    def __cinit__(self, ptr_type ptr_val):
        self.cpp_element = <cydecl.cyelem.Element *>ptr_val

    def __init__(self, ptr_val):
        if not isinstance(ptr_val, int) or ptr_val < 256:
            raise ValueError("Do not use Element constructor directly;"
                " use Element.get_element method to get an Element instance")

    # possibly long-term hack for interoperation with ctypes
    def __delattr__(self, name):
        if name == "_c_pointer" or name == "_c_pointer_ref":
            self._deleted = True
        else:
            super().__delattr__(name)

    def __eq__(self, val):
        if type(val) == Element:
            return val is self
        elif type(val) == int:
            return val == self.number
        elif type(val) == str:
            return val == self.name
        raise ValueError("Cannot compare Element to %s" % repr(val))

    def __hash__(self):
        return id(self)

    @property
    def cpp_pointer(self):
        return int(<ptr_type>self.cpp_element)
    @property
    def _c_pointer(self):
        return c_void_p(self.cpp_pointer)
    @property
    def _c_pointer_ref(self):
        return byref(self._c_pointer)

    @property
    def has_custom_attrs(self):
        return False

    @property
    def is_alkali_metal(self):
        "Supported API. Is atom an alkali metal?  Read only"
        return self.cpp_element.is_alkali_metal()

    @property
    def is_halogen(self):
        "Supported API. Is atom a halogen?  Read only"
        return self.cpp_element.is_halogen()

    @property
    def is_metal(self):
        "Supported API. Is atom a metal?  Read only"
        return self.cpp_element.is_metal()

    @property
    def is_noble_gas(self):
        "Supported API. Is atom a noble gas?  Read only"
        return self.cpp_element.is_noble_gas()

    @property
    def mass(self):
        '''Supported API. Atomic mass, taken from
           http://en.wikipedia.org/wiki/List_of_elements_by_atomic_weight.  Read only.
        '''
        return self.cpp_element.mass()

    @property
    def name(self):
        "Supported API. Atomic symbol.  Read only"
        return self.cpp_element.name().decode()

    names = { sym.decode() for sym in cydecl.cyelem.Element.names() }
    '''Set of known element names'''

    @property
    def number(self):
        "Supported API. Atomic number.  Read only"
        return self.cpp_element.number()

    @property
    def valence(self):
        "Supported API. Electronic valence number, for example 7 for chlorine. Read only"
        return self.cpp_element.valence()

    def __str__(self):
        # make printing easier
        return self.name

    @staticmethod
    cdef float _bond_length(ptr_type e1, ptr_type e2):
        return cydecl.cyelem.Element.bond_length(
            dereference(<cydecl.cyelem.Element*>e1), dereference(<cydecl.cyelem.Element*>e2))

    @staticmethod
    def bond_length(e1, e2):
        '''Supported API. Standard single-bond length between two elements.
           Arguments can be element instances, atomic numbers, or element names
        '''
        if not isinstance(e1, Element):
            e1 = Element.get_element(e1)
        if not isinstance(e2, Element):
            e2 = Element.get_element(e2)
        return Element._bond_length(e1.cpp_pointer, e2.cpp_pointer)

    @staticmethod
    cdef float _bond_radius(ptr_type e):
        return cydecl.cyelem.Element.bond_radius(dereference(<cydecl.cyelem.Element*>e))

    @staticmethod
    def bond_radius(e):
        '''Supported API. Standard single-bond 'radius'
           (the amount this element would contribute to bond length).
           Argument can be an element instance, atomic number, or element name
        '''
        if not isinstance(e, Element):
            e = Element.get_element(e)
        return Element._bond_radius(e.cpp_pointer)

    @staticmethod
    def c_ptr_to_existing_py_inst(ptr_type ptr_val):
        return (<cydecl.cyelem.Element *>ptr_val).py_instance(False)

    @staticmethod
    def c_ptr_to_py_inst(ptr_type ptr_val):
        return (<cydecl.cyelem.Element *>ptr_val).py_instance(True)

    @staticmethod
    cdef const cydecl.cyelem.Element* _string_to_cpp_element(const char* ident):
        return &cydecl.cyelem.Element.get_named_element(ident)

    @staticmethod
    cdef const cydecl.cyelem.Element* _int_to_cpp_element(int ident):
        return &cydecl.cyelem.Element.get_element(ident)

    @staticmethod
    def get_element(ident):
        "Supported API.  Given an atomic symbol or atomic number, return the corresponding Element instance."
        cdef const cydecl.cyelem.Element* ele_ptr
        if isinstance(ident, int):
            if ident < 0 or ident > cydecl.cyelem.Element.AS.NUM_SUPPORTED_ELEMENTS:
                raise ValueError("Cannot create element with atomic number %d" % ident)
            ele_ptr = Element._int_to_cpp_element(ident)
        else:
            ele_ptr = Element._string_to_cpp_element(ident.encode())
        return ele_ptr.py_instance(True)

cydecl.cyelem.Element.set_py_class(Element)

cdef class CyResidue:
    '''Base class for Residue, and is present only for performance reasons.'''
    cdef cydecl.Residue *cpp_res

    def __cinit__(self, ptr_type ptr_val, *args, **kw):
        self.cpp_res = <cydecl.Residue *>ptr_val

    def __init__(self, ptr_val):
        self._deleted = False

    @property
    def cpp_pointer(self):
        if self._deleted: raise RuntimeError("Residue already deleted")
        return int(<ptr_type>self.cpp_res)
    @property
    def _c_pointer(self):
        return c_void_p(self.cpp_pointer)
    @property
    def _c_pointer_ref(self):
        return byref(self._c_pointer)

    def __hash__(self):
        return id(self)

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


    # properties...

    @property
    def alt_loc(self):
        if self._deleted: raise RuntimeError("Residue already deleted")
        for a in self.atoms:
            if a.alt_loc != ' ':
                return a.alt_loc
        return ' '

    @property
    def alt_locs(self):
        if self._deleted: raise RuntimeError("Residue already deleted")
        alt_locs = set()
        for a in self.atoms:
            alt_locs.update(a.alt_locs)
        return alt_locs

    @property
    def atoms(self):
        "Supported API. :class:`.Atoms` collection containing all atoms of the residue."
        from .molarray import Atoms
        import numpy
        # work around non-const-correct code by using temporary...
        if self._deleted: raise RuntimeError("Residue already deleted")
        atoms = self.cpp_res.atoms()
        return Atoms(numpy.array([<ptr_type>a for a in atoms], dtype=numpy.uintp))

    @property
    def atomspec(self):
        if self._deleted: raise RuntimeError("Residue already deleted")
        return self.string(style="command")

    @property
    def chain(self):
        "Supported API. :class:`.Chain` that this residue belongs to, if any. Read only."
        if self._deleted: raise RuntimeError("Residue already deleted")
        chain_ptr = self.cpp_res.chain()
        if chain_ptr:
            from chimerax.atomic import Chain
            chain = Chain.c_ptr_to_existing_py_inst(<ptr_type>chain_ptr)
            if chain:
                return chain
            return Chain(<ptr_type>chain_ptr)
        return None

    @property
    def chain_id(self):
        "Supported API. PDB chain identifier."
        if self._deleted: raise RuntimeError("Residue already deleted")
        return self.cpp_res.chain_id().decode()

    @chain_id.setter
    def chain_id(self, new_chain_id):
        if self._deleted: raise RuntimeError("Residue already deleted")
        self.cpp_res.set_chain_id(new_chain_id.encode())

    chi_info = {
        'ARG': [("N", "CA", "CB", "CG"),
            ("CA", "CB", "CG", "CD"),
            ("CB", "CG", "CD", "NE"),
            ("CG", "CD", "NE", "CZ")],
        'LYS': [("N", "CA", "CB", "CG"),
            ("CA", "CB", "CG", "CD"),
            ("CB", "CG", "CD", "CE"),
            ("CG", "CD", "CE", "NZ")],
        'MET': [("N", "CA", "CB", "CG"),
            ("CA", "CB", "CG", "SD"),
            ("CB", "CG", "SD", "CE")],
        'GLU': [("N", "CA", "CB", "CG"),
            ("CA", "CB", "CG", "CD"),
            ("CB", "CG", "CD", "OE1")],
        'GLN': [("N", "CA", "CB", "CG"),
            ("CA", "CB", "CG", "CD"),
            ("CB", "CG", "CD", "OE1")],
        'ASP': [("N", "CA", "CB", "CG"),
            ("CA", "CB", "CG", "OD1")],
        'ASN': [("N", "CA", "CB", "CG"),
            ("CA", "CB", "CG", "OD1")],
        'ILE': [("N", "CA", "CB", "CG1"),
            ("CA", "CB", "CG1", "CD1")],
        'LEU': [("N", "CA", "CB", "CG"),
            ("CA", "CB", "CG", "CD1")],
        'HIS': [("N", "CA", "CB", "CG"),
            ("CA", "CB", "CG", "ND1")],
        'TRP': [("N", "CA", "CB", "CG"),
            ("CA", "CB", "CG", "CD1")],
        'TYR': [("N", "CA", "CB", "CG"),
            ("CA", "CB", "CG", "CD1")],
        'PHE': [("N", "CA", "CB", "CG"),
            ("CA", "CB", "CG", "CD1")],
        'PRO': [("N", "CA", "CB", "CG"),
            ("CA", "CB", "CG", "CD")],
        'THR': [("N", "CA", "CB", "OG1")],
        'VAL': [("N", "CA", "CB", "CG1")],
        'SER': [("N", "CA", "CB", "OG")],
        'CYS': [("N", "CA", "CB", "SG")],
    }

    chi_sym_info = set([('PHE', 2), ('TYR', 2), ('ASP', 2), ('GLU', 3)])

    @property
    def chi1(self):
        return self.get_chi(1, False)

    @chi1.setter
    def chi1(self, val):
        self.set_chi(1, val)

    @property
    def chi2(self):
        return self.get_chi(2, False)

    @chi2.setter
    def chi2(self, val):
        self.set_chi(2, val)

    @property
    def chi3(self):
        return self.get_chi(3, False)

    @chi3.setter
    def chi3(self, val):
        self.set_chi(3, val)

    @property
    def chi4(self):
        return self.get_chi(4, False)

    @chi4.setter
    def chi4(self, val):
        self.set_chi(4, val)

    @property
    def sym_chi1(self):
        return self.get_chi(1, True)

    @sym_chi1.setter
    def sym_chi1(self, val):
        self.set_chi(1, val)

    @property
    def sym_chi2(self):
        return self.get_chi(2, True)

    @sym_chi2.setter
    def sym_chi2(self, val):
        self.set_chi(2, val)

    @property
    def sym_chi3(self):
        return self.get_chi(3, True)

    @sym_chi3.setter
    def sym_chi3(self, val):
        self.set_chi(3, val)

    @property
    def sym_chi4(self):
        return self.get_chi(4, True)

    @sym_chi4.setter
    def sym_chi4(self, val):
        self.set_chi(4, val)

    @property
    def center(self):
        "Average of atom positions as a length 3 array, 64-bit float values."
        return sum([a.coord for a in self.atoms]) / self.num_atoms

    @property
    def deleted(self):
        "Supported API. Has the C++ side been deleted?"
        return self._deleted

    @property
    def description(self):
        '''Description of residue (if available) from HETNAM/HETSYN records or equivalent'''
        return getattr(self.structure, '_hetnam_descriptions', {}).get(self.name, None)

    @property
    def insertion_code(self):
        "Supported API. PDB residue insertion code. 1 character or empty string."
        if self._deleted: raise RuntimeError("Residue already deleted")
        code = chr(self.cpp_res.insertion_code())
        return "" if code == ' ' else code

    @insertion_code.setter
    def insertion_code(self, ic):
        if not ic:
            ic = ' '
        elif len(ic) != 1:
            raise ValueError("Insertion code must be single character, not '%s'" % ic)
        if self._deleted: raise RuntimeError("Residue already deleted")
        self.cpp_res.set_insertion_code(ord(ic[0]))

    @property
    def is_helix(self):
        "Supported API. Whether this residue belongs to a protein alpha helix. Boolean value. "
        if self._deleted: raise RuntimeError("Residue already deleted")
        return self.cpp_res.is_helix()

    @is_helix.setter
    def is_helix(self, val):
        if self._deleted: raise RuntimeError("Residue already deleted")
        self.cpp_res.set_is_helix(val)

    @property
    def is_strand(self):
        "Supported API. Whether this residue belongs to a protein beta sheet. Boolean value. "
        if self._deleted: raise RuntimeError("Residue already deleted")
        return self.cpp_res.is_strand()

    @is_strand.setter
    def is_strand(self, val):
        if self._deleted: raise RuntimeError("Residue already deleted")
        self.cpp_res.set_is_strand(val)

    @property
    def label_one_letter_code(self):
        """
        The code that Actions->Label->Residues uses, which can actually be just the residue name
        (i.e. more that one letter) for non-polymers
        """
        if self._deleted: raise RuntimeError("Residue already deleted")
        code = self.one_letter_code
        if code is None:
            code = self.name
        return code

    @property
    def label_specifier(self):
        "The specifier that Actions->Label->Residues uses, which never includes the model ID"
        if self._deleted: raise RuntimeError("Residue already deleted")
        return self.string(omit_structure=True, style="command")

    @property
    def mmcif_chain_id(self):
        "mmCIF chain identifier. Limited to 4 characters. Read only string."
        if self._deleted: raise RuntimeError("Residue already deleted")
        return self.cpp_res.mmcif_chain_id().decode()

    @property
    def name(self):
        "Supported API. Residue name. Maximum length 4 characters."
        if self._deleted: raise RuntimeError("Residue already deleted")
        return self.cpp_res.name().decode()

    @name.setter
    def name(self, new_name):
        if self._deleted: raise RuntimeError("Residue already deleted")
        self.cpp_res.set_name(new_name.encode())

    @property
    def neighbors(self):
        "Supported API. Residues directly bonded to this residue. Read only."
        from chimerax.atomic import concatenate
        return [r for r in concatenate(self.atoms.bonds.atoms).unique_residues if r != self]

    @property
    def num_atoms(self):
        "Supported API. Number of atoms belonging to the residue. Read only."
        if self._deleted: raise RuntimeError("Residue already deleted")
        return self.cpp_res.atoms().size()

    @property
    def number(self):
        "Supported API. Integer sequence position number from input data file."
        if self._deleted: raise RuntimeError("Residue already deleted")
        return self.cpp_res.number()

    @number.setter
    def number(self, num):
        if self._deleted: raise RuntimeError("Residue already deleted")
        self.cpp_res.set_number(num)

    @property
    def omega(self):
        '''Supported API. Get/set omega angle.  If not an amino acid (or missing needed backbone atoms),
           setting is a no-op and getting returns None.'''
        n = self.find_atom("N")
        if n is None:
            return None
        ca = self.find_atom("CA")
        if ca is None:
            return None
        for nb in n.neighbors:
            if nb.residue == self:
                continue
            if nb.name == "C":
                prev_c = nb
                break
        else:
            return None
        prev_ca = prev_c.residue.find_atom("CA")
        if prev_ca is None:
            return None
        from chimerax.geometry import dihedral
        return dihedral(prev_ca.coord, prev_c.coord, n.coord, ca.coord)

    @omega.setter
    def omega(self, val):
        self.set_omega(val)

    # Cython doesn't seem to allow **kw in a property setter despite always_allow_keywords=True, so...
    def set_omega(self, val, **kw):
        cur_omega = self.omega
        if cur_omega is None:
            return
        n = self.find_atom("N")
        for nb in n.neighbors:
            if nb.residue == self:
                continue
            if nb.name == "C":
                prev_c = nb
                break
        else:
            return
        try:
            i = prev_c.neighbors.index(n)
        except IndexError:
            return
        _set_angle(self.session, prev_c, prev_c.bonds[i], val, cur_omega, "omega", **kw)

    @property
    def one_letter_code(self):
        return self.get_one_letter_code()

    @property
    def phi(self):
        '''Supported API. Get/set phi angle.  If not an amino acid (or missing needed backbone atoms),
           setting is a no-op and getting returns None.'''
        n = self.find_atom("N")
        if n is None:
            return None
        ca = self.find_atom("CA")
        if ca is None:
            return None
        c = self.find_atom("C")
        if c is None:
            return None
        for nb in n.neighbors:
            if nb.residue == self:
                continue
            if nb.name == "C":
                prev_c = nb
                break
        else:
            return None
        from chimerax.geometry import dihedral
        return dihedral(prev_c.coord, n.coord, ca.coord, c.coord)

    @phi.setter
    def phi(self, val):
        self.set_phi(val)

    # Cython doesn't seem to allow **kw in a property setter despite always_allow_keywords=True, so...
    def set_phi(self, val, **kw):
        cur_phi = self.phi
        if cur_phi is None:
            return
        n = self.find_atom("N")
        ca = self.find_atom("CA")
        try:
            i = n.neighbors.index(ca)
        except IndexError:
            return
        _set_angle(self.session, n, n.bonds[i], val, cur_phi, "phi", **kw)

    @property
    def psi(self):
        '''Supported API. Get/set psi angle.  If not an amino acid (or missing needed backbone atoms),
           setting is a no-op and getting returns None.'''
        n = self.find_atom("N")
        if n is None:
            return None
        ca = self.find_atom("CA")
        if ca is None:
            return None
        c = self.find_atom("C")
        if c is None:
            return None
        for nb in c.neighbors:
            if nb.residue == self:
                continue
            if nb.name == "N":
                next_n = nb
                break
        else:
            return None
        from chimerax.geometry import dihedral
        return dihedral(n.coord, ca.coord, c.coord, next_n.coord)

    @psi.setter
    def psi(self, val):
        self.set_psi(val)

    # Cython doesn't seem to allow **kw in a property setter despite always_allow_keywords=True, so...
    def set_psi(self, val, **kw):
        cur_psi = self.psi
        if cur_psi is None:
            return
        ca = self.find_atom("CA")
        c = self.find_atom("C")
        try:
            i = ca.neighbors.index(c)
        except IndexError:
            return
        _set_angle(self.session, ca, ca.bonds[i], val, cur_psi, "psi", **kw)

    PT_NONE, PT_AMINO, PT_NUCLEIC = range(3)
    PT_PROTEIN = PT_AMINO
    @property
    def polymer_type(self):
        '''Supported API.  Polymer type of residue. Values are:
             * PT_NONE: not a polymeric residue
             * PT_AMINO: amino acid
             * PT_NUCLEIC: nucleotide

	   (Access as Residue.PT_XXX)
        '''
        if self._deleted: raise RuntimeError("Residue already deleted")
        return self.cpp_res.polymer_type()

    @property
    def principal_atom(self):
        '''The 'chain trace' :class:`.Atom`\\ , if any.
        Normally returns the C4' from a nucleic acid since that is always present,
        but in the case of a P-only trace it returns the P.'''
        if self._deleted: raise RuntimeError("Residue already deleted")
        princ_ptr = self.cpp_res.principal_atom()
        if princ_ptr:
            return princ_ptr.py_instance(True)
        return None

    @property
    def ribbon_adjust(self):
        "Smoothness adjustment factor (no adjustment = 0 <= factor <= 1 = idealized)."
        if self._deleted: raise RuntimeError("Residue already deleted")
        return self.cpp_res.ribbon_adjust()

    @ribbon_adjust.setter
    def ribbon_adjust(self, ra):
        if self._deleted: raise RuntimeError("Residue already deleted")
        self.cpp_res.set_ribbon_adjust(ra)

    @property
    def ribbon_display(self):
        "Whether to display the residue as a ribbon/pipe/plank. Boolean value."
        if self._deleted: raise RuntimeError("Residue already deleted")
        return self.cpp_res.ribbon_display()

    @ribbon_display.setter
    def ribbon_display(self, rd):
        if self._deleted: raise RuntimeError("Residue already deleted")
        self.cpp_res.set_ribbon_display(rd)

    @property
    def ribbon_hide_backbone(self):
        "Whether a ribbon automatically hides the residue backbone atoms. Boolean value."
        if self._deleted: raise RuntimeError("Residue already deleted")
        return self.cpp_res.ribbon_hide_backbone()

    @ribbon_hide_backbone.setter
    def ribbon_hide_backbone(self, rhb):
        if self._deleted: raise RuntimeError("Residue already deleted")
        self.cpp_res.set_ribbon_hide_backbone(rhb)

    @property
    def ribbon_color(self):
        "Ribbon color RGBA length 4 sequence/array. Values in range 0-255"
        if self._deleted: raise RuntimeError("Residue already deleted")
        color = self.cpp_res.ribbon_color()
        return (color.r, color.g, color.b, color.a)

    @ribbon_color.setter
    @cython.boundscheck(False)  # turn off bounds checking
    @cython.wraparound(False)  # turn off negative index wrapping
    def ribbon_color(self, rgba):
        if len(rgba) != 4:
            raise ValueError("Residue.ribbon_color = rgba: 'rgba' must be length 4")
        if self._deleted: raise RuntimeError("Residue already deleted")
        self.cpp_res.set_ribbon_color(rgba[0], rgba[1], rgba[2], rgba[3])

    @property
    def ring_display(self):
        "Whether to display the residue's rings as filled. Boolean value."
        if self._deleted: raise RuntimeError("Residue already deleted")
        return self.cpp_res.ring_display()

    @ring_display.setter
    def ring_display(self, rd):
        if self._deleted: raise RuntimeError("Residue already deleted")
        self.cpp_res.set_ring_display(rd)

    @property
    def ring_color(self):
        "Ring color RGBA length 4 sequence/array. Values in range 0-255"
        if self._deleted: raise RuntimeError("Residue already deleted")
        color = self.cpp_res.ring_color()
        return (color.r, color.g, color.b, color.a)

    @ring_color.setter
    @cython.boundscheck(False)  # turn off bounds checking
    @cython.wraparound(False)  # turn off negative index wrapping
    def ring_color(self, rgba):
        if len(rgba) != 4:
            raise ValueError("Residue.ring_color = rgba: 'rgba' must be length 4")
        if self._deleted: raise RuntimeError("Residue already deleted")
        self.cpp_res.set_ring_color(rgba[0], rgba[1], rgba[2], rgba[3])

    @property
    def selected(self):
        "Supported API. Whether any atom in the residue is selected."
        if self._deleted: raise RuntimeError("Residue already deleted")
        return self.cpp_res.selected()

    RN_AUTHOR, RN_CANONICAL, RN_UNIPROT = range(3)
    def set_number(self, numbering, num):
        if self._deleted: raise RuntimeError("Residue already deleted")
        self.cpp_res.set_number(numbering, num)

    @property
    def standard_aa_name(self):
        '''If this is a standard amino acid or modified amino acid, return the 3-letter
        name of the corresponding standard amino acid.  Otherwise return None.  The
        ability to determine the standard name of a modified amino acid may depend on
        the presence of MODRES records or their equivalent in the original input.'''
        return self.__class__.get_standard_aa_name(self.name)

    @property
    def thin_rings(self):
        "Whether to display the residue's rings as filled. Boolean value."
        if self._deleted: raise RuntimeError("Residue already deleted")
        return self.cpp_res.thin_rings()

    @thin_rings.setter
    def thin_rings(self, rd):
        if self._deleted: raise RuntimeError("Residue already deleted")
        self.cpp_res.set_thin_rings(rd)

    @property
    def session(self):
        return self.structure.session

    @property
    def ss_id(self):
        "Secondary structure id number. Integer value."
        if self._deleted: raise RuntimeError("Residue already deleted")
        return self.cpp_res.ss_id()

    @ss_id.setter
    def ss_id(self, new_ss_id):
        if self._deleted: raise RuntimeError("Residue already deleted")
        self.cpp_res.set_ss_id(new_ss_id)

    SS_COIL, SS_HELIX, SS_STRAND = range(3)
    SS_SHEET = SS_STRAND
    @property
    def ss_type(self):
        '''Supported API. Secondary structure type of residue. Integer value.
           One of Residue.SS_COIL, Residue.SS_HELIX, Residue.SS_SHEET (a.k.a. SS_STRAND)
        '''
        if self._deleted: raise RuntimeError("Residue already deleted")
        return self.cpp_res.ss_type()

    @ss_type.setter
    def ss_type(self, st):
        if self._deleted: raise RuntimeError("Residue already deleted")
        self.cpp_res.set_ss_type(st)

    @property
    def structure(self):
        "Supported API. ':class:`.AtomicStructure` that this residue belongs to. Read only."
        if self._deleted: raise RuntimeError("Residue already deleted")
        return self.cpp_res.structure().py_instance(True)

    water_res_names = set(["HOH", "WAT", "H2O", "D2O", "TIP3"])


    # instance methods...

    def add_atom(self, CyAtom atom):
        '''Supported API. Add the specified :class:`.Atom` to this residue.
        An atom can only belong to one residue, and all atoms
        must belong to a residue.'''
        if self._deleted: raise RuntimeError("Residue already deleted")
        self.cpp_res.add_atom(<cydecl.Atom*>atom.cpp_atom)

    def bonds_between(self, CyResidue other_res):
        "Supported API. Return the bonds between this residue and other_res as a Bonds collection."
        from .molarray import Bonds
        import numpy
        # work around non-const-correct code by using temporary...
        if self._deleted: raise RuntimeError("Residue already deleted")
        between = self.cpp_res.bonds_between(<cydecl.Residue*>other_res.cpp_res)
        return Bonds(numpy.array([<ptr_type>b for b in between], dtype=numpy.uintp))

    def clean_alt_locs(self):
        "Change the current alt locs in this residue to 'regular' locations and delete all alt locs"
        if self._deleted: raise RuntimeError("Residue already deleted")
        self.cpp_res.clean_alt_locs()

    def connects_to(self, CyResidue other_res):
        '''Supported API. Return True if this residue is connected by at least one bond
           (not pseudobond) to other_res
        '''
        if self._deleted: raise RuntimeError("Residue already deleted")
        return self.cpp_res.connects_to(<cydecl.Residue*>other_res.cpp_res)

    def delete(self):
        "Supported API. Delete this Residue from it's Structure"
        if self._deleted: raise RuntimeError("Residue already deleted")
        self.cpp_res.structure().delete_atoms(self.cpp_res.atoms())

    def delete_alt_loc(self, loc):
        '''Deletes the specified alt loc in this residue and possibly other residues
           if their alt locs are 'connected'.  If deleting this residue's current alt
           loc, the best remaining one will become current.  For simply deleting all
           alt locs in the structure except the current ones (and changing those to
           non-alt locs) use Structure.delete_alt_locs().
        '''
        if len(loc) != 1:
            raise ValueError("Alt loc must be single character, not '%s'" % loc)
        if self._deleted: raise RuntimeError("Atom already deleted")
        self.cpp_res.delete_alt_loc(ord(loc[0]))

    def find_atom(self, atom_name):
        '''Supported API. Return the atom with the given name, or None if not found.
           If multiple atoms in the residue have that name, an arbitrary one that matches will
           be returned.'''
        if self._deleted: raise RuntimeError("Residue already deleted")
        fa_ptr = self.cpp_res.find_atom(atom_name.encode())
        if fa_ptr:
            return fa_ptr.py_instance(True)
        return None

    def get_chi(self, chi_num, account_for_symmetry):
        # Don't need to explicitly check that the standard name is not None,
        # since sending None will return None -- just the same as GLX or ALA will
        if chi_num < 1 or chi_num > 4:
            raise ValueError("Chi number not in the range 1-4")
        if self._deleted: raise RuntimeError("Residue already deleted")
        std_name = self.standard_aa_name
        chi_atoms = self.get_chi_atoms(std_name, chi_num)
        if chi_atoms is None:
            return None
        from chimerax.geometry import dihedral
        chi = dihedral(*[a.coord for a in chi_atoms])
        if account_for_symmetry:
            if (std_name, chi_num) in self.chi_sym_info:
                while chi > 90.0:
                    chi -= 180.0
                while chi <= -90.0:
                    chi += 180.0
        return chi

    def get_chi_atoms(self, std_type, chi_num):
        if self._deleted: raise RuntimeError("Residue already deleted")
        try:
            chi_atom_names = self.chi_info[std_type][chi_num-1]
        except (KeyError, IndexError):
            return None
        chi_atoms = []
        for name in chi_atom_names:
            a = self.find_atom(name)
            if a:
                chi_atoms.append(a)
            else:
                return None
        return chi_atoms

    def get_one_letter_code(self, *, non_polymeric_returns=None):
        """
        In this context, 'non_polymeric' means residues that are incapable of being in a polymer
        and therefore a singleton amino or nucleic acid is not 'non_polymeric' despite not being in
        an actual polymer.
        """
        if self._deleted: raise RuntimeError("Residue already deleted")
        from chimerax.atomic import Sequence
        code = Sequence.rname3to1(self.name)
        if code == 'X' and self.polymer_type == self.PT_NONE:
            return non_polymeric_returns
        return code

    def is_missing_heavy_template_atoms(self, *, no_template_okay=False):
        if self._deleted: raise RuntimeError("Residue already deleted")
        return self.cpp_res.is_missing_heavy_template_atoms(no_template_okay)

    # Cython kind of has trouble with a C++ class variable that is a map of maps, and where the key
    # type of the nested map is a varidic template; so ideal_chirality is exposed via ctypes instead

    def set_alt_loc(self, loc):
        "Set the appropriate atoms in the residue to the given (existing) alt loc"
        if not loc:
            loc = ' '
        if self._deleted: raise RuntimeError("Residue already deleted")
        self.cpp_res.set_alt_loc(ord(loc[0]))

    def set_chi(self, chi_num, val):
        cur_chi = self.get_chi(chi_num, False)
        if cur_chi is None:
            return
        a1, a2, a3, a4 = self.get_chi_atoms(self.standard_aa_name, chi_num)
        try:
            i = a3.neighbors.index(a2)
        except IndexError:
            return
        _set_angle(self.session, a3, a3.bonds[i], val, cur_chi, "chi%s" % chi_num)

    def remove_atom(self, CyAtom atom):
        "Supported API.  Remove the atom from this residue."
        self.cpp_res.remove_atom(atom.cpp_atom)

    def string(self, *, residue_only=False, omit_structure=None, style=None, minimal=False, omit_chain=None):
        '''Supported API.  Get text representation of Residue
           If 'omit_structure' is None, the structure will be omitted only if exactly one structure is open
        '''
        if style == None:
            from .settings import settings
            style = settings.atomspec_contents
        ic = self.insertion_code
        if style.startswith("simple"):
            res_str = self.name + " " + str(self.number) + ic
        else:
            res_str = ":" + str(self.number) + ic
        if residue_only:
            return res_str
        if omit_chain is None:
            if minimal:
                omit_chain = len(set(self.structure.residues.chain_ids)) ==  1
            else:
                omit_chain = False
        if omit_chain:
            chain_str = ""
        else:
            from chimerax.atomic import Chain
            chain_str = Chain.chain_id_to_atom_spec(self.chain_id)
        if omit_chain and omit_structure:
            return res_str
        if omit_structure is None:
            from .structure import Structure
            omit_structure = len([s for s in self.structure.session.models.list()
                if isinstance(s, Structure)]) == 1
        if omit_structure:
            format_string = "%s%s" if style.startswith("command") else "%s %s"
            return format_string % (chain_str, res_str)
        if omit_structure:
            struct_string = ""
        else:
            struct_string = self.structure.string(style=style)
            if style.startswith("serial"):
                struct_string += " "
        if style.startswith("simple"):
            return '%s%s %s' % (struct_string, chain_str, res_str)
        if style.startswith("command"):
            return struct_string + chain_str + res_str
        return struct_string

    # static methods...

    @staticmethod
    def c_ptr_to_existing_py_inst(ptr_type ptr_val):
        return (<cydecl.Residue *>ptr_val).py_instance(False)

    @staticmethod
    def c_ptr_to_py_inst(ptr_type ptr_val):
        return (<cydecl.Residue *>ptr_val).py_instance(True)

    @staticmethod
    def set_py_class(klass):
        cydecl.Residue.set_py_class(klass)

    @staticmethod
    def set_templates_dir(tmpl_dir):
        cydecl.Residue.set_templates_dir(tmpl_dir.encode())

    @staticmethod
    def set_user_templates_dir(tmpl_dir):
        cydecl.Residue.set_user_templates_dir(tmpl_dir.encode())

    @staticmethod
    def get_standard_aa_name(res_name):
        '''If 'res_name' is a standard amino acid or modified amino acid 3-letter name, return
        the 3-letter name of the corresponding standard amino acid.  Otherwise return None.
        The ability to determine the standard name of a modified amino acid may depend on
        the presence of MODRES records or their equivalent in the original input.'''
        from chimerax.atomic import Sequence
        try:
            return Sequence.protein1to3[Sequence.protein3to1(res_name)]
        except KeyError:
            return None

def _set_angle(session, torsion_atom2, bond, new_angle, cur_angle, attr_name, **kw):
    br = session.bond_rotations.new_rotation(bond, **kw)
    br.angle += new_angle - cur_angle
    res = bond.atoms[0].residue
    res.structure.change_tracker.add_modified(res, attr_name + " changed")
    session.bond_rotations.delete_rotation(br)
