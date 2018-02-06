# distutils: language = c++
#cython: language_level=3, boundscheck=False
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


cimport cydecl
import collections
from numpy import array, uint8, empty
cimport numpy as np
from cython.operator import dereference
from sys import getrefcount
from ctypes import c_void_p, byref
cimport cython

cdef class CyAtom:
    cdef cydecl.Atom *cpp_atom
    cdef cydecl.bool _deleted

    SPHERE_STYLE, BALL_STYLE, STICK_STYLE = range(3)
    HIDE_RIBBON = 0x1
    HIDE_ISOLDE = 0x2
    HIDE_NUCLEOTIDE = 0x4
    BBE_MIN, BBE_RIBBON, BBE_MAX = range(3)

    idatm_tuple = collections.namedtuple('idatm', ['geometry', 'substituents', 'description'])
    idatm_tuple.geometry.__doc__ = "arrangement of bonds; 0: no bonds; 1: one bond;" \
        " 2: linear; 3: planar; 4: tetrahedral"
    idatm_tuple.substituents.__doc__ = "number of bond partners"
    idatm_tuple.description.__doc__ = "text description of atom type"
    _non_const_map = cydecl.Atom.get_idatm_info_map()
    idatm_info_map = { idatm_type.decode():
        idatm_tuple(info['geometry'], info['substituents'], info['description'].decode())
        for idatm_type, info in _non_const_map.items()
    }

    def __cinit__(self, long ptr_val):
        self.cpp_atom = <cydecl.Atom *>ptr_val
        self._deleted = False

    def __init__(self, ptr_val):
        self._coord = None

    def __delattr__(self, name):
        if name == "_c_pointer" or name == "_c_pointer_ref":
            self._deleted = True
        else:
            super().__delattr__(name)

    # possibly long-term hack so that Residue.add_atom can work, as well as creating Collections
    @property
    def cpp_pointer(self):
        return int(<long>self.cpp_atom)
    @property
    def _c_pointer(self):
        return c_void_p(self.cpp_pointer)
    @property
    def _c_pointer_ref(self):
        return byref(self._c_pointer)

    #TODO: setters
    @property
    def alt_loc(self):
        return chr(self.cpp_atom.alt_loc())

    @property
    def alt_locs(self):
        alt_locs = self.cpp_atom.alt_locs()
        return [chr(al) for al in alt_locs]

    @property
    def bfactor(self):
        return self.cpp_atom.bfactor()

    @property
    def bonds(self):
        # work around non-const-correct code by using temporary...
        bonds = self.cpp_atom.bonds()
        from . import Bond
        return [Bond.c_ptr_to_py_inst(<long>b) for b in bonds]

    @property
    def color(self):
        color = self.cpp_atom.color()
        return array([color.r, color.g, color.b, color.a], dtype=uint8)

    @color.setter
    @cython.boundscheck(False)  # turn off bounds checking
    @cython.wraparound(False)  # turn off negative index wrapping
    def color(self, np.ndarray[np.uint8_t, ndim=1] rgba):
        if rgba.shape[0] != 4:
            raise ValueError("set_color(rgba): 'rgba' must be 1x4 numpy uint8 array")
        self.cpp_atom.set_color(rgba[0], rgba[1], rgba[2], rgba[3])

    @property
    def coord(self):
        if self._coord is None or getrefcount(self._coord) > 2:
            self._coord = empty(3, dtype='d')
        crd = self.cpp_atom.coord()
        self._coord[:] = (crd[0], crd[1], crd[2])
        return self._coord

    @property
    def deleted(self):
        return self._deleted

    @property
    def display(self):
        return self.cpp_atom.display()

    @display.setter
    def display(self, cydecl.bool disp):
        self.cpp_atom.set_display(disp)

    @property
    def draw_mode(self):
        return self.cpp_atom.draw_mode()

    @draw_mode.setter
    def draw_mode(self, int dm):
        self.cpp_atom.set_draw_mode(<cydecl.DrawMode>dm)

    @property
    def element(self):
        from . import Element
        return Element.c_ptr_to_py_inst(<long>&self.cpp_atom.element())

    @property
    def hide(self):
        return self.cpp_atom.hide()

    @hide.setter
    def hide(self, int hide_bits):
        self.cpp_atom.set_hide(hide_bits)

    @property
    def idatm_type(self):
        return self.cpp_atom.idatm_type().decode()

    @idatm_type.setter
    def idatm_type(self, idatm_type):
        string_type = "" if idatm_type is None else idatm_type
        self.cpp_atom.set_idatm_type(string_type.encode())

    @property
    def name(self):
        return self.cpp_atom.name().decode()

    @property
    def neighbors(self):
        # work around Cython not always generating const-correct code
        tmp = <cydecl.vector[cydecl.Atom*]>self.cpp_atom.neighbors()
        return [nb.py_instance(True) for nb in tmp]

    @property
    def num_bonds(self):
        return self.cpp_atom.bonds().size()

    @property
    def radius(self):
        return self.cpp_atom.radius()

    @property
    def residue(self):
        from . import Residue
        return Residue.c_ptr_to_py_inst(<long>self.cpp_atom.residue())

    @property
    def scene_coord(self):
        raise RuntimeError("scene_coord not Cythonized yet")

    #TODO: make 'create' a keyword that defaults to False
    def set_alt_loc(self, loc, create):
        self.cpp_atom.set_alt_loc(ord(loc[0]), create, False)

    @cython.boundscheck(False)  # turn off bounds checking
    @cython.wraparound(False)  # turn off negative index wrapping
    def set_coord(self, np.ndarray[np.float64_t, ndim=1] xyz, int cs_id):
        cdef int size = xyz.shape[0]
        if size != 3:
            raise ValueError('setcoord(xyz, cs_id): "xyz" must by numpy array of dimension 1x3')
        cdef cydecl.CoordSet* cs = self.cpp_atom.structure().find_coord_set(cs_id)
        if not cs:
            raise ValueError("No such coordset ID: %d" % cs_id)
        self.cpp_atom.set_coord(cydecl.Point(xyz[0], xyz[1], xyz[2]), cs)

    @property
    def structure(self):
        from . import Structure
        return Structure.c_ptr_to_py_inst(<long>self.cpp_atom.structure())


    @staticmethod
    def c_ptr_to_existing_py_inst(long ptr_val):
        return (<cydecl.Atom *>ptr_val).py_instance(False)

    @staticmethod
    def c_ptr_to_py_inst(long ptr_val):
        return (<cydecl.Atom *>ptr_val).py_instance(True)

    @staticmethod
    def set_py_class(klass):
        cydecl.Atom.set_py_class(klass)

'''
cdef class Element:
    cdef cydecl.Element *cpp_element

    NUM_SUPPORTED_ELEMENTS = cydecl.Element.AS.NUM_SUPPORTED_ELEMENTS

    def __cinit__(self, long ptr_val):
        self.cpp_element = <cydecl.Element *>ptr_val

    @property
    def is_halogen(self):
        return self.cpp_element.is_halogen()

    @property
    def is_metal(self):
        return self.cpp_element.is_metal()

    @property
    def name(self):
        return self.cpp_element.name().decode()

    @property
    def number(self):
        return self.cpp_element.number()

    @staticmethod
    cdef float _bond_length(long e1, long e2):
        return cydecl.Element.bond_length(
            dereference(<cydecl.Element*>e1), dereference(<cydecl.Element*>e2))

    @staticmethod
    def bond_length(e1, e2):
        if not isinstance(e1, Element):
            e1 = Element.get_element(e1)
        if not isinstance(e2, Element):
            e2 = Element.get_element(e2)
        return Element._bond_length(e1.cpp_element, e2.cpp_element)

    @staticmethod
    def c_ptr_to_existing_py_inst(long ptr_val):
        return (<cydecl.Element *>ptr_val).py_instance(False)

    @staticmethod
    def c_ptr_to_py_inst(long ptr_val):
        return (<cydecl.Element *>ptr_val).py_instance(True)

    @staticmethod
    cdef const cydecl.Element* _ident_to_cpp_element(int ident):
        return &cydecl.Element.get_element(ident)
        """
        if isinstance(ident, int):
            return &cydecl.Element.get_element(ident)
        return &cydecl.Element.get_named_element(ident.encode())
        """

    @staticmethod
    def get_element(ident):
        """
        if isinstance(ident, int):
            return Element._get_numeric_element(ident)
        return Element._get_string_element(ident)
        """
        return Element.c_ptr_to_py_inst(<long>Element._ident_to_cpp_element(ident))

    @staticmethod
    cdef _get_numeric_element(int e_num):
        return Element.c_ptr_to_py_inst(<long>&cydecl.Element.get_element(e_num))

    @staticmethod
    cdef _get_string_element(const char* e_name):
        return Element.c_ptr_to_py_inst(<long>&cydecl.Element.get_element(e_name))

cydecl.Element.set_py_class(Element)
'''
