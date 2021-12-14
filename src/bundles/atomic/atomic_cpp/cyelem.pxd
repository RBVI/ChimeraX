# distutils: language=c++
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

from libcpp.string cimport string
from libcpp.map cimport map
from libcpp.set cimport set
from libcpp.vector cimport vector
from libcpp cimport bool

cdef extern from "<element/Element.h>" namespace "element::Element":
    ctypedef enum AS:
        NUM_SUPPORTED_ELEMENTS

cdef extern from "<element/Element.h>" namespace "element":
    cdef cppclass Element:
        bool is_alkali_metal()
        bool is_halogen()
        bool is_metal()
        bool is_noble_gas()
        float mass()
        const char* name()
        int number()
        object py_instance(bool)
        int valence()

        @staticmethod
        float bond_length(Element&, Element&)

        @staticmethod
        float bond_radius(Element&)

        @staticmethod
        const Element& get_element(int) except +

        @staticmethod
        const Element& get_named_element "get_element"(const char*)

        @staticmethod
        const set[string]& names()

        @staticmethod
        void set_py_class(object)
