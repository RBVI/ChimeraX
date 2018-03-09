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
        const Element& get_element(int)

        @staticmethod
        const Element& get_named_element "get_element"(const char*)

        @staticmethod
        const set[string]& names()

        @staticmethod
        void set_py_class(object)

cdef extern from "<atomstruct/Structure.h>" namespace "atomstruct":
    cdef cppclass CoordSet

    cdef cppclass Structure:
        CoordSet* find_coord_set(int)
        object py_instance(bool)

cdef extern from "<atomstruct/Residue.h>" namespace "atomstruct":
    cdef cppclass Residue:
        object py_instance(bool)

cdef extern from "<atomstruct/Atom.h>" namespace "atomstruct":
    ctypedef string AtomType
    cdef cppclass Bond

    cdef cppclass Coord:
        double operator[](int)

    cdef cppclass Point:
        Point(double x, double y, double z)

    cdef cppclass Rgba:
        ctypedef unsigned char Channel
        Channel r, g, b, a

cdef extern from "<atomstruct/Atom.h>" namespace "atomstruct::Atom":
    ctypedef enum IdatmGeometry:
        Ion, Single, Linear, Planar, Tetrahedral
    ctypedef struct IdatmInfo:
        IdatmGeometry geometry
        unsigned int substituents
        string description
    #ctypedef map[AtomType, IdatmInfo] IdatmInfoMap
    ctypedef enum DrawMode:
        Sphere, EndCap, Ball
    ctypedef vector[Bond*] Bonds

cdef extern from "<atomstruct/Atom.h>" namespace "atomstruct":
    cdef cppclass Atom:
        ctypedef vector[Atom*] Neighbors
        char alt_loc()
        set[char] alt_locs()
        float bfactor()
        Bonds bonds()
        const Rgba& color()
        const Coord& coord()
        bool display()
        DrawMode draw_mode()
        const Element& element()
        int hide()
        const char* idatm_type()
        const char* name()
        const Neighbors& neighbors()
        object py_instance(bool)
        float radius()
        Residue* residue()
        void set_alt_loc(char, bool, bool)
        void set_bfactor(float)
        void set_color(Rgba.Channel, Rgba.Channel, Rgba.Channel, Rgba.Channel)
        void set_coord(const Point&)
        void set_coord(const Point&, CoordSet*)
        void set_display(bool)
        void set_draw_mode(DrawMode)
        void set_hide(int)
        void set_idatm_type(const char*)
        Structure* structure()

        @staticmethod
        const map[AtomType, IdatmInfo]& get_idatm_info_map()
        @staticmethod
        void set_py_class(object)
