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
    cdef cppclass Atom

    cdef cppclass Structure:
        void delete_atom(Atom*)
        void delete_atoms(vector[Atom*])
        CoordSet* find_coord_set(int)
        object py_instance(bool)

cdef extern from "<atomstruct/Chain.h>" namespace "atomstruct":
    cdef cppclass Chain:
        object py_instance(bool)

cdef extern from "<atomstruct/Atom.h>" namespace "atomstruct":
    ctypedef string AtomType
    cdef cppclass Bond
    cdef cppclass Ring

    cdef cppclass Coord:
        double operator[](int)

    cdef cppclass Point:
        Point(double x, double y, double z)

    cdef cppclass Rgba:
        ctypedef unsigned char Channel
        Channel r, g, b, a

    ctypedef enum BackboneExtent:
        BBE_MIN, BBE_RIBBON, BBE_MAX

cdef extern from "<atomstruct/Residue.h>" namespace "atomstruct":
    ctypedef enum PolymerType:
        PT_NONE, PT_AMINO, PT_NUCLEIC

    cdef cppclass Residue:
        ctypedef enum SSType:
            SS_COIL, SS_HELIX, SS_STRAND

        void add_atom(Atom*)
        const vector[Atom*]& atoms()
        vector[Bond*] bonds_between(Residue*)
        Chain* chain()
        string chain_id()
        bool connects_to(Residue*)
        Atom* find_atom(const char*)
        char insertion_code()
        bool is_helix()
        bool is_strand()
        string mmcif_chain_id()
        string name()
        int number()
        PolymerType polymer_type()
        Atom* principal_atom()
        object py_instance(bool)
        float ribbon_adjust()
        const Rgba& ribbon_color()
        bool ribbon_display()
        bool ribbon_hide_backbone()
        void set_alt_loc(char)
        void set_insertion_code(char)
        void set_is_helix(bool)
        void set_is_strand(bool)
        void set_name(const char*)
        void set_ribbon_adjust(float)
        void set_ribbon_color(Rgba.Channel, Rgba.Channel, Rgba.Channel, Rgba.Channel)
        void set_ribbon_display(bool)
        void set_ribbon_hide_backbone(bool)
        void set_ss_id(int)
        void set_ss_type(SSType)
        int ss_id()
        SSType ss_type()
        Structure* structure()

        @staticmethod
        void set_py_class(object)

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
    ctypedef vector[const Ring*] Rings
    ctypedef enum StructCat:
        Unassigned "atomstruct::Atom::StructCat::Unassigned",
        Main "atomstruct::Atom::StructCat::Main",
        Ligand "atomstruct::Atom::StructCat::Ligand",
        Ions "atomstruct::Atom::StructCat::Ions",
        Solvent "atomstruct::Atom::StructCat::Solvent"

cdef extern from "<atomstruct/Atom.h>" namespace "atomstruct":
    cdef cppclass Atom:
        ctypedef vector[Atom*] Neighbors
        char alt_loc()
        set[char] alt_locs()
        const vector[float]* aniso_u()
        float bfactor()
        Bonds bonds()
        void clear_hide_bits(int)
        const Rgba& color()
        bool connects_to(Atom*)
        const Coord& coord()
        const Coord& coord(char)
        const Coord& coord(CoordSet*)
        int coord_index()
        float default_radius()
        bool display()
        DrawMode draw_mode()
        const Element& element()
        bool has_alt_loc(char)
        int hide()
        const char* idatm_type()
        bool is_backbone(BackboneExtent)
        bool is_ribose()
        bool is_side_connector()
        bool is_side_chain(bool)
        float maximum_bond_radius(float)
        const char* name()
        const Neighbors& neighbors()
        int num_explicit_bonds()
        float occupancy()
        object py_instance(bool)
        float radius()
        Residue* residue()
        const Rings& rings(bool, int)
        Coord scene_coord()
        Coord scene_coord(char)
        Coord scene_coord(CoordSet*)
        bool selected()
        int serial_number()
        void set_alt_loc(char)
        void set_alt_loc(char, bool, bool)
        void set_aniso_u(float, float, float, float, float, float)
        void set_bfactor(float)
        void set_color(Rgba.Channel, Rgba.Channel, Rgba.Channel, Rgba.Channel)
        void set_coord(const Point&)
        void set_coord(const Point&, CoordSet*)
        void set_display(bool)
        void set_draw_mode(DrawMode)
        void set_hide(int)
        void set_hide_bits(int)
        void set_idatm_type(const char*)
        void set_name(const char*)
        void set_occupancy(float)
        void set_radius(float)
        void set_selected(bool)
        void set_serial_number(int)
        Structure* structure()
        StructCat structure_category()
        void use_default_radius()
        bool visible()

        @staticmethod
        const map[AtomType, IdatmInfo]& get_idatm_info_map()
        @staticmethod
        void set_py_class(object)
