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
cimport cyelem
cimport cycoord

cdef extern from "<atomstruct/tmpl/Atom.h>" namespace "tmpl":
    cdef cppclass Atom:
        const vector[Bond*]& bonds()
        cycoord.Coord& coord()
        cyelem.Element& element()
        const char* idatm_type()
        string name()
        const vector[Atom*]& neighbors()
        object py_instance(bool)
        Residue* residue()

        @staticmethod
        void set_py_class(object)

cdef extern from "<atomstruct/tmpl/Bond.h>" namespace "tmpl":
    cdef cppclass Bond:
        Atom[2]* const atoms()
        float length()
        Atom* other_atom(Atom* a)
        object py_instance(bool)

        @staticmethod
        void set_py_class(object)

cdef extern from "<atomstruct/tmpl/residues.h>" namespace "tmpl":
    const Residue* find_template_residue(const char*, bool, bool) except +

cdef extern from "<atomstruct/tmpl/Residue.h>" namespace "tmpl":
    cdef cppclass Residue:
        vector[Atom*] atoms()
        Atom* chief()
        Atom* find_atom(const char*)
        Atom* link()
        const vector[Atom*]& link_atoms()
        string name()
        bool pdbx_ambiguous
        string description()
        object py_instance(bool)

        @staticmethod
        void set_py_class(object)
