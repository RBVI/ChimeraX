# distutils: language=c++
# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2022 Regents of the University of California. All rights reserved.
# The ChimeraX application is provided pursuant to the ChimeraX license
# agreement, which covers academic and commercial uses. For more details, see
# <https://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
#
# This particular file is part of the ChimeraX library. You can also
# redistribute and/or modify it under the terms of the GNU Lesser General
# Public License version 2.1 as published by the Free Software Foundation.
# For more details, see
# <https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html>
#
# THIS SOFTWARE IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND, EITHER
# EXPRESSED OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
# OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. ADDITIONAL LIABILITY
# LIMITATIONS ARE DESCRIBED IN THE GNU LESSER GENERAL PUBLIC LICENSE
# VERSION 2.1
#
# This notice must be embedded in or attached to all copies, including partial
# copies, of the software or any revisions or derivations thereof.
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
        map[string, vector[string]] metadata

        @staticmethod
        void set_py_class(object)
