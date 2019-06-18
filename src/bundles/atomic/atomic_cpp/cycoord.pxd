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

cdef extern from "<atomstruct/Atom.h>" namespace "atomstruct":
    cdef cppclass Coord:
        double operator[](int)

    cdef cppclass Point:
        Point(double x, double y, double z)
