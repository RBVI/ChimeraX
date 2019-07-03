# distutils: language=c++
#cython: language_level=3, boundscheck=False, auto_pickle=False 
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


cimport cytmpl
from tinyarray import array, zeros
from cython.operator import dereference, postincrement
cimport cython

IF UNAME_SYSNAME == "Windows":
    ctypedef long long ptr_type
ELSE:
    ctypedef long ptr_type

cdef class TmplResidue:
    '''Template residue class.'''
    cdef cytmpl.Residue *cpp_res

    def __cinit__(self, ptr_type ptr_val):
        self.cpp_res = <cytmpl.Residue *>ptr_val

    def __init__(self, ptr_val):
        if not isinstance(ptr_val, int) or ptr_val < 256:
            raise ValueError("Do not use %s constructor directly; use TmplResidue.get_template method"
                " and go from there" % self.__class__.__name__)

    @property
    def atoms(self):
        # use tmp to avoid Cython taking address of a copy
        tmp = self.cpp_res.atoms()
        return [a.py_instance(True) for a in tmp]

    @property
    def chief(self):
        return self.cpp_res.chief().py_instance(True)

    def find_atom(self, atom_name):
        '''Return the TmplAtom with the given name in this residue (or None if non-existent)'''
        fa_ptr = self.cpp_res.find_atom(atom_name.encode())
        if fa_ptr:
            return fa_ptr.py_instance(True)
        return None

    @staticmethod
    def get_template(res_name, *, start=False, end=False):
        '''Return the TmplResidue with the given name (or raise ValueError if no such template).
           start/end, if True, return a template residue for the corresponding chain terminus.
        '''
        tmpl_res = cytmpl.find_template_residue(res_name.encode(), start, end)
        if not tmpl_res:
            raise ValueError("No template for residue type %s" % res_name)
        return tmpl_res.py_instance(True)

    @property
    def link(self):
        return self.cpp_res.link().py_instance(True)

    @property
    def link_atoms(self):
        # fool bad Cython const declaration by using an intermediate...
        temp = self.cpp_res.link_atoms()
        return [la.py_instance(True) for la in temp]

    @property
    def name(self):
        return self.cpp_res.name().decode()

    @property
    def pdbx_ambiguous(self):
        return self.cpp_res.pdbx_ambiguous

cytmpl.Residue.set_py_class(TmplResidue)

cdef class TmplAtom:
    '''Template atom class.'''
    cdef cytmpl.Atom *cpp_atom

    def __cinit__(self, ptr_type ptr_val):
        self.cpp_atom = <cytmpl.Atom *>ptr_val

    def __init__(self, ptr_val):
        if not isinstance(ptr_val, int) or ptr_val < 256:
            raise ValueError("Do not use %s constructor directly; use TmplResidue.get_template method"
                " and go from there" % self.__class__.__name__)

    @property
    def bonds(self):
        tmp = self.cpp_atom.bonds()
        return [b.py_instance(True) for b in tmp]

    @property
    def coord(self):
        crd = self.cpp_atom.coord()
        return array((crd[0], crd[1], crd[2]))

    @property
    def element(self):
        return self.cpp_atom.element().py_instance(True)

    @property
    def idatm_type(self):
        return self.cpp_atom.idatm_type().decode()

    @property
    def name(self):
        return self.cpp_atom.name().decode()

    @property
    def neighbors(self):
        tmp = self.cpp_atom.neighbors()
        return [nb.py_instance(True) for nb in tmp]

    @property
    def residue(self):
        return self.cpp_atom.residue().py_instance(True)

cytmpl.Atom.set_py_class(TmplAtom)

cdef class TmplBond:
    '''Template bond class.'''
    cdef cytmpl.Bond *cpp_bond

    def __cinit__(self, ptr_type ptr_val):
        self.cpp_bond = <cytmpl.Bond *>ptr_val

    def __init__(self, ptr_val):
        if not isinstance(ptr_val, int) or ptr_val < 256:
            raise ValueError("Do not use %s constructor directly; use TmplResidue.get_template method"
                " and go from there" % self.__class__.__name__)

    @property
    def atoms(self):
        return [self.cpp_bond.atoms()[0].py_instance(True), self.cpp_bond.atoms()[1].py_instance(True)]

    def other_atom(self, TmplAtom a):
        return self.cpp_bond.other_atom(a.cpp_atom).py_instance(True)

cytmpl.Bond.set_py_class(TmplBond)

