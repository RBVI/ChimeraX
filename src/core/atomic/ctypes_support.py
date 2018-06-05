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

# -------------------------------------------------------------------------------
# These routines convert C++ pointers to Python objects and are used for defining
# the object properties.
#
def atoms(p):
    from .molarray import Atoms
    return Atoms(p)
def atom_pair(p):
    from .molobject import Atom
    return (Atom.c_ptr_to_py_inst(p[0]), Atom.c_ptr_to_py_inst(p[1]))
def atom_or_none(p):
    from .molobject import Atom
    return Atom.c_ptr_to_py_inst(p) if p else None
def bonds(p):
    from .molarray import Bonds
    return Bonds(p)
def chain(p):
    if not p:
        return None
    from .molobject import Chain
    return Chain.c_ptr_to_py_inst(p)
def coordset(p):
    from .molobject import CoordSet
    return CoordSet.c_ptr_to_py_inst(p)
def element(p):
    from .molobject import Element
    return Element.c_ptr_to_py_inst(p)
def pseudobonds(p):
    from .molarray import Pseudobonds
    return Pseudobonds(p)
def residue(p):
    from .molobject import Residue
    return Residue.c_ptr_to_py_inst(p)
def residues(p):
    from .molarray import Residues
    return Residues(p)
def rings(p):
    from .molarray import Rings
    return Rings(p)
def non_null_residues(p):
    from .molarray import Residues
    return Residues(p[p!=0])
def residue_or_none(p):
    from .molobject import Residue
    return Residue.c_ptr_to_py_inst(p) if p else None
def residues_or_nones(p):
    return [residue_or_none(rptr) for rptr in p]
def chains(p):
    from .molarray import Chains
    return Chains(p)
def atomic_structure(p):
    from .molobject import StructureData
    return StructureData.c_ptr_to_py_inst(p) if p else None
def pseudobond_group(p):
    from .molobject import PseudobondGroupData
    return PseudobondGroupData.c_ptr_to_py_inst(p)
def pseudobond_group_map(pbgc_map):
    pbg_map = dict((name, pseudobond_group(pbg)) for name, pbg in pbgc_map.items())
    return pbg_map
