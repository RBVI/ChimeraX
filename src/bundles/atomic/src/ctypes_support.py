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

# -------------------------------------------------------------------------------
# These routines convert C++ pointers to Python objects and are used for defining
# the object properties.
#
def atoms(p):
    from . import Atoms
    return Atoms(p)
def atom_pair(p):
    from . import Atom
    return (Atom.c_ptr_to_py_inst(p[0]), Atom.c_ptr_to_py_inst(p[1]))
def atom_or_none(p):
    from . import Atom
    return Atom.c_ptr_to_py_inst(p) if p else None
def bonds(p):
    from . import Bonds
    return Bonds(p)
def chain(p):
    if not p:
        return None
    from . import Chain
    return Chain.c_ptr_to_py_inst(p)
def coordset(p):
    from . import CoordSet
    return CoordSet.c_ptr_to_py_inst(p)
def coordsets(p):
    from . import CoordSets
    return CoordSets(p)
def element(p):
    from . import Element
    return Element.c_ptr_to_py_inst(p)
def pseudobonds(p):
    from . import Pseudobonds
    return Pseudobonds(p)
def residue(p):
    from . import Residue
    return Residue.c_ptr_to_py_inst(p)
def residues(p):
    from . import Residues
    return Residues(p)
def rings(p):
    from .molarray import Rings
    return Rings(p)
def non_null_residues(p):
    from . import Residues
    return Residues(p[p!=0])
def residue_or_none(p):
    from . import Residue
    return Residue.c_ptr_to_py_inst(p) if p else None
def residues_or_nones(p):
    return [residue_or_none(rptr) for rptr in p]
def chains(p):
    from . import Chains
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
