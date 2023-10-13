# vim: set expandtab shiftwidth=4 softtabstop=4:
# distutils: language=c++
# distutils: include_dirs=search_cpp
#cython: language_level=3, boundscheck=False, auto_pickle=False 

# === UCSF ChimeraX Copyright ===
# Copyright 2022 Regents of the University of California. All rights reserved.
# This software is provided pursuant to the ChimeraX license agreement, which
# covers academic and commercial uses. For more information, see
# <http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
#
# This file is part of the ChimeraX library. You can also redistribute and/or
# modify it under the GNU Lesser General Public License version 2.1 as
# published by the Free Software Foundation. For more details, see
# <https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html>
#
# This file is distributed WITHOUT ANY WARRANTY; without even the implied
# warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. This notice
# must be embedded in or attached to all copies, including partial copies, of
# the software or any revisions or derivations thereof.
# === UCSF ChimeraX Copyright ===


cimport ast
from libcpp.vector cimport vector
from libcpp cimport bool

ctypedef ast.Atom* atom_ptr
IF UNAME_SYSNAME == "Windows":
    ctypedef long long ptr_type
ELSE:
    ctypedef long ptr_type

cdef class CyAtomSearchTree:
    '''AtomSearchTree is a specialization of an 'adaptive k-d tree'
       as per "The Design and Analysis of Spatial Data Structures" pp. 70-71.
       Basically, given a set of k-dimensional points (each dimension referred
       to as an "attribute") with associated data, they are partitioned into
       leaf nodes.  Each leaf nodes hold lists of associated data whose
       corresponding attributes vary by less than an initially-supplied threshold
       ('sep_val').  Also, each leaf node holds a bounding box of the leaf data.
       If 'scene_coords' is True, then the atoms' scene_coords will be used for
       the search rather than their coords.

       The interior nodes of the tree contain details of the partitioning.
       In particular, what attribute this node partitions along ('axis'),
       and what value ('median') partitions left child node from right child node.
       Whether a node is interior or leaf is stored in 'type'.

       The specialization is that this is a 3D tree of Atoms.
    '''
    cdef ast.CppAtomSearchTree *cpp_ast

    def __cinit__(self, atoms, *, bool scene_coords=True, double sep_val=5.0, **kw):
        cdef vector[atom_ptr] atom_ptrs = vector[atom_ptr]()
        from chimerax.atomic import Atoms
        # though Atoms Collections could use the generic 'for a in atoms' code,
        # use specialized code to avoid generating Python objects for every Atom
        # in the search tree
        if isinstance(atoms, Atoms):
            for a_ptr in atoms.pointers:
                atom_ptrs.push_back(<atom_ptr><ptr_type>a_ptr)
        else:
            for a in atoms:
                atom_ptrs.push_back(<atom_ptr><ptr_type>a.cpp_pointer)
        self.cpp_ast = new ast.CppAtomSearchTree(atom_ptrs, scene_coords, sep_val)

    def search(self, target, double window):
        """Search tree for all leaves within 'window' of target.  Target must be an Atom
        or a sequence of 3 numbers.

        The cumulative difference of all three coordinates from target must
        be less than 'window'.

        Note that unlike chimerax.geometry.AdaptiveTree, only items that are
        within 'window' are returned, rather than all items in leaf nodes that are
        within 'window' (so some individual items might not be within 'window').
        """
        from chimerax.atomic import Atom
        if isinstance(target, Atom):
            leaf_atom_ptrs = self.cpp_ast.search(<atom_ptr><ptr_type>target.cpp_pointer, window)
        else:
            x, y, z = target
            leaf_atom_ptrs = self.cpp_ast.search(ast.Coord(x, y, z), window)

        leaf_atoms = [a.py_instance(True) for a in leaf_atom_ptrs]
        if self.data_lookup:
            return [self.data_lookup[id(a)] for a in leaf_atoms]
        return leaf_atoms

    def __dealloc__(self):
        del self.cpp_ast

class AtomSearchTree(CyAtomSearchTree):
    def __init__(self, atoms, *, scene_coords=True, sep_val=5.0, data=None):
        """See class doc string for basic info.

           If data is not None, then it should be a sequence/array of the same length
           as 'atoms' in which case searches will return the corresponding data items
           (otherwise searches return the appropriate atoms).
        """
        self.data_lookup = {}
        if data:
            for a, datum in zip(atoms, data):
                self.data_lookup[id(a)] = datum
