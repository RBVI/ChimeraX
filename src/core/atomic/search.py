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

"""
search: finding atoms in 3D space
=================================

TODO
"""

def atom_search_tree(atoms, data=None, sep_val=5.0, scene_coords=True):
    """atom_search_tree creates and returns an :py:class:`..geometry.AdaptiveTree`
       for the given :py:class:`.Atoms` collection, or list of atoms.  If data is not
       None, then it should be a sequence/array of the same length as 'atoms' in which
       case searches will return the corresponding data items (otherwise searches return
       the appropriate atoms).  'sep_val' is passed to the AdaptiveTree constructor.  If
       'scene_coords' is True, then the atoms' scene_coords will be used for the search
       rather than their coords.
    """
    from . import Atoms
    if not isinstance(atoms, Atoms):
        atoms = Atoms(atoms)
    coords = atoms.scene_coords if scene_coords else atoms.coords
    if data is None:
        data = atoms
    from ..geometry import AdaptiveTree
    return AdaptiveTree(coords, data, sep_val)
