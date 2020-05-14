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

def centroid(atoms, *, use_scene_coords=None, mass_weighting=False):
    """
    Compute a centroid from an Atoms collection.

    If `mass_weighting` is True, weight each atom by its mass.
    If `use_scene_coords` is True then use the atoms' scene coordinates,
    if false use untransformed coordinates, if None then use scene
    coordinates if the atoms are in multiple structures, else
    untransformed coordinates.

    Returns an xyz array.
    """
    if use_scene_coords is None:
        use_scene_coords = len(atoms.unique_structures) > 1
    if use_scene_coords:
        crds = atoms.scene_coords
    else:
        crds = atoms.coords
    if mass_weighting:
        masses = atoms.elements.masses
        avg_mass = masses.sum() / len(masses)
        import numpy
        crds = crds * masses[:, numpy.newaxis] / avg_mass
    return crds.mean(0)
