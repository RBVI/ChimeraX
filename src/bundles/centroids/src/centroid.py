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

def centroid(xyzs, *, weights=False):
    """
    Compute a centroid from a numpy Nx3 array of floats,
    optionally weighted by a 'weights' array of length N.

    The 'xyzs' is frequently obtained from an Atoms collection
    from either its 'coords' or 'scene_coords' attributes,
    usually depending whether the centroid should be in particular
    structure's coordinate system, or in the global coordinate
    system containing several structures.

    Returns an xyz array.
    """
    if weights is not None:
        xyzs = xyzs * weights
    return xyzs.mean(0)
