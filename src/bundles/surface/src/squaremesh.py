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

# -------------------------------------------------------------------------------------
#
def hide_diagonals(surface):
    vertices = surface.vertices
    from numpy import array, uint8
    edge_mask = array([diagonal_mask(t, vertices) for t in surface.triangles], uint8)
    surface.edge_mask = edge_mask

def diagonal_mask(triangle, vertices):
    '''
    The edge mask uses the lowest 3 bits to indicate which of the 3 edges
    of a triangles should be shown.
    '''
    i0,i1,i2 = triangle
    v0,v1,v2 = vertices[i0],vertices[i1],vertices[i2]
    mask = 0
    if parallel_x_or_y_or_z(v0,v1):
        mask |= 0x1
    if parallel_x_or_y_or_z(v1,v2):
        mask |= 0x2
    if parallel_x_or_y_or_z(v2,v0):
        mask |= 0x4
    return mask

def parallel_x_or_y_or_z(xyz1,xyz2):
    return (xyz1 == xyz2).sum() >= 2
