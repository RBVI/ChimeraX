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

# -----------------------------------------------------------------------------
#
def surface_area(varray, tarray):
    '''
    Return the surface area of a triangulation specified by vertex
    and triangle arrays.
    '''
    from ._surface import surface_area
    area = surface_area(varray, tarray)
    return area

# -----------------------------------------------------------------------------
#
def enclosed_volume(varray, tarray):
    '''
    Return the enclosed volume of a surface triangulation specified by vertex
    and triangle arrays. Also returns the number of holes in the surface,
    defined as the number of boundary curves.
    '''
    from ._surface import enclosed_volume
    vol, hole_count = enclosed_volume(varray, tarray)
    if vol < 0:
        return None, hole_count
    return vol, hole_count

# -----------------------------------------------------------------------------
# Calculate volume enclosed by a surface and surface area.
#
def surface_volume_and_area(model):
    '''
    Return the surface area, enclosed volume and number of holes (i.e. boundary
    curves) of a surface triangulation specified by vertex and triangle arrays.
    '''
# TODO: exclude surface caps and outline boxes.
    volume = holes = area = 0
    for d in model.all_drawings():
        varray, tarray = d.geometry
        if not varray is None:
            v, hc = enclosed_volume(varray, tarray)
            volume += 0 if v is None else v
            holes += hc
            area += surface_area(varray, tarray)
    return volume, area, holes
