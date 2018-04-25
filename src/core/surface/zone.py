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
# Show only part of the surface model within specified distances of the given
# list of points.  The points are in model coordinates.
#
def surface_zone(model, points, distance, auto_update = False, max_components = None):

    from ..models import Model
    for d in model.child_drawings():
        if not isinstance(d, Model):
            surface_zone(d, points, distance, auto_update, max_components)

    t = model.triangles
    if t is None:
        return

    from ..geometry import find_close_points
    i1, i2 = find_close_points(model.vertices, points, distance)

    nv = len(model.vertices)
    from numpy import zeros, bool, put, logical_and
    mask = zeros((nv,), bool)
    put(mask, i1, 1)
    tmask = logical_and(mask[t[:,0]], mask[t[:,1]])
    logical_and(tmask, mask[t[:,2]], tmask)
    model.triangle_mask = tmask

    if not max_components is None:
        # TODO: port hide dust
        from . import dust
        dust.show_only_largest_blobs(model, True, max_components)


    if auto_update:
        remask = ZoneRemask(model, points, distance, max_components)
    else:
        remask = None
    model.auto_remask_triangles = remask

# -----------------------------------------------------------------------------
#
def path_points(atoms, bonds, bond_point_spacing = None):

    points = atoms.scene_coords

    if bonds is not None:
        from .bondzone import bond_points, concatenate_points
        bpoints = bond_points(bonds, xform_to_surface, bond_point_spacing)
        if len(bpoints) > 0:
            points = concatenate_points(points, bpoints)

    return points

# -----------------------------------------------------------------------------
#
class ZoneRemask:
    def __init__(self, model, points, distance, max_components):
        self.model = model
        self.points = points
        self.distance = distance
        self.max_components = max_components

    def __call__(self):
        surf = self.model
        surface_zone(surf, self.points, self.distance,
                     max_components = self.max_components, auto_update = False)
        surf.auto_remask_triangles = self

# -----------------------------------------------------------------------------
#
def surface_unzone(model):

    from ..models import Model
    for d in model.child_drawings():
        if not isinstance(d, Model):
            surface_unzone(d)

    model.auto_remask_triangles = None
    model.triangle_mask = None
