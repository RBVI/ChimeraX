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
def surface_zone(surface, points, distance, auto_update = False, max_components = None):

    zm = ZoneMask(surface, points, distance, max_components)
    zm.set_surface_mask()

    if auto_update:
        from .updaters import add_updater_for_session_saving
        add_updater_for_session_saving(surface.session, zm)
    else:
        zm = None
    surface.auto_remask_triangles = zm

# -----------------------------------------------------------------------------
#
def path_points(atoms, bonds, bond_point_spacing = None):

    points = atoms.scene_coords

    if bonds is not None and len(bonds) > 0:
        from .bondzone import bond_points, concatenate_points
        bpoints = bond_points(bonds, bond_point_spacing)
        if len(bpoints) > 0:
            points = concatenate_points(points, bpoints)

    return points

# -----------------------------------------------------------------------------
#
from chimerax.core.state import State
class ZoneMask(State):
    def __init__(self, surface, points, distance, max_components):
        self.surface = surface
        self.points = points
        self.distance = distance
        self.max_components = max_components

    def __call__(self):
        self.set_surface_mask()

    def active(self):
        s = self.surface
        return s is not None and s.auto_remask_triangles is self

    def set_surface_mask(self):
        surface = self.surface
        surface.auto_remask_triangles = self
        
        v, t = surface.vertices, surface.triangles
        if t is None:
            return

        from chimerax.geometry import find_close_points
        i1, i2 = find_close_points(v, self.points, self.distance)

        nv = len(v)
        from numpy import zeros, put, logical_and
        mask = zeros((nv,), bool)
        put(mask, i1, 1)
        tmask = logical_and(mask[t[:,0]], mask[t[:,1]])
        logical_and(tmask, mask[t[:,2]], tmask)
        surface.triangle_mask = tmask
        surface.auto_remask_triangles = self	# setting triangle_mask clears remasking

        if not self.max_components is None:
            from . import dust
            dust.show_only_largest_blobs(surface, True, self.max_components)

    # -------------------------------------------------------------------------
    #
    def take_snapshot(self, session, flags):
        data = {
            'surface': self.surface,
            'points': self.points,
            'distance': self.distance,
            'max_components': self.max_components,
            'version': 1,
        }
        return data

    # -------------------------------------------------------------------------
    #
    @classmethod
    def restore_snapshot(cls, session, data):
        surf = data['surface']
        if surf is None:
            return None		# Surface to mask is gone.
        c = cls(surf, data['points'], data['distance'], data['max_components'])
        surf.auto_remask_triangles = c
        c.set_surface_mask()
        return c

# -----------------------------------------------------------------------------
#
def zoning(surface):
    zm = surface.auto_remask_triangles
    return zm if isinstance(zm, ZoneMask) else None

# -----------------------------------------------------------------------------
#
def surface_unzone(surface):
    surface.auto_remask_triangles = None
    surface.triangle_mask = None
