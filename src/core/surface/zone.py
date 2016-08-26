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
        # TODO: port zone updating
        zone_updater.auto_zone(model, points, distance, max_components)

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
def get_atom_coordinates(atoms, xform):

    from _multiscale import get_atom_coordinates
    xyz = get_atom_coordinates(atoms, transformed = True)

    from _contour import affine_transform_vertices
    from Matrix import xform_matrix
    affine_transform_vertices(xyz, xform_matrix(xform))

    return xyz
        
# -----------------------------------------------------------------------------
#
def is_surface_piece_deleted(p):

    try:
        p.display
    except:
        return True
    return False
        
# -----------------------------------------------------------------------------
# Stop updating surface zone.
#
def no_surface_zone(model):
    
    zone_updater.stop_zone(model)
    import Surface
    Surface.reshow_surface(model)
            
# -----------------------------------------------------------------------------
#
class Zone_Updater:

    def __init__(self):

        self.models = {}

        import SimpleSession
        import chimera
        chimera.triggers.addHandler(SimpleSession.SAVE_SESSION,
                                    self.save_session_cb, None)
            
    # -------------------------------------------------------------------------
    #
    def auto_zone(self, model, points, distance, max_components):

        add_callback = not self.models.has_key(model)
        self.models[model] = (points, distance, max_components)
        if add_callback:
            from Surface import set_visibility_method
            set_visibility_method('surface zone', model, self.stop_zone)
            model.addGeometryChangedCallback(self.surface_changed_cb)
            import chimera
            chimera.addModelClosedCallback(model, self.model_closed_cb)
            
    # -------------------------------------------------------------------------
    #
    def stop_zone(self, model):

        if model in self.models:
            del self.models[model]
            model.removeGeometryChangedCallback(self.surface_changed_cb)
            
    # -------------------------------------------------------------------------
    #
    def surface_changed_cb(self, p, detail):

        if detail == 'removed':
            return

        m = p.model
        (points, distance, max_components) = self.models[m]
        surface_zone_piece(p, points, distance, max_components)
            
    # -------------------------------------------------------------------------
    #
    def model_closed_cb(self, model):

        if model in self.models:
            del self.models[model]
    
    # -------------------------------------------------------------------------
    #
    def save_session_cb(self, trigger, x, file):

        import session
        session.save_surface_zone_state(self.models, file)

# -----------------------------------------------------------------------------
#
def zonable_surface_models():

    import Surface as s
    return s.surface_models()

# -----------------------------------------------------------------------------
#
def showing_zone(model):
    return model in zone_updater.models
def zone_points_and_distance(model):
    return zone_updater.models[model][:2]

# -----------------------------------------------------------------------------
#
#zone_updater = Zone_Updater()
