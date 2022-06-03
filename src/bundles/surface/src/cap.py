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

def update_clip_caps(view):
    from .settings import settings
    if not settings.clipping_surface_caps:
        return
    cp = view.clip_planes
    planes = cp.planes()
    # TODO: The check of cp.changed is unreliable because View.check_for_drawing_change()
    #  clears this flag, so it can be cleared before caps get a chance to update as happened
    #  in chimerax bug #2751.  Need a reliable mechanism to detect clipping changes.
    update = (cp.changed or
              (planes and
               (view.shape_changed or
                (view.camera.redraw_needed and cp.have_camera_plane()))))
    # TODO: Update caps only on specific drawings whose shape changed.
    if update:
        drawings = view.drawing.all_drawings()
        show_surface_clip_caps(planes, drawings,
                               offset = settings.clipping_cap_offset,
                               subdivision = settings.clipping_cap_subdivision,
                               cap_meshes = settings.clipping_cap_on_mesh)

def show_surface_clip_caps(planes, drawings, offset = 0.01, subdivision = 0.0, cap_meshes = False):
    for p in planes:
        for d in drawings:
            if d.was_deleted:
                continue	# Cap drawing that was removed.
            # Clip only drawings that have "clip_cap" attribute true.
            if _drawing_shows_caps(d, cap_meshes):
                varray, narray, tarray = compute_cap(d, p, offset, subdivision)
                set_cap_drawing_geometry(d, p.name, varray, narray, tarray)
            else:
                _remove_cap(d, p.name)

    # Remove caps for clip planes that are gone.
    plane_names = set(p.name for p in planes)
    for cap in drawings:
        if hasattr(cap, 'clip_cap_owner') and cap.clip_plane_name not in plane_names:
            _remove_cap(cap.clip_cap_owner, cap.clip_plane_name)

def _drawing_shows_caps(d, cap_meshes = False):
    return (d.display and d.parents_displayed and
            getattr(d, 'clip_cap', False)
            and d.triangles is not None
            and not hasattr(d, 'clip_cap_owner')
            and (cap_meshes or d.display_style != d.Mesh))

def remove_clip_caps(drawings):
    for cap in drawings:
        if hasattr(cap, 'clip_cap_owner'):
            d = cap.clip_cap_owner
            del d._clip_cap_drawings[cap.clip_plane_name]
            from chimerax.core.models import Model
            if isinstance(cap, Model):
                cap.session.models.close([cap])
            else:
                cap.parent.remove_drawing(cap)

def compute_cap(drawing, plane, offset, subdivision = 0.0):
    # Undisplay cap for drawing with no geometry shown.
    d = drawing
    if not d.display or not d.parents_displayed:
        return None, None, None

    # Handle surfaces with duplicate vertices, such as molecular
    # surfaces with sharp edges between atoms.
    if hasattr(d, 'joined_triangles'):
        t = d.joined_triangles
        if d.triangle_mask is not None and d.triangle_mask.sum() < len(d.triangle_mask):
            # TODO: triangle mask not handled for joined triangles.
            return None, None, None
    else:
        t = d.triangles
        if d.triangle_mask is not None and d.triangle_mask.sum() < len(d.triangle_mask):
            t = t[d.triangle_mask]

    # Compute cap geometry.
    # TODO: Cap instances
    np = len(d.get_scene_positions(displayed_only = True))
    if np > 1:
        varray, tarray, pnormal = compute_instances_cap(d, t, plane, offset)
    else:
        dp = d.scene_position.inverse()
        pnormal = dp.transform_vector(plane.normal)
        from chimerax.geometry import inner_product
        poffset = inner_product(pnormal, dp*plane.plane_point) + offset + getattr(d, 'clip_offset', 0)
        from . import compute_cap
        varray, tarray = compute_cap(-pnormal, -poffset, d.vertices, t)

    if tarray is None or len(tarray) == 0:
        return None, None, None
    narray = varray.copy()
    narray[:] = -pnormal

    if subdivision > 0:
        from . import refine_mesh
        varray, tarray = refine_mesh(varray, tarray, subdivision)
        if len(narray) > 0:
            normal = narray[0,:]
            narray = varray.copy()
            narray[:] = normal

    return varray, narray, tarray

def compute_instances_cap(drawing, triangles, plane, offset):
    d = drawing
    doffset = offset + getattr(d, 'clip_offset', 0)
    point = plane.plane_point
    normal = plane.normal

    b = d.geometry_bounds()
    if b is None:
        return None, None, None
        
    dpos = d.get_scene_positions(displayed_only = True)
    ipos = box_positions_intersecting_plane(dpos, b, point, normal)
    if len(ipos) == 0:
        return None, None, None
    geom = []
    for pos in ipos:
        pinv = pos.inverse()
        pnormal = pinv.transform_vector(normal)
        from chimerax.geometry import inner_product
        poffset = inner_product(pnormal, pinv*point) + doffset
        from . import compute_cap
        ivarray, itarray = compute_cap(-pnormal, -poffset, d.vertices, triangles)
        pos.transform_points(ivarray, in_place = True)
        geom.append((ivarray, itarray))
    varray, tarray = concatenate_geometry(geom)
    return varray, tarray, normal

def box_positions_intersecting_plane(positions, b, origin, normal):
    c, r = b.center(), b.radius()
    pc = positions * c
    pc -= origin
    from numpy import dot, abs
    dist = abs(dot(pc,normal))
    bint = (dist <= r)
    ipos = positions.masked(bint)
    return ipos

def concatenate_geometry(geom):
    from numpy import concatenate
    varray = concatenate(tuple(v for v,t in geom))
    tarray = concatenate(tuple(t for v,t in geom))
    voffset = ts = 0
    for v,t in geom:
        nt = len(t)
        tarray[ts:ts+nt,:] += voffset
        ts += nt
        voffset += len(v)
    return varray, tarray

def set_cap_drawing_geometry(drawing, plane_name, varray, narray, tarray):
    d = drawing
    # Set cap drawing geometry.
    if not hasattr(d, '_clip_cap_drawings'):
        d._clip_cap_drawings = {}
    mcap = d._clip_cap_drawings.get(plane_name, None)     # Find cap drawing
    if mcap and mcap.was_deleted:
        mcap = None

    if mcap:
        cm = mcap
    elif varray is None:
        return
    else:
        np = len(d.get_scene_positions(displayed_only = True))
        if np == 1:
            cm = _new_cap(d, plane_name)
        else:
            cm = _new_cap(d, plane_name, parent = _parent_above_instances(d))
            cm.pickable = False	  # Don't want pick of one cap to pick all instance caps.

    cm.set_geometry(varray, narray, tarray)

def _new_cap(drawing, plane_name, parent = None):
    if parent is None:
        parent = drawing

    cap_name = 'cap ' + plane_name
    if parent is not drawing:
        cap_name += ' ' + drawing.name
    
    from chimerax.core.models import Model, Surface
    if isinstance(drawing, Model):
        # Make cap a model when capping a model so color can be set by command.
        c = ClipCap(plane_name, drawing)
        c.name = cap_name
        parent.add([c])
    else:
        # Cap is on a Drawing that is not a Model
        c = ClipCapDrawing(plane_name, drawing)
        c.name = cap_name
        parent.add_drawing(c)
    
    return c

def _parent_above_instances(drawing):
    p = None
    for d in drawing.parent.drawing_lineage:
        if d.num_displayed_positions > 1:
            return p
        p = d
    return p

from chimerax.core.models import Surface
class ClipCap(Surface):
    is_clip_cap = True
    def __init__(self, plane_name, drawing):
        Surface.__init__(self, 'cap ' + plane_name, drawing.session)
        self.clip_plane_name = plane_name
        self.clip_cap_owner = drawing
        self.color = drawing.color
        self.display_style = drawing.display_style
        if not hasattr(drawing, '_clip_cap_drawings'):
            drawing._clip_cap_drawings = {}
        drawing._clip_cap_drawings[plane_name] = self

    # State save/restore in ChimeraX
    def take_snapshot(self, session, flags):
        data = {
            'clip_plane_name': self.clip_plane_name,
            'clip_cap_owner': self.clip_cap_owner,
            'model state': Surface.take_snapshot(self, session, flags),
            'version': 1
        }
        return data

    @staticmethod
    def restore_snapshot(session, data):
        drawing = data['clip_cap_owner']
        if drawing is None:
            return None	# Volume surface was not restored, so cannot restore cap.
        c = ClipCap(data['clip_plane_name'], drawing)
        Surface.set_state_from_snapshot(c, session, data['model state'])
        return c


from chimerax.graphics import Drawing
class ClipCapDrawing(Drawing):
    is_clip_cap = True
    def __init__(self, plane_name, drawing):
        Drawing.__init__(self, 'cap ' + plane_name)
        self.clip_plane_name = plane_name
        self.clip_cap_owner = drawing
        self.color = drawing.color
        self.display_style = drawing.display_style
        drawing._clip_cap_drawings[plane_name] = self

def _remove_cap(drawing, plane_name):
    d = drawing
    if not hasattr(d, '_clip_cap_drawings'):
        return
    cap = d._clip_cap_drawings.get(plane_name, None)     # Find cap drawing
    if cap is None:
        return
    del d._clip_cap_drawings[plane_name]
    if cap.was_deleted:
        return
    from chimerax.core.models import Model
    if isinstance(cap, Model):
        cap.session.models.close([cap])
    else:
        cap.parent.remove_drawing(cap)

