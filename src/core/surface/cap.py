def update_clip_caps(view):
    cp = view.clip_planes
    planes = cp.planes()
    cpos = view.camera.position
    for p in planes:
        p.update_direction(cpos)
    update = (cp.changed or (view.shape_changed and planes))
    # TODO: Don't update caps if shape change is drawing that does not show caps.
    if update:
        drawings = view.drawing.all_drawings()
        show_surface_clip_caps(planes, drawings)

def show_surface_clip_caps(planes, drawings, offset = 0.01):
    for p in planes:
        for d in drawings:
            # Clip only drawings that have "clip_cap" attribute true.
            if (hasattr(d, 'clip_cap') and d.clip_cap and
                d.triangles is not None and not hasattr(d, 'is_clip_cap')):
                varray, narray, tarray = compute_cap(d, p, offset)
                set_cap_drawing_geometry(d, p.name, varray, narray, tarray)

    # Remove caps for clip planes that are gone.
    cap_names = set('cap ' + p.name for p in planes)
    for d in drawings:
        if hasattr(d, 'is_clip_cap') and d.name not in cap_names:
            delattr(d.is_clip_cap, '_clip_cap_drawing_%s' % p.name)
            d.parent.remove_drawing(d)

def compute_cap(drawing, plane, offset):
    # Undisplay cap for drawing with no geometry shown.
    d = drawing
    if (not d.display or
        (d.triangle_mask is not None and
         d.triangle_mask.sum() < len(d.triangle_mask))):
        return None, None, None

    # Handle surfaces with duplicate vertices, such as molecular
    # surfaces with sharp edges between atoms.
    if d.clip_cap == 'duplicate vertices':
        from . import unique_vertex_map
        vmap = unique_vertex_map(d.vertices)
        t = vmap[d.triangles]
    else:
        t = d.triangles

    # Compute cap geometry.
    # TODO: Cap instances
    dp = d.scene_position.inverse()
    pnormal = dp.apply_without_translation(plane.normal)
    from ..geometry import inner_product
    poffset = inner_product(pnormal, dp*plane.plane_point) + offset + getattr(d, 'clip_offset', 0)
    from . import compute_cap
    cvarray, ctarray = compute_cap(pnormal, poffset, d.vertices, t)
    if len(ctarray) == 0:
        return None, None, None
    cnarray = cvarray.copy()
    cnarray[:] = plane.normal

    return cvarray, cnarray, ctarray

def set_cap_drawing_geometry(drawing, plane_name, varray, narray, tarray):
    d = drawing
    # Set cap drawing geometry.
    mcap = getattr(d, '_clip_cap_drawing_%s' % plane_name, None)     # Find cap drawing
    if varray is None:
        if mcap:
            mcap.display = False
        return

    if mcap:
        cm = mcap
    else:
        cap_name = 'cap ' + plane_name
        cm = d.new_drawing(cap_name)
        cm.is_clip_cap = d
    cm.vertices = varray
    cm.triangles = tarray
    cm.normals = narray
    cm.color = d.color
    cm.display = True

    #print ('capping ', d.name, len(ctarray), len(d.vertices), len(d.triangles), normal, poffset,
    #       d.bounds().xyz_min, d.bounds().xyz_max)
