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
        normal = p.normal
        from ..geometry import inner_product
        poffset = inner_product(normal, p.plane_point) + offset
        cap_name = 'cap ' + p.name
        for d in drawings:
            if not hasattr(d, 'clip_cap') or not d.clip_cap or d.triangles is None or hasattr(d, 'is_clip_cap'):
                continue
            if d.clip_cap == 'duplicate vertices':
                from . import unique_vertex_map
                vmap = unique_vertex_map(d.vertices)
                t = vmap[d.triangles]
#                from . import check_surface_topology
#                check_surface_topology(t, d.name)
            else:
                t = d.triangles
            from . import compute_cap
            cvarray, ctarray = compute_cap(normal, poffset, d.vertices, t)
            mcap = [cm for cm in d.child_drawings() if cm.name == cap_name]
            if mcap:
                cm = mcap[0]
            else:
                cm = d.new_drawing(cap_name)
                cm.is_clip_cap = True
            cm.vertices = cvarray
            cm.triangles = ctarray
#            print ('capping ', d.name, len(ctarray), len(d.vertices), len(d.triangles), normal, poffset,
#                   d.bounds().xyz_min, d.bounds().xyz_max)
            n = cvarray.copy()
            n[:] = normal
            cm.normals = n
            cm.display = True

    cap_names = set('cap ' + p.name for p in planes)
    for d in drawings:
        if hasattr(d, 'is_clip_cap') and d.name not in cap_names:
            d.parent.remove_drawing(d)
