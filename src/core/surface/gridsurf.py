def ses_surface_geometry(xyz, radii, probe_radius = 1.4, grid_spacing = 0.5, sas = False):
    '''
    Calculate a solvent excluded molecular surface using a distance grid
    contouring method.  Vertex, normal and triangle arrays are returned.
    If sas is true then the solvent accessible surface is returned instead.
    '''

    # Compute bounding box for atoms
    xyz_min, xyz_max = xyz.min(axis = 0), xyz.max(axis = 0)
    pad = 2*probe_radius + radii.max()
    origin = [x-pad for x in xyz_min]

    # Create 3d grid for computing distance map
    from math import ceil
    s = grid_spacing
    shape = [int(ceil((xyz_max[a] - xyz_min[a] + 2*pad) / s))
             for a in (2,1,0)]
    print('grid size', shape, 'spheres', len(xyz))
    from numpy import empty, float32, sqrt
    matrix = empty(shape, float32)
    max_index_range = 2
    matrix[:,:,:] = max_index_range

    # Transform centers and radii to grid index coordinates
    from ..geometry.place import Place
    xyz_to_ijk_tf = Place(((1.0/s, 0, 0, -origin[0]/s),
                           (0, 1.0/s, 0, -origin[1]/s),
                           (0, 0, 1.0/s, -origin[2]/s)))
    ijk = xyz.copy()
    xyz_to_ijk_tf.move(ijk)
    ri = radii.copy()
    ri += probe_radius
    ri /= s

    # Compute distance map from surface of spheres, positive outside.
    from ..map import sphere_surface_distance
    sphere_surface_distance(ijk, ri, max_index_range, matrix)

    # Get the SAS surface as a contour surface of the distance map
    from ..map import contour_surface
    level = 0
    sas_va, sas_ta, sas_na = contour_surface(matrix, level, cap_faces = False,
                                             calculate_normals = True)
    if sas:
        xyz_to_ijk_tf.inverse().move(sas_va)
        return sas_va, sas_na, sas_ta

    # Compute SES surface distance map using SAS surface vertex
    # points as probe sphere centers.
    matrix[:,:,:] = max_index_range
    rp = empty((len(sas_va),), float32)
    rp[:] = float(probe_radius)/s
    sphere_surface_distance(sas_va, rp, max_index_range, matrix)
    ses_va, ses_ta, ses_na = contour_surface(matrix, level, cap_faces = False,
                                             calculate_normals = True)

    # Transform surface from grid index coordinates to atom coordinates
    xyz_to_ijk_tf.inverse().move(ses_va)

    # Delete connected components more than 1.5 probe radius from atom spheres.
    kvi = []
    kti = []
    from ._surface import connected_pieces
    vtilist = connected_pieces(ses_ta)
    for vi,ti in vtilist:
        v0 = ses_va[vi[0],:]
        d = xyz - v0
        d2 = (d*d).sum(axis = 1)
        adist = (sqrt(d2) - radii).min()
        if adist < 1.5*probe_radius:
            kvi.append(vi)
            kti.append(ti)
    from .split import reduce_geometry
    from numpy import concatenate
    va,na,ta = reduce_geometry(ses_va, ses_na, ses_ta,
                               concatenate(kvi), concatenate(kti))
    return va, na, ta
