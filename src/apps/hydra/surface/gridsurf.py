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
    from ..map.map_cpp import sphere_surface_distance
    sphere_surface_distance(ijk, ri, max_index_range, matrix)

    # Get the SAS surface as a contour surface of the distance map
    from ..map.map_cpp import surface
    level = 0
    sas_va, sas_ta, sas_na = surface(matrix, level, cap_faces = False,
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
    ses_va, ses_ta, ses_na = surface(matrix, level, cap_faces = False,
                                     calculate_normals = True)

    # Transform surface from grid index coordinates to atom coordinates
    xyz_to_ijk_tf.inverse().move(ses_va)

    # Delete connected components more than 1.5 probe radius from atom spheres.
    kvi = []
    kti = []
    from .surface_cpp import connected_pieces
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

def show_surface(name, va, na, ta, session, color = (.7,.7,.7,1), place = None):

    from ..models import Model
    surf = Model(name)
    if not place is None:
        surf.position = place
    surf.geometry = va, ta
    surf.normals = na
    surf.color = color
    session.add_model(surf)
    return surf

def surface_command(cmdname, args, session):

    from ..commands.parse import atoms_arg, float_arg, no_arg, parse_arguments
    req_args = (('atoms', atoms_arg),)
    opt_args = ()
    kw_args = (('probeRadius', float_arg),
               ('gridSpacing', float_arg),
               ('waters', no_arg),)

    kw = parse_arguments(cmdname, args, session, req_args, opt_args, kw_args)
    kw['session'] = session
    surface(**kw)

def surface(atoms, session, probeRadius = 1.4, gridSpacing = 0.5, waters = False):
    '''
    Compute and display a solvent excluded molecular surfaces for specified atoms.
    If waters is false then water residues (residue name HOH) are removed from
    the atom set before computing the surface.
    '''
    if not waters:
        atoms = atoms.exclude_water()
    xyz = atoms.coordinates()           # Scene coordinates
    r = atoms.radii()
    va,na,ta = ses_surface_geometry(xyz, r, probeRadius, gridSpacing)

    # Create surface model to show surface
    m0 = atoms.molecules()[0]
    p = m0.position
    if not p.is_identity(tolerance = 0):
        p.inverse().move(va)    # Move to model coordinates.
        
    name = '%s SES surface' % m0.name
    surf = show_surface(name, va, na, ta, session, color = (180,205,128,255), place = p)
    surf.ses_atoms = atoms

    return surf
