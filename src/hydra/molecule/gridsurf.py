#
# Calculate a solvent excluded molecular surface using a grid contouring method.
# This is test code, preparing for a production implementation in Chimera 2.
#
def ses_surface(atoms, probe_radius = 1.4, grid_spacing = 0.5, name = None):

    xyz = atoms.coordinates()
    r = atoms.radii()

    # Compute bounding box for atoms
    xyz_min, xyz_max = xyz.min(axis = 0), xyz.max(axis = 0)
    pad = 2*probe_radius + r.max()
    origin = [x-pad for x in xyz_min]

    # Create 3d grid for computing distance map
    from math import ceil
    s = grid_spacing
    shape = [int(ceil((xyz_max[a] - xyz_min[a] + 2*pad) / s))
             for a in (2,1,0)]
    print('grid size', shape, 'atoms', atoms.count())
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
    ri = r.copy()
    ri += probe_radius
    ri /= s

    # Compute distance map from surface of spheres, positive outside.
    from .._image3d import sphere_surface_distance
    sphere_surface_distance(ijk, ri, max_index_range, matrix)

    # Get the SAS surface as a contour surface of the distance map
    from .._image3d import surface
    level = 0
    va, ta, na = surface(matrix, level, cap_faces = False,
                         calculate_normals = True)

    # Transform surface from grid index coordinates to atom coordinates
    ijk_to_xyz_tf = xyz_to_ijk_tf.inverse()

    # Create surface model to show SAS surface
#    show_surface('SAS surface', va, ijk_to_xyz_tf, ta, na)

    # Compute SES surface distance map using SAS surface vertex
    # points as probe sphere centers.
    matrix[:,:,:] = max_index_range
    rp = empty((len(va),), float32)
    rp[:] = float(probe_radius)/s
    sphere_surface_distance(va, rp, max_index_range, matrix)
    ses_va, ses_ta, ses_na = surface(matrix, level, cap_faces = False,
                                     calculate_normals = True)

    # Create surface model to show surface
    m0 = atoms.molecules()[0]
    nm = ('%s SES surface' % m0.name) if name is None else name
    surf = show_surface(nm, ses_va, ijk_to_xyz_tf, ses_ta, ses_na,
                        color = (.7,.8,.5,1))
    surf.place = m0.place

    # Delete connected components more than 1.5 probe radius from atom spheres.
    from ..surface import split_surfaces
    split_surfaces(surf.surface_pieces(), in_place = True)
    outside = []
    for p in surf.surface_pieces():
        pva, pta = p.geometry
        v0 = pva[0,:]
        d = xyz - v0
        d2 = (d*d).sum(axis = 1)
        adist = (sqrt(d2) - r).min()
#        print(v0, adist)
        if adist >= 1.5*probe_radius:
            outside.append(p)
    surf.removePieces(outside)

    return surf

def show_surface(name, va, ijk_to_xyz_tf, ta, na, color = (.7,.7,.7,1)):

    va_xyz = va.copy()
    ijk_to_xyz_tf.move(va_xyz)
    from ..surface import Surface
    surf = Surface(name)
    p = surf.newPiece()
    p.geometry = va_xyz, ta
    p.normals = na
    p.color = color
    from ..ui.gui import main_window
    main_window.view.add_model(surf)
    return surf

def surface_command(cmdname, args):

    from ..ui.commands import atoms_arg, float_arg, no_arg, parse_arguments
    req_args = (('atoms', atoms_arg),)
    opt_args = ()
    kw_args = (('probeRadius', float_arg),
               ('gridSpacing', float_arg),
               ('waters', no_arg),)

    kw = parse_arguments(cmdname, args, req_args, opt_args, kw_args)
    surface(**kw)

def surface(atoms, probeRadius = 1.4, gridSpacing = 0.5, waters = False):

    if not waters:
        atoms = atoms.exclude_water()
    s = ses_surface(atoms, probeRadius, gridSpacing)
    return s
