# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2022 Regents of the University of California. All rights reserved.
# This software is provided pursuant to the ChimeraX license agreement, which
# covers academic and commercial uses. For more information, see
# <http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
#
# This file is part of the ChimeraX library. You can also redistribute and/or
# modify it under the GNU Lesser General Public License version 2.1 as
# published by the Free Software Foundation. For more details, see
# <https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html>
#
# This file is distributed WITHOUT ANY WARRANTY; without even the implied
# warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. This notice
# must be embedded in or attached to all copies, including partial copies, of
# the software or any revisions or derivations thereof.
# === UCSF ChimeraX Copyright ===

def ses_surface_geometry(xyz, radii, probe_radius = 1.4, grid_spacing = 0.5, sas = False):
    '''
    Calculate a solvent excluded molecular surface using a distance grid
    contouring method.  Vertex, normal and triangle arrays are returned.
    If sas is true then the solvent accessible surface is returned instead.
    '''

    # Compute bounding box for atoms
    xyz_min, xyz_max = xyz.min(axis = 0), xyz.max(axis = 0)
    pad = 2*probe_radius + radii.max() + grid_spacing
    origin = [x-pad for x in xyz_min]

    # Create 3d grid for computing distance map
    from math import ceil
    s = grid_spacing
    shape = [int(ceil((xyz_max[a] - xyz_min[a] + 2*pad) / s))
             for a in (2,1,0)]
#    print('ses surface grid size', shape, 'spheres', len(xyz))
    from numpy import empty, float32, sqrt
    try:
        matrix = empty(shape, float32)
    except (MemoryError, ValueError):
        raise MemoryError('Surface calculation out of memory trying to allocate a grid %d x %d x %d '
                          % (shape[2], shape[1], shape[0]) + 
                          'to cover xyz bounds %.3g,%.3g,%.3g ' % tuple(xyz_min) +
                          'to %.3g,%.3g,%.3g ' % tuple(xyz_max) +
                          'with grid size %.3g' % grid_spacing)
                          
    max_index_range = 2
    matrix[:,:,:] = max_index_range

    # Transform centers and radii to grid index coordinates
    from chimerax.geometry import Place
    xyz_to_ijk_tf = Place(((1.0/s, 0, 0, -origin[0]/s),
                           (0, 1.0/s, 0, -origin[1]/s),
                           (0, 0, 1.0/s, -origin[2]/s)))
    from numpy import float32
    ijk = xyz.astype(float32)
    xyz_to_ijk_tf.transform_points(ijk, in_place = True)
    ri = radii.astype(float32)
    ri += probe_radius
    ri /= s

    # Compute distance map from surface of spheres, positive outside.
    from chimerax.map import sphere_surface_distance
    sphere_surface_distance(ijk, ri, max_index_range, matrix)

    # Get the SAS surface as a contour surface of the distance map
    from chimerax.map import contour_surface
    level = 0
    sas_va, sas_ta, sas_na = contour_surface(matrix, level, cap_faces = False,
                                             calculate_normals = True)
    if sas:
        xyz_to_ijk_tf.inverse().transform_points(sas_va, in_place = True)
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
    xyz_to_ijk_tf.inverse().transform_points(ses_va, in_place = True)

    # Delete connected components more than 1.5 probe radius from atom spheres.
    from ._surface import connected_pieces
    vtilist = connected_pieces(ses_ta)
    from numpy import array, float32, sqrt
    vc0 = array([ses_va[vi[0],:] for vi,ti in vtilist], float32)
    rmax = radii.max()
    from chimerax.geometry import find_closest_points
    i1, i2, n1 = find_closest_points(vc0, xyz, 1.5*probe_radius + rmax)
    dxyz = xyz[n1] - vc0[i1]
    adist = sqrt((dxyz*dxyz).sum(axis=1)) - radii[n1] 
    ikeep = i1[adist < 1.5*probe_radius]
    kvi = [vtilist[i][0] for i in ikeep]
    kti = [vtilist[i][1] for i in ikeep]
    from numpy import concatenate
    keepv = concatenate(kvi) if kvi else []
    keept = concatenate(kti) if kti else []
    from .split import reduce_geometry
    va,na,ta = reduce_geometry(ses_va, ses_na, ses_ta, keepv, keept)

    return va, na, ta
