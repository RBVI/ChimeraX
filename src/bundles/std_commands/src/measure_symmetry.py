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

# -----------------------------------------------------------------------------
#
def measure_symmetry(session, volumes, minimum_correlation = 0.99, n_max = 8,
                     helix = None, points = 10000, set = True):

    if len(volumes) == 0:
        from chimerax.core.errors import UserError
        raise UserError('No volume specified')

    for v in volumes:
        if helix:
            rise, angle, n, optimize = helix
            syms, msg = find_helix_symmetry(v, rise, angle, n, optimize, n_max,
                                            minimum_correlation, points)
        else:
            syms, msg = find_point_symmetry(v, n_max, minimum_correlation, points)

        if set and syms:
            v.data.symmetries = syms

        session.logger.status(msg, log = True)

# -----------------------------------------------------------------------------
#
from chimerax.core.commands import Annotation
class HelixArg(Annotation):
    '''Helix symmetry search parameters'''
    name = 'helix parameters'

    @staticmethod
    def parse(text, session):
        from chimerax.core.commands import next_token, AnnotationError
        if not text:
            raise AnnotationError("Expected %s" % HelixArg.name)
        token, text, rest = next_token(text)
        fields = token.split(',')
        optimize = (fields and fields[-1] == 'opt')
        if optimize:
            fields = fields[:-1]
        herr = 'Invalid helix option rise,angle[,n][,opt]'
        if len(fields) in (2,3):
            try:
                rise, angle = [float(f) for f in fields[:2]]
                n = int(fields[2]) if len(fields) == 3 else None
            except ValueError:
                raise AnnotationError(herr)
        else:
            raise AnnotationError(herr)
        hparams =  (rise, angle, n, optimize)
        return hparams, text, rest

# -----------------------------------------------------------------------------
#
def register_command(logger):
    from chimerax.core.commands import CmdDesc, register, FloatArg, IntArg, BoolArg
    from chimerax.map import MapsArg
    desc = CmdDesc(
        required = [('volumes', MapsArg)],
        keyword = [('minimum_correlation', FloatArg),
                   ('n_max', IntArg),
                   ('helix', HelixArg),
                   ('points', IntArg),
                   ('set', BoolArg),],
        synopsis = 'compute density map symmetry')
    register('measure symmetry', desc, measure_symmetry, logger=logger)

# -----------------------------------------------------------------------------
#
def find_point_symmetry(v, n_max = 8, minimum_correlation = 0.99,
                        maximum_points = 10000):

    centers, xyz, w = centers_and_points(v, maximum_points)
    for center in centers:
        descrip, syms = point_symmetry(v, center, xyz, w,
                                       minimum_correlation, n_max)
        if syms:
            break

    if syms is None:
        msg = 'No symmetry found for %s' % v.name
    else:
        icenter = [round(i) for i in v.data.xyz_to_ijk(center)]
        msg = ('Symmetry %s: %s, center %s' %
               (v.name, descrip, '%d %d %d' % tuple(icenter)))

    return syms, msg

# -----------------------------------------------------------------------------
#
def centers_and_points(v, maximum_points = 10000):

    m = v.full_matrix()
    lev = v.minimum_surface_level
    from chimerax.map import high_indices
    ijk = high_indices(m, lev)
    ijk_mean = ijk.mean(axis = 0)

    # Test at most maximum_points in correlation calculation to make it faster.
    if not maximum_points is None and len(ijk) > maximum_points:
        from numpy.random import randint
        r = randint(len(ijk), size = maximum_points)
        ijk = ijk[r,:]

    from numpy import float32
    xyz = ijk.astype(float32)
    d = v.data
    d.ijk_to_xyz_transform.transform_points(xyz, in_place = True)
    w = v.interpolated_values(xyz)

    icenter = [round(i) for i in ijk_mean]
    center = tuple(d.ijk_to_xyz(icenter))
    centers = [center]
    for offset in (0, -1):
        icenter = [(n-1)/2 if n%2 else (n/2)+offset for n in d.size]
        center = tuple(d.ijk_to_xyz(icenter))
        if not center in centers:
            centers.append(center)

    return centers, xyz, w

# -----------------------------------------------------------------------------
#
def point_symmetry(v, center, xyz, w, minimum_correlation = 0.99, nmax = 8):

    from chimerax import geometry

    # Look for icosahedral symmetry.
    csname = icosahedral_symmetry(v, center, xyz, w, minimum_correlation)
    if csname:
        syms = geometry.icosahedral_symmetry_matrices(csname, center)
        return ('Icosahedral %s' % csname), syms

    # Look for z-axis cyclic symmetry.
    n = cyclic_symmetry(nmax, v, (0,0,1), center, xyz, w, minimum_correlation)
    if n is None:
        return None, None

    # Check for octahedral symmetry.
    if n == 4:
        c4x, cm = correlation(v, xyz, w, (1,0,0), 90, center)
        if c4x >= minimum_correlation:
            syms = geometry.octahedral_symmetry_matrices(center)
            return 'Octahedral', syms

    # Check for tetrahedral symmetry 3-fold on z, EMAN convention.
    if n == 3:
        from math import sqrt
        a3yz = (0,-2*sqrt(2)/3,-1.0/3)  # 3-fold in yz plane
        c3yz, cm = correlation(v, xyz, w, a3yz, 120, center)
        if c3yz >= minimum_correlation:
            syms = geometry.tetrahedral_symmetry_matrices('z3', center)
            return 'Tetrahedral z3', syms

    # Check for tetrahedral symmetry, 2-folds on x,y,z.
    c2x, cm = correlation(v, xyz, w, (1,0,0), 180, center)
    c2y, cm = correlation(v, xyz, w, (0,1,0), 180, center)
    if n == 2 and c2x >= minimum_correlation and c2y >= minimum_correlation:
        c3d, cm = correlation(v, xyz, w, (1,1,1), 120, center)
        if c3d >= minimum_correlation:
            syms = geometry.tetrahedral_symmetry_matrices('222', center)
            return 'Tetrahedral', syms
        
    # Check for dihedral symmetry, x axis flip.
    if c2x >= minimum_correlation:
        syms = geometry.dihedral_symmetry_matrices(n, center)
        return 'D%d' % n, syms

    # Check for dihedral symmetry, y axis flip.
    if c2y >= minimum_correlation:
        zrot = geometry.rotation((0,0,1), -90)
        syms = geometry.dihedral_symmetry_matrices(n).transform_coordinates(zrot)
        syms = geometry.recenter_symmetries(syms, center)
        return 'D%d, y flip' % n, syms

    syms = geometry.cyclic_symmetry_matrices(n, center)
    return ('C%d' % n), syms

# -----------------------------------------------------------------------------
#
def icosahedral_symmetry(v, center, xyz, w, minimum_correlation):

    from chimerax import geometry
    a23, a25, a35 = geometry.icosahedron_angles()
    from math import sin, cos
    a2 = (0,0,1)
    a5 = (0, sin(a25), cos(a25))
    for cs in geometry.icosahedral_orientations:
        tf = geometry.icosahedral_coordinate_system_transform('222', cs)
        c2, cm2 = correlation(v, xyz, w, tf*a2, 180, center)
        c5, cm5 = correlation(v, xyz, w, tf*a5, 72, center)
        if c2 >= minimum_correlation and c5 >= minimum_correlation:
            return cs
    return None

# -----------------------------------------------------------------------------
# If more than one symmetry N value exceeds minimum correlation, return highest
# correlation N that is not a subgroup.
#
def cyclic_symmetry(nmax, v, axis, center, xyz, w, minimum_correlation):

    nlist = range(2, nmax+1)
    ntry = set(nlist)
    nsym = {}
    for n in nlist:
        if n in ntry:
            angle = 360.0 / n
            c, cm = correlation(v, xyz, w, axis, angle, center)
            if c >= minimum_correlation:
                nsym[n] = c
            else:
                # Skip multiples of N having correlation too low.
                for k in range(2*n,nmax+1,n):
                    if k in ntry:
                        ntry.remove(k)

    # Remove divisors.
    for n in tuple(nsym.keys()):
        for k in range(2*n,nmax+1,n):
            if k in nsym:
                del nsym[n]
                break

    c,n = max([(c,n) for n,c in nsym.items()]) if nsym else (None,None)
    return n

# -----------------------------------------------------------------------------
#
def correlation(v, xyz, w, axis, angle, center, rise = None):

    from chimerax.geometry import rotation, translation
    tf = rotation(axis, angle, center)
    if not rise is None:
        shift = translation([x*rise for x in axis])
        tf = shift * tf
    xf = v.scene_position * tf
    wtf = v.interpolated_values(xyz, xf)
    from chimerax.map_fit.fitmap import overlap_and_correlation
    olap, cor, corm = overlap_and_correlation(w, wtf)
    return cor, corm

# -----------------------------------------------------------------------------
#
def find_helix_symmetry(v, rise, angle, n = None, optimize = False, nmax = 8,
                        minimum_correlation = 0.99, maximum_points = 10000):

    axis = (0,0,1)
    centers, xyz, w = centers_and_points(v, maximum_points)
    xyz_min, xyz_max = v.xyz_bounds()
    zmin, zmax = xyz_min[2], xyz_max[2]
    zsize = zmax - zmin
    if zsize < 2*abs(rise):
        return None, 'Map z size must be at least twice helical rise'

    # Exclude points within rise of top of map so correlation is not
    # reduced by points falling outside map.
    inside = ((xyz[:,2] <= (zmax - rise)) if rise > 0
              else (xyz[:,2] >= (zmin + rise)))
    i = inside.nonzero()[0]
    xyz = xyz.take(i, axis = 0)
    w = w.take(i, axis = 0)

    if optimize:
        rise, angle, corr = \
            optimize_helix_paramters(v, xyz, w, axis, rise, angle, centers[0])
        if corr < minimum_correlation:
            return None, 'Optimizing helical parameters failed. Self-correlation %.4g < %.4g.' % (corr, minimum_correlation)

    if n is None:
        n = max(1, int(round(zsize / rise)))

    syms = None
    for center in centers:
        c, cm = correlation(v, xyz, w, axis, angle, center, rise)
        if c >= minimum_correlation:
            from chimerax.geometry import helical_symmetry_matrices
            syms = helical_symmetry_matrices(rise, angle, axis, center, n)
            break

    if syms is None:
        msg = 'No symmetry found for %s' % v.name
        return syms, msg

    # Look for z-axis cyclic symmetry.
    Cn = cyclic_symmetry(nmax, v, (0,0,1), center, xyz, w,
                         minimum_correlation)
    if not Cn is None:
        from chimerax.geometry import cyclic_symmetry_matrices
        csyms = cyclic_symmetry_matrices(Cn, center)
        syms = syms * csyms

    icenter = [round(i) for i in v.data.xyz_to_ijk(center)]
    cyclic = (' C%d' % Cn) if Cn else ''
    msg = ('Symmetry %s: Helix%s, rise %.4g, angle %.4g, center %s, n %d' %
           (v.name, cyclic, rise, angle, '%d %d %d' % tuple(icenter), n))

    return syms, msg

# -----------------------------------------------------------------------------
#
def optimize_helix_paramters(v, xyz, w, axis, rise, angle, center):

    m = v.full_matrix()
    xyz_to_ijk_transform = v.data.xyz_to_ijk_transform
    max_steps = 2000
    ijk_step_size_min, ijk_step_size_max = 0.01, 0.5
    optimize_translation = optimize_rotation = True
    metric = 'correlation'

    from chimerax.geometry import helical_symmetry_matrix
    htf = helical_symmetry_matrix(rise, angle, axis, center)
    tf = xyz_to_ijk_transform * htf

    from chimerax.map_fit.fitmap import locate_maximum
    move_tf, stats = locate_maximum(xyz, w, m, tf, max_steps,
                                    ijk_step_size_min, ijk_step_size_max,
                                    optimize_translation, optimize_rotation,
                                    metric)

    ohtf = htf * move_tf
    oaxis, ocenter, oangle, orise = ohtf.axis_center_angle_shift()
    from chimerax.geometry import inner_product
    if inner_product(axis,oaxis) < 0:
        orise = -orise
        oangle = -oangle
    corr = stats['correlation']

    return  orise, oangle, corr
