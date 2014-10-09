# -----------------------------------------------------------------------------
#
def find_point_symmetry(v, nMax = 8, minimum_correlation = 0.99,
                        maximum_points = 10000):

    centers, xyz, w = centers_and_points(v, maximum_points)
    for center in centers:
        descrip, syms = point_symmetry(v, center, xyz, w,
                                       minimum_correlation, nMax)
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
    lev = min(v.surface_levels)
    import _volume
    ijk = _volume.high_indices(m, lev)
    ijk_mean = ijk.mean(axis = 0)

    # Test at most maximum_points in correlation calculation to make it faster.
    if not maximum_points is None and len(ijk) > maximum_points:
        from numpy.random import randint
        r = randint(len(ijk), size = maximum_points)
        ijk = ijk[r,:]

    from numpy import float32
    xyz = ijk.astype(float32)
    d = v.data
    import Matrix as M
    M.transform_points(xyz, d.ijk_to_xyz_transform)
    w = v.interpolated_values(xyz)

    icenter = [round(i) for i in ijk_mean]
    center = d.ijk_to_xyz(icenter)
    centers = [center]
    for offset in (0, -1):
        icenter = [(n-1)/2 if n%2 else (n/2)+offset for n in d.size]
        center = d.ijk_to_xyz(icenter)
        if not center in centers:
            centers.append(center)

    return centers, xyz, w

# -----------------------------------------------------------------------------
#
def point_symmetry(v, center, xyz, w, minimum_correlation = 0.99, nmax = 8):

    import Symmetry as S

    # Look for icosahedral symmetry.
    csname = icosahedral_symmetry(v, center, xyz, w, minimum_correlation)
    if csname:
        syms = S.icosahedral_symmetry_matrices(csname, center)
        return ('Icosahedral %s' % csname), syms

    # Look for z-axis cyclic symmetry.
    n = cyclic_symmetry(nmax, v, (0,0,1), center, xyz, w, minimum_correlation)
    if n is None:
        return None, None

    # Check for octahedral symmetry.
    if n == 4:
        c4x, cm = correlation(v, xyz, w, (1,0,0), 90, center)
        if c4x >= minimum_correlation:
            syms = S.octahedral_symmetry_matrices(center)
            return 'Octahedral', syms

    # Check for tetrahedral symmetry 3-fold on z, EMAN convention.
    if n == 3:
        from math import sqrt
        a3yz = (0,-2*sqrt(2)/3,-1.0/3)  # 3-fold in yz plane
        c3yz, cm = correlation(v, xyz, w, a3yz, 120, center)
        if c3yz >= minimum_correlation:
            syms = S.tetrahedral_symmetry_matrices('z3', center)
            return 'Tetrahedral z3', syms

    # Check for tetrahedral symmetry, 2-folds on x,y,z.
    c2x, cm = correlation(v, xyz, w, (1,0,0), 180, center)
    c2y, cm = correlation(v, xyz, w, (0,1,0), 180, center)
    if n == 2 and c2x >= minimum_correlation and c2y >= minimum_correlation:
        c3d, cm = correlation(v, xyz, w, (1,1,1), 120, center)
        if c3d >= minimum_correlation:
            syms = S.tetrahedral_symmetry_matrices('222', center)
            return 'Tetrahedral', syms
        
    # Check for dihedral symmetry, x axis flip.
    if c2x >= minimum_correlation:
        syms = S.dihedral_symmetry_matrices(n, center)
        return 'D%d' % n, syms

    # Check for dihedral symmetry, y axis flip.
    if c2y >= minimum_correlation:
        import Matrix as M
        syms = M.coordinate_transform_list(S.dihedral_symmetry_matrices(n),
                                           M.rotation_transform((0,0,1), -90))
        syms = S.recenter_symmetries(syms, center)
        return 'D%d, y flip' % n, syms

    syms = S.cyclic_symmetry_matrices(n, center)
    return ('C%d' % n), syms

# -----------------------------------------------------------------------------
#
def icosahedral_symmetry(v, center, xyz, w, minimum_correlation):

    import Icosahedron as I
    a23, a25, a35 = I.icosahedron_angles()
    from math import sin, cos
    a2 = (0,0,1)
    a5 = (0, sin(a25), cos(a25))
    import Matrix as M
    for cs in I.coordinate_system_names:
        tf = I.coordinate_system_transform('222', cs)
        c2, cm2 = correlation(v, xyz, w, M.apply_matrix(tf, a2), 180, center)
        c5, cm5 = correlation(v, xyz, w, M.apply_matrix(tf, a5), 72, center)
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

    for n, c in sorted(nsym.items()):
        print n,c

    # Remove divisors.
    for n in tuple(nsym.keys()):
        for k in range(2*n,nmax+1,n):
            if k in nsym:
                del nsym[n]
                break

    for n, c in sorted(nsym.items()):
        print n,c

    c,n = max([(c,n) for n,c in nsym.items()]) if nsym else (None,None)
    return n

# -----------------------------------------------------------------------------
#
def correlation(v, xyz, w, axis, angle, center, rise = None):

    import Matrix as M
    tf = M.rotation_transform(axis, angle, center)
    if not rise is None:
        shift = M.translation_matrix([x*rise for x in axis])
        tf = M.multiply_matrices(shift, tf)
    vxf = v.openState.xform
    xf = M.chimera_xform(M.multiply_matrices(M.xform_matrix(vxf), tf))
    wtf = v.interpolated_values(xyz, xf)
    import FitMap
    olap, cor, corm = FitMap.overlap_and_correlation(w, wtf)
    return cor, corm

# -----------------------------------------------------------------------------
#
def find_helix_symmetry(v, rise, angle, n = None, optimize = False, nmax = 8,
                        minimum_correlation = 0.99, maximum_points = 10000):

    axis = (0,0,1)
    centers, xyz, w = centers_and_points(v, maximum_points)
    zmin, zmax = zip(*v.xyz_bounds())[2]
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
            import Symmetry as S
            syms = S.helical_symmetry_matrices(rise, angle, axis, center, n)
            break

    if syms is None:
        msg = 'No symmetry found for %s' % v.name
        return syms, msg

    # Look for z-axis cyclic symmetry.
    Cn = cyclic_symmetry(nmax, v, (0,0,1), center, xyz, w,
                         minimum_correlation)
    if not Cn is None:
        import Symmetry as S
        csyms = S.cyclic_symmetry_matrices(Cn, center)
        import Matrix as M
        syms = M.matrix_products(syms, csyms)

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

    import Symmetry as S
    htf = S.helical_symmetry_matrix(rise, angle, axis, center)
    import Matrix as M
    tf = M.multiply_matrices(xyz_to_ijk_transform, htf)

    import FitMap as F
    move_tf, stats = F.locate_maximum(xyz, w, m, tf, max_steps,
                                      ijk_step_size_min, ijk_step_size_max,
                                      optimize_translation, optimize_rotation,
                                      metric)

    ohtf = M.multiply_matrices(htf, move_tf)
    oaxis, ocenter, oangle, orise = M.axis_center_angle_shift(ohtf)
    if M.inner_product(axis,oaxis) < 0:
        orise = -orise
        oangle = -oangle
    corr = stats['correlation']

    return  orise, oangle, corr
