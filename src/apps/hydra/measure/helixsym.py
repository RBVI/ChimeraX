# -----------------------------------------------------------------------------
#
def find_helix_symmetry(v, rise = None, angle = None,
                        n = None, optimize = False, nmax = 8,
                        minimum_correlation = 0.99, maximum_points = 10000,
                        details = False):

    if rise is None:
        rise = find_helix_rise(v, minimum_correlation, details)
        if rise is None:
            return None, 'Could not determine helix z repeat'

    from symmetry import centers_and_points, correlation, cyclic_symmetry
    
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

    if n is None:
        n = max(1, int(round(zsize / rise)))

    syms = None
    optimized = False
    msg = None
    for center in centers:
        if angle is None:
            angle = find_helix_rotation(v, rise, center, minimum_correlation,
                                        details)
            if angle is None:
                msg = 'Could not determine helix twist.'
                continue
        if optimize and not optimized:
            r, a, corr = \
                  optimize_helix_paramters(v, xyz, w, axis, rise, angle, center)
            if corr < minimum_correlation:
                msg = 'Optimizing helical parameters failed. Self-correlation %.4g < %.4g.' % (corr, minimum_correlation)
                continue
            rise, angle = r, a
            optimized = False
        c, cm = correlation(v, xyz, w, axis, angle, center, rise)
        if c >= minimum_correlation:
            import Symmetry as S
            syms = S.helical_symmetry_matrices(rise, angle, axis, center, n)
            break

    if syms is None:
        if msg is None:
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

# -----------------------------------------------------------------------------
# Assumes helix axis is parallel to z.
# Looks for repeat in average density as function of z.
#
def find_helix_rise(v, minimum_correlation = 0.9, details = False):

    m = v.full_matrix()
    level = max(v.surface_levels)
#
# TODO: May want to threshold data so that large noise region outside
#       filament does not add noise to average z density.
#    m = m * (m >= level)       # This multiplys by value > 1. Needs fix.
#
    zdens = m.sum(axis = 2).sum(axis = 1)
    zdens -= zdens.mean()
    if details:
        plot_1d(zdens, 'average density', 'z')
    nz = len(zdens)
    zmid = zdens[nz/4:3*nz/4]
    from numpy import convolve, ones, sqrt
    zc = convolve(zdens, zmid[::-1], mode = 'valid')
    zd2 = zdens*zdens
    zn = sqrt(convolve(zd2, ones((len(zmid),), zd2.dtype), mode = 'valid'))
    zc /= zn
    zc /= sqrt((zmid*zmid).sum())
    if details:
        plot_1d(zc, 'self correlation', 'delta z')
    r = peak_spacing(zc, nz/4, minimum_correlation)
    if r is None:
#      print 'No z translation repeat'
      return None

    rise = r*v.data.step[2]
    return rise

# -----------------------------------------------------------------------------
#
def find_helix_rotation(v, rise, center = (0,0,0), minimum_correlation = 0.9,
                        details = False):

    # TODO: Should add multiple radii to make sure some circles are inside
    #       filament.  Try 0.2, 0.4, 0.6, 0.8 times center to box face.
    r = 0.25*min(s*gs for s,gs in zip(v.data.step, v.data.size)[:2])
    from math import ceil, pi
    n = max(10, ceil(2*pi*r / min(v.data.step)))
    cv0 = circle_values(v, r, center, -0.5*rise, n)
    cv0 -= cv0.mean()
    cv1 = circle_values(v, r, center, 0.5*rise, n)
    cv1 -= cv1.mean()
    from numpy import convolve, sqrt, concatenate
    cc = convolve(concatenate((cv0,cv0)), cv1[::-1], mode = 'valid')
    cn = sqrt((cv0*cv0).sum()*(cv1*cv1).sum())
    cc /= cn
    if details:
        plot_1d(cc, 'circle correlation', 'rotation angle')

    i0 = next_peak(cc, 0, minimum_correlation)
    i1 = next_peak(cc[::-1], 0, minimum_correlation)
    if i0 is None or i1 is None:
#      print 'No rotation peak'
      return None

    a = 360*float(min(i0,i1))/n
#    print 'rotation', a
    return a

# -----------------------------------------------------------------------------
# Look at midpoints of intervals of 1-d array c with values above min_c.
# Return the spacing between interval midpoint for index i and next interval
# midpoint to right or to left.
#
def peak_spacing(c, i, min_c):

    ct = (c >= min_c)
    if not ct[i]:
      return None
    i1,i2 = nonzero_range(ct, i)
    imid = 0.5 * (i1 + i2-1)

    from numpy import logical_not
    ctn = logical_not(ct)
    j1,j2 = nonzero_range(ctn, i2)
    k1,k2 = nonzero_range(ct, j2)
    if k2 > k1:
      imid1 = 0.5 * (k1 + k2-1)
      return imid1 - imid

    j1,j2 = nonzero_range(ctn, i1-1)
    k1,k2 = nonzero_range(ct, j1-1)
    if k2 > k1:
      imid0 = 0.5 * (k1 + k2-1)
      return imid - imid0

    return None

# -----------------------------------------------------------------------------
#
def nonzero_range(a, i):

  n = len(a)
  if i < 0 or i >= n:
    return i,i
  i0 = i1 = i
  while i1 >= 0 and i1 < n and a[i1]:
    i1 += 1
  while i0 >= 0 and i0 < n and a[i0]:
    i0 -= 1
  return i0+1,i1

# -----------------------------------------------------------------------------
#
def next_peak(c, i, min_c):

  n = len(c)
  i0 = i
  while i0 < n and c[i0] < min_c:
    i0 += 1
  i1 = i0
  while i1 < n and c[i1] >= min_c:
    i1 += 1
  imid = 0.5*(i0 + i1-1) if i1 > i0 else None
  return imid

# -----------------------------------------------------------------------------
#
def circle_values(v, r, center, zoffset, n):

  from numpy import arange, float32, pi, array, cos, sin, zeros
  a = arange(n, dtype = float32)*(2*pi/n)
  xyz = array((r*cos(a), r*sin(a), zeros((n,),float32)), float32).transpose()
  xyz[:,:] += center
  xyz[:,2] += zoffset
  cv = v.interpolated_values(xyz)
  return cv
  
# -----------------------------------------------------------------------------
#
def plot_1d(d, yname, xname):

    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.set_xlabel(xname)
    ax.set_ylabel(yname)
    ax.plot(d, linewidth=1.0)
    fig.canvas.manager.show()
