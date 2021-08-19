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

# -----------------------------------------------------------------------------
# Angle arguments must be in radians.
#
def space_group_matrices(space_group, a, b, c, alpha, beta, gamma):

    from . import space_groups
    sgtable = space_groups.space_group_symmetry_table()
    sgu = space_group.upper()
    if not sgu in sgtable:
        return []
    unit_cell_matrices = sgtable[sgu]
    
    r2r_symops = []
    u2r = unit_cell_to_xyz_matrix(a, b, c, alpha, beta, gamma)
    r2u = u2r.inverse()

    for u2u in unit_cell_matrices:
        r2r = u2u if u2u.is_identity() else u2r * u2u * r2u
        r2r_symops.append(r2r)

    from chimerax.geometry import Places
    return Places(r2r_symops)

# -----------------------------------------------------------------------------
# Angle arguments must be in radians.
#
def unit_cell_axes(a, b, c, alpha, beta, gamma):

  from math import sin, cos, sqrt
  cg = cos(gamma)
  sg = sin(gamma)
  cb = cos(beta)
  ca = cos(alpha)
  c1 = (ca - cb*cg)/sg

  axes = ((a, 0, 0),
          (b*cg, b*sg, 0),
          (c*cb, c*c1, c*sqrt(1 - cb*cb - c1*c1)))

  return axes

# -----------------------------------------------------------------------------
# Angle arguments must be in radians.
#
def unit_cell_skew(alpha, beta, gamma):

  from math import sin, cos, sqrt
  cg = cos(gamma)
  sg = sin(gamma)
  cb = cos(beta)
  ca = cos(alpha)
  c1 = (ca - cb*cg)/sg

  skew = ((1, cg, cb),
          (0, sg, c1),
          (0, 0, sqrt(1 - cb*cb - c1*c1)))

  return skew

# -----------------------------------------------------------------------------
# Angle arguments must be in radians.
#
def unit_cell_to_xyz_matrix(a, b, c, alpha, beta, gamma):

    axes = unit_cell_axes(a, b, c, alpha, beta, gamma)
    from chimerax.geometry import Place
    return Place(axes = axes)

# -----------------------------------------------------------------------------
# Compute the corner position of unit cell containing a specified point.
# The grid_origin is specified in crystallographic coords (ie the axes basis).
#
def cell_origin(grid_origin, axes, interior_point):

    from numpy import array, transpose, subtract, floor, add
    from numpy import dot as matrix_multiply
    from numpy.linalg import inv as invert_matrix
    c2rt = array(axes)
    c2r = transpose(c2rt)                # Crystal coords to real coords
    r2c = invert_matrix(c2r)             # Real coords to crystal coords
    ip_c = matrix_multiply(r2c, interior_point) # To crystal coords.
    ipo_c = subtract(ip_c, grid_origin)  # Relative to grid origin.
    co_c = floor(ipo_c)                  # Offset to containing cell.
    o_c = add(grid_origin, co_c)         # Origin of containing cell.
    co = matrix_multiply(c2r, o_c)       # To real coords.

    return tuple(co)
    
# ---------------------------------------------------------------------------
# Find center of cell in unit cell grid containing specified point.
#
def cell_center(grid_origin, axes, interior_point):

    origin = cell_origin(grid_origin, axes, interior_point)
    from numpy import add
    asum = add(axes[0], axes[1])
    add(asum, axes[2], asum)
    c = add(origin, .5 * asum)
    return tuple(c)

# -----------------------------------------------------------------------------
#
def close_packing_matrices(tflist, ref_point, center,
                           a, b, c, alpha, beta, gamma):

  u2r = unit_cell_to_xyz_matrix(a, b, c, alpha, beta, gamma)
  r2u = u2r.inverse()

  stflist = []
  from chimerax.geometry import translation
  from numpy import array, subtract, add
  for tf in tflist:
#    shift = u2r * -map(int, r2u * (tf * center - center) + (.5,.5,.5))
    tfc = tf * ref_point
    csep = subtract(tfc, center)
    ucsep = r2u * csep
    import math
    ushift = [math.floor(x) for x in add(ucsep, (.5,.5,.5))]
    shift = u2r * ushift
    neg_shift = [-x for x in shift]
    tfshift = translation(neg_shift)
    stf = tfshift * tf
    stflist.append(stf)

  return stflist

# -----------------------------------------------------------------------------
# Return a list of symmetry matrices by adding translations to tflist matrices
# so that they map ref_point into the unit cell box containing ref_point.
# The origin of the unit cell grid is given by grid_origin.
#
def pack_unit_cell(uc, grid_origin, ref_point, tflist):

    a, b, c, alpha, beta, gamma = uc
    axes = unit_cell_axes(a, b, c, alpha, beta, gamma)
    center = cell_center(grid_origin, axes, ref_point)
    tflist = close_packing_matrices(tflist, ref_point, center,
                                    a, b, c, alpha, beta, gamma)
    return tflist

# -----------------------------------------------------------------------------
#
def matrix_products(m1, m2, group = False):

  if group:
    if is_transform(m1):
      if is_transform(m2):
        return m1 * m2
      else:
        return [matrix_products(m1,m,group) for m in m2]
    else:
      return [matrix_products(m,m2,group) for m in m1]
  else:
    return [mat1*mat2 for mat1 in m1 for mat2 in m2]

# -----------------------------------------------------------------------------
#
def is_transform(m):

  return isinstance(m[0][0], (float,int))

# -----------------------------------------------------------------------------
# Identity matrix is first in list.
#
def translation_matrices(cell_axes, tranges):

  from chimerax.geometry import translation
  (xmin, xmax), (ymin, ymax), (zmin, zmax) = tranges
  mlist = []
  for z in range(zmin, zmax+1):
    for y in range(ymin, ymax+1):
      for x in range(xmin, xmax+1):
        t = [0,0,0]
        for a in range(3):
          t[a] = x*cell_axes[0][a] + y*cell_axes[1][a] + z*cell_axes[2][a]
        m = translation(t)
        if (x,y,z) == (0,0,0):
          # Put the 0 translation first in the list.
          mlist.insert(0, m)
        else:
          mlist.append(m)
  return mlist

# -----------------------------------------------------------------------------
#
def unit_cell_translations(uc, oc, nc, tflist, group = False):

  ranges = [(o,o+n-1) for o,n in zip(oc,nc)]
  cell_axes = unit_cell_axes(*uc)
  mlist = translation_matrices(cell_axes, ranges)
  tlist = matrix_products(mlist, tflist, group)
  return tlist

# -----------------------------------------------------------------------------
# Combine crystal and non-crystal symmetry matrices.
#
def unit_cell_matrices(slist, mlist, uc = None, pack = None, group = False):

  if slist and mlist:
    smlist = matrix_products(slist, mlist, group)
  elif slist:
    smlist = slist
  elif mlist:
    smlist = mlist
  else:
    from chimerax.geometry import Place
    smlist = [Place()]

  if pack:
      ref_point, grid_origin = pack
      if group:
        smlist = [pack_unit_cell(uc, grid_origin, ref_point, sm)
                  for sm in smlist]
      else:
        smlist = pack_unit_cell(uc, grid_origin, ref_point, smlist)
  return smlist
