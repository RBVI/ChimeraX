# -----------------------------------------------------------------------------
# Extend an asymmetric unit map using crystal symmetry.
#
def map_covering_box(v, ijk_min, ijk_max, ijk_cell_size, symmetries, step):

  d = v.data
  if ijk_cell_size == d.size and len(symmetries) == 0:
    # Full unit cell and no symmetries to average.
    g = v.grid_data(subregion = 'all', step = step, mask_zone = False)
    from VolumeViewer import volume
    cg = volume.map_from_periodic_map(g, ijk_min, ijk_max)
    return cg

  out_ijk_size = tuple(a-b+1 for a,b in zip(ijk_max, ijk_min))
  import Matrix
  out_ijk_to_vijk_transform = Matrix.translation_matrix(ijk_min)

  ijk_symmetries = ijk_symmetry_matrices(d, symmetries)

  from numpy import empty, float32
  shape = list(reversed(out_ijk_size))
  m = empty(shape, float32)

  from _volume import extend_crystal_map
  nnc, dmax = extend_crystal_map(v.full_matrix(), ijk_cell_size,
                                 ijk_symmetries, m, out_ijk_to_vijk_transform)

  from chimera import replyobj as ro
  ro.info('Extended map %s to box of size (%d, %d, %d),\n'
          % ((v.name,) + out_ijk_size) +
          '  cell size (%d, %d, %d) grid points, %d symmetry operations,\n'
          % (tuple(ijk_cell_size) + (len(ijk_symmetries),)) +
          '  %d points not covered by any symmetry,\n' % nnc +
          '  maximum value difference where symmetric map copies overlap = %.5g\n'
          % dmax)
  if nnc > 0:
    ro.status('%d grid points not covered' % nnc)

  origin = d.ijk_to_xyz(ijk_min)
  from VolumeData import Array_Grid_Data
  g = Array_Grid_Data(m, origin, d.step, d.cell_angles, name = v.name + ' extended')

  return g

# -----------------------------------------------------------------------------
# This routine is available in C++ in _volume.
#
def extend_crystal_map(volarray, ijk_cell_size, ijk_symmetries,
                       out_array, out_ijk_to_vol_ijk_transform):

  ksz, jsz, isz = out_array.shape
  otf = out_ijk_to_vol_ijk_transform
  from Matrix import identity_matrix, apply_matrix
  itf = identity_matrix()
  nsym = len(ijk_symmetries)
  from numpy import empty, float32, delete as delete_array_elements
  values = empty((nsym,), float32)
  nnc = 0       # Number of grid points not covered by symmetry
  dmax = None   # Maximum value discrepancy for multiple overlaps
  from _interpolate import interpolate_volume_data
  for k in range(ksz):
    for j in range(jsz):
      for i in range(isz):
        vijk = apply_matrix(otf, (i,j,k))
        vlist = []
        for sym in ijk_symmetries:
          svijk = apply_matrix(sym, vijk)
          scijk = [p % s for p,s in zip(svijk, ijk_cell_size)] # Wrap to unit cell.
          vlist.append(scijk)
        values[:] = 0
        outside = interpolate_volume_data(vlist, itf, volarray, 'linear', values)[1]
        n = len(values) - len(outside)
        if n >= 1:
          out_array[k,j,i] = float(values.sum()) / n
          if n >= 2:
            # Record maximum value discrepancy for multiple overlaps
            vinside = delete_array_elements(values, outside)
            d = vinside.max() - vinside.min()
            if dmax is None or d > dmax:
              dmax = d
        else:
          nnc += 1       # Count grid points not covered by symmetry.
  
  return nnc, dmax


# -----------------------------------------------------------------------------
#
def ijk_symmetry_matrices(data, symmetries):

  import Matrix

  if len(symmetries) == 0:
    return [Matrix.identity_matrix()]

  x2i = data.xyz_to_ijk_transform
  i2x = data.ijk_to_xyz_transform
  isyms = [Matrix.multiply_matrices(x2i, x2x, i2x) for x2x in symmetries]

  return isyms

# -----------------------------------------------------------------------------
#
def cover_box_bounds(volume, step, atoms, pad, box, fBox, iBox):

    grid = volume.data
    if atoms:
        from VolumeViewer.volume import atom_bounds
        ijk_min, ijk_max = atom_bounds(atoms, pad, volume)
    elif box:
        origin, gstep = grid.origin, grid.step
        ijk_min, ijk_max = [[(None if x is None else (x-o)/s)
                             for x,o,s in zip(xyz, origin, gstep)]
                            for xyz in box]
    elif iBox:
        ijk_min, ijk_max = iBox
    elif fBox:
        size = grid.size
        ijk_min = [(None if f is None else f*s) for f, s in zip(fBox[0],size)]
        ijk_max = [(None if f is None else f*s-1) for f, s in zip(fBox[1],size)]

    # Fill in unspecified dimensions.
    ijk_min = [(0 if i is None else i) for i in ijk_min]
    ijk_max = [(s-1 if i is None else i) for i,s in zip(ijk_max,grid.size)]

    # Integer bounds.
    from math import floor, ceil
    ijk_min = [int(floor(i)) for i in ijk_min]
    ijk_max = [int(ceil(i)) for i in ijk_max]

    # Handle step size > 1.
    ijk_min = [i/s for i,s in zip(ijk_min,step)]
    ijk_max = [i/s for i,s in zip(ijk_max,step)]

    return ijk_min, ijk_max
