# ---------------------------------------------------------------------------
#
def write_brix(grid_data, path, options = {}, progress = None):

    d = grid_data
    xyz_size = map(lambda a,b: a*b, d.step, d.size)
    origin = tuple([round(-x) for x in d.xyz_to_ijk((0,0,0))])
    size = d.size
    cell_size = d.size
    cell = tuple(xyz_size) + tuple(d.cell_angles)
    matrix = d.full_matrix()
    dmin = float(matrix.min())
    dmax = float(matrix.max())
    prod = 255.0/(dmax - dmin)
    plus = -int(dmin*prod)
    sigma = matrix.std()

    header_length = 512
    fields = (":-)",
              "origin%5d%5d%5d" % origin,
              "extent%5d%5d%5d" % size,
              "grid%5d%5d%5d" % cell_size,
              "cell %10.3f%10.3f%10.3f%10.3f%10.3f%10.3f" % cell,
              "prod%12.5f" % prod,
              "plus%8d" % plus,
              "sigma %12.5f" % sigma,
              )
    h = ' '.join(fields)
    h += ' ' * (header_length - len(h))

    f = open(path, 'wb')
    f.write(h)
    write_data(matrix, prod, plus, f, progress)
    f.close()
  
# ---------------------------------------------------------------------------
# Write data in 8 by 8 by 8 blocks of bytes.
#
def write_data(matrix, prod, plus, file, progress):

    block_size = 8
    bsize3 = block_size*block_size*block_size
    zsize, ysize, xsize = matrix.shape
    xblocks, yblocks, zblocks = [(s-1)/block_size+1
                                 for s in (xsize,ysize,zsize)]
    from numpy import zeros, uint8
    d = zeros((block_size,block_size,block_size), matrix.dtype)
    for zblock in range(zblocks):
      if progress:
        progress.fraction(float(zblock)/zblocks)
      z = block_size * zblock
      zbsize = min(z + block_size, zsize) - z
      for yblock in range(yblocks):
        y = block_size * yblock
        ybsize = min(y + block_size, ysize) - y
        for xblock in range(xblocks):
          x = block_size * xblock
          xbsize = min(x + block_size, xsize) - x
          d[:zbsize,:ybsize,:xbsize] = matrix[z:z+zbsize,y:y+ybsize,x:x+xbsize]
          d[zbsize:,ybsize:,xbsize:] = 0  # For fragment blocks.
          d1d = d.ravel()
          bytes = (d1d*prod + plus).astype(uint8)   # Scale to 0-255
          file.write(bytes.tostring())

    if progress:
      progress.done()
