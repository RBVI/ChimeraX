# -----------------------------------------------------------------------------
# Tile planes of a volume so they are layout out on a grid.  This allows seeing
# many cross-sections at the same time.
#
# Allowed fill orders are strings like ulhr = upper-left-horizontal-reverse.
#   (u|l)(l|r)(h|v)[r] means (upper|lower)(left|right)(horz|vert)[reverse]
#
def tile_planes(v = None, axis = 'z', pstep = 1, trim = 0,
                rows = None, columns = None, fill_order = 'ulh',
                step = None, subregion = None, model_id = None,
                open = True):

  if v is None:
    from VolumeViewer import active_volume
    v = active_volume()
    if v is None:
      return

  vreg = v.subregion(step = step, subregion = subregion)
  reg = [list(ijk) for ijk in vreg]
  ac,ar,a = {'x':(1,2,0), 'y':(2,0,1), 'z':(0,1,2)}[axis]
  reg[0][a] += trim
  reg[1][a] -= trim
  reg[2][a] = pstep
  dorigin, dstep = v.region_origin_and_step(reg)
  
  m = v.region_matrix(reg)

  tcount = m.shape[2-a]
  if tcount == 0:
    return

  if rows is None and columns is None:
    # Choose columns to make square aspect ratio.
    w,h = m.shape[2-ac]*dstep[ac], m.shape[2-ar]*dstep[ar]
    from math import sqrt, ceil
    columns = min(tcount, int(ceil(sqrt(tcount*float(h)/w))))
    rows = (tcount - 1 + columns) / columns
  elif rows is None:
    rows = (tcount - 1 + columns) / columns
  elif columns is None:
    columns = (tcount - 1 + rows) / rows

  s0, s1, s2 = m.shape
  if axis == 'z': tshape = (1,rows*s1,columns*s2)
  elif axis == 'y': tshape = (columns*s0,1,rows*s2)
  elif axis == 'x': tshape = (rows*s0,columns*s1,1)
  from numpy import zeros
  ta = zeros(tshape, m.dtype)

  for i in range(tcount):
    # Start with top image in upper left corner.
    p,r,c = tile_position(i, rows, columns, tcount, fill_order)
    if axis == 'z':
      ta[0,r*s1:(r+1)*s1,c*s2:(c+1)*s2] = m[p,:,:]
    elif axis == 'y':
      ta[c*s0:(c+1)*s0,0,r*s2:(r+1)*s2] = m[:,p,:]
    elif axis == 'x':
      ta[r*s0:(r+1)*s0,c*s1:(c+1)*s1,0] = m[:,:,p]

  from VolumeData import Array_Grid_Data
  td = Array_Grid_Data(ta, dorigin, dstep)
  td.name = v.name + ' tiled %s' % axis
  from VolumeViewer import volume_from_grid_data
  tv = volume_from_grid_data(td, show_data = False, model_id = model_id,
                             open_model = open, show_dialog = open)
  tv.copy_settings_from(v, copy_region = False, copy_active = False,
                        copy_xform = open)
  if open:
    tv.show()
    v.unshow()          # Hide original map

  return tv

# -----------------------------------------------------------------------------
#
def tile_position(i, rows, columns, tcount, fill_order):

  if len(fill_order) > 3 and fill_order[3] == 'r':
    p = tcount-1-i
  else:
    p = i
  if fill_order[2] == 'h':
    c,r = i % columns, i / columns
  else:
    c,r = i / rows, i % rows
  if fill_order[0] == 'u':
    r = rows - 1 - r
  if fill_order[1] == 'r':
    c = columns - 1 - c

  return p,r,c
