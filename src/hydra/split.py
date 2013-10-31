# -----------------------------------------------------------------------------
#
def split_selected_surfaces(in_place = True):

  import Surface
  plist = Surface.selected_surface_pieces()
  if plist:
    pplist = split_surfaces(plist, in_place)
    from chimera.replyobj import status
    status('%d surface pieces' % len(pplist))

# -----------------------------------------------------------------------------
#
def split_surfaces(plist, in_place = False):

  surf = None
  if not in_place:
    name = '%s split' % plist[0].surface.name if plist else 'split surface'
    from .surface import Surface
    surf = Surface(name)
    om.add([surf])
    from ..ui.gui import main_window
    main_window.view.add_model(surf)

  pplist = []
  for p in plist:
    pieces = split_surface_piece(p, surf or p.surface)
    pplist.extend(pieces)
    if pieces:
      # TODO: Select pieces if original surface selected.
      if in_place:
        p.surface.removePiece(p)
      else:
        p.display = False

  return pplist

# -----------------------------------------------------------------------------
#
def split_surface_piece(p, into_surf):

  varray, tarray = p.geometry
  from ._image3d import connected_pieces
  cplist = connected_pieces(tarray)
  if len(cplist) <= 1 and p.surface == into_surf:
    return []
  pplist = copy_surface_piece_blobs(p, varray, tarray, cplist, into_surf)
  return pplist

# -----------------------------------------------------------------------------
#
def copy_surface_piece_blobs(p, varray, tarray, cplist, into_surf):

  from numpy import zeros, int32
  vmap = zeros(len(varray), int32)

  pplist = []
  m = p.surface
  narray = p.normals
  color = p.color
  vrgba = p.vertex_colors
  temask = p.triangleAndEdgeMask
  for pi, (vi,ti) in enumerate(cplist):
    pp = copy_piece_blob(into_surf, varray, tarray, narray, color, vrgba, temask,
                         vi, ti, vmap)
    copy_piece_attributes(p, pp)
    pp.name = '%s %d' % (p.name, pi+1)
    pplist.append(pp)

  return pplist

# -----------------------------------------------------------------------------
#
def copy_piece_blob(m, varray, tarray, narray, color, vrgba, temask,
                     vi, ti, vmap):

  va = varray.take(vi, axis = 0)
  ta = tarray.take(ti, axis = 0)

  # Remap triangle vertex indices for shorter vertex list.
  from numpy import arange
  vmap[vi] = arange(len(vi), dtype = vmap.dtype)
  ta = vmap.take(ta.ravel()).reshape((len(ti),3))

  gp = m.newPiece()
  gp.geometry = va, ta
  gp.save_in_session = True

  na = narray.take(vi, axis = 0)
  gp.normals = na

  gp.color = color

  if not vrgba is None:
    gp.vertex_colors = vrgba.take(vi, axis = 0)
  if not temask is None:
    gp.triangleAndEdgeMask = temask.take(ti, axis = 0)

  return gp

# -----------------------------------------------------------------------------
#
def copy_piece_attributes(g, gp):

  gp.display = g.display
  gp.displayStyle = g.displayStyle
#  gp.useLighting = g.useLighting
#  gp.lineThickness = g.lineThickness
  gp.name = g.name
