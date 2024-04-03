# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2022 Regents of the University of California. All rights reserved.
# The ChimeraX application is provided pursuant to the ChimeraX license
# agreement, which covers academic and commercial uses. For more details, see
# <http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
#
# This particular file is part of the ChimeraX library. You can also
# redistribute and/or modify it under the terms of the GNU Lesser General
# Public License version 2.1 as published by the Free Software Foundation.
# For more details, see
# <https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html>
#
# THIS SOFTWARE IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND, EITHER
# EXPRESSED OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
# OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. ADDITIONAL LIABILITY
# LIMITATIONS ARE DESCRIBED IN THE GNU LESSER GENERAL PUBLIC LICENSE
# VERSION 2.1
#
# This notice must be embedded in or attached to all copies, including partial
# copies, of the software or any revisions or derivations thereof.
# === UCSF ChimeraX Copyright ===

# -----------------------------------------------------------------------------
#
def split_selected_surfaces(session, in_place = True):

  import Surface
  plist = Surface.selected_surface_pieces()
  if plist:
    pplist = split_surfaces(plist, session, in_place)
    from chimera.replyobj import status
    status('%d surface pieces' % len(pplist))

# -----------------------------------------------------------------------------
#
def split_surfaces(plist, session, in_place = False):

  surf = None
  if not in_place:
    name = '%s split' % plist[0].surface.name if plist else 'split surface'
    from chimerax.graphics import Drawing
    surf = Drawing(name)
    session.models.add_models([surf])

  pplist = []
  for p in plist:
    pieces = split_surface_piece(p, surf or p.surface)
    pplist.extend(pieces)
    if pieces:
      # TODO: Select pieces if original surface selected.
      if in_place:
        p.surface.remove_drawing(p)
      else:
        p.display = False

  return pplist

# -----------------------------------------------------------------------------
#
def split_surface_piece(p, into_surf):

  varray, tarray = p.vertices, p.triangles
  from ._surface import connected_pieces
  cplist = connected_pieces(tarray)
  if len(cplist) <= 1 and p.surface == into_surf:
    return []
  pplist = copy_surface_piece_blobs(p, varray, tarray, cplist, into_surf)
  return pplist

# -----------------------------------------------------------------------------
#
def reduce_geometry(va, na, ta, vi, ti):

  from numpy import zeros, int32
  vmap = zeros(len(va), int32)
  rva = va.take(vi, axis = 0)
  rna = na.take(vi, axis = 0)
  rta = ta.take(ti, axis = 0)
  # Remap triangle vertex indices to use shorter vertex list.
  from numpy import arange
  vmap[vi] = arange(len(vi), dtype = vmap.dtype)
  rta = vmap.take(rta.ravel()).reshape((len(ti),3))

  return rva, rna, rta

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
  temask = p.triangle_and_edge_mask
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
  na = narray.take(vi, axis = 0)
  ta = tarray.take(ti, axis = 0)

  # Remap triangle vertex indices for shorter vertex list.
  from numpy import arange
  vmap[vi] = arange(len(vi), dtype = vmap.dtype)
  ta = vmap.take(ta.ravel()).reshape((len(ti),3))

  gp = m.new_drawing('blob copy')
  gp.set_geometry(va, na, ta)
  gp.save_in_session = True

  gp.color = color

  if not vrgba is None:
    gp.vertex_colors = vrgba.take(vi, axis = 0)
  if not temask is None:
    gp.triangle_and_edge_mask = temask.take(ti, axis = 0)

  return gp

# -----------------------------------------------------------------------------
#
def copy_piece_attributes(g, gp):

  gp.display = g.display
  gp.display_style = g.display_style
#  gp.use_lighting = g.use_lighting
#  gp.lineThickness = g.lineThickness
  gp.name = g.name
