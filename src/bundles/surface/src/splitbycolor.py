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
#
def surface_split_by_color(session, surfaces):
    '''
    Copy each colored part of a surface.  All triangles of one part have the same color.
    If some triangles have 3 vertices of different colors make a separate "multicolor" part.
    Masked parts of the surface are included and the masking is preserved in the copies.
    Each copied part has all vertices from the original surface (they are not culled).
    '''
    split_surfs = []
    for surface in surfaces:
        tlists, tmulticolor = _triangles_by_color(surface.triangles, surface.vertex_colors)
        if tlists or tmulticolor:
            from chimerax.core.models import Model
            g = Model(surface.name + ' split', session)
            tlists.sort(key = lambda t: -len(t))  # Largest f
            csurfs = [_copy_surface_triangles(surface, tlist, name = 'piece %d' % (i+1),
                                              single_color = True)
                      for i,tlist in enumerate(tlists)]
            if tmulticolor:
                csurfs.append(_copy_surface_triangles(surface, tmulticolor, name = 'multicolor'))
            g.add(csurfs)
            split_surfs.append(g)
    session.models.add(split_surfs)

# -----------------------------------------------------------------------------
#
def _triangles_by_color(triangles, vertex_colors):
    '''
    Group triangles by color.  All 3 vertices must have the same color.
    Return lists of triangle indices for triangles with the same color
    and a list of triangle indices for triangles that are multi-colored.
    '''
    
    if triangles is None or len(triangles) == 0:
        return [], []
        
    if vertex_colors is None:
        from numpy import arange, int32
        tlists = [arange(len(triangles), dtype = int32)]
        tmulticolor = []
    else:
        tc = {}
        mc = []
        for t,(v0,v1,v2) in enumerate(triangles):
            c0,c1,c2 = [vertex_colors[v,:3] for v in (v0,v1,v2)]
            if (c0 == c1).all() and (c0 == c2).all():
                c = tuple(c0)
                if c in tc:
                    tc[c].append(t)
                else:
                    tc[c] = [t]
            else:
                mc.append(t)
        tlists = list(tc.values())
        tmulticolor = mc
    return tlists, tmulticolor

# -----------------------------------------------------------------------------
#
def _copy_surface_triangles(surface, triangle_indices, name = None,
                            single_color = False):
    '''
    Copy a subset of triangles from a surface, making a new Surface model.
    The triangle mask is preserved.  All vertices of the original surface
    are in the copy (they are not culled).
    '''
    if name is None:
        name = surface.name + ' copy'
    from chimerax.core.models import Surface
    surf = Surface(name, surface.session)
    ta = surface.triangles[triangle_indices,:]
    surf.set_geometry(surface.vertices, surface.normals, ta)
    if surface.triangle_mask is not None:
        surf.triangle_mask = surface.triangle_mask[triangle_indices]

    surf.color = surface.color
    if single_color:
        if surface.vertex_colors is not None and len(triangle_indices) > 0:
            surf.color = surface.vertex_colors[ta[0,0]]
    else:
        surf.vertex_colors = surface.vertex_colors

    return surf

# -----------------------------------------------------------------------------
#
def register_splitbycolor_command(logger):
    from chimerax.core.commands import CmdDesc, register
    from chimerax.core.commands import SurfacesArg
    desc = CmdDesc(
        required=[('surfaces', SurfacesArg)],
        synopsis='Split a surface by color'
    )
    register('surface splitbycolor', desc, surface_split_by_color, logger=logger)
