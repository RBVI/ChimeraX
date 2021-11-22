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
# Command to create a molecule model from a surface mesh.
#
def meshmol(session, surfaces, radius = 1):

    marker_sets = []
    for s in surfaces:
        if getattr(s, 'is_clip_cap', False):
            continue
        edges = s.masked_edges
        if len(edges) == 0:
            continue

        from chimerax.markers import MarkerSet, create_link
        m = MarkerSet(session, 'Mesh ' + s.name)
        marker_sets.append(m)
        
        # Create markers at vertices
        varray = s.vertices
        vcolors = s.vertex_colors
        markers = {}
        for edge in edges:
            for v in edge:
                if not v in markers:
                    xyz = varray[v]
                    color = s.color if vcolors is None else vcolors[v]
                    markers[v] = m.create_marker(xyz, color, radius)

        # Create links between markers for edges
        for v1,v2 in edges:
            m1 = markers[v1]
            m2 = markers[v2]
            color = tuple((a+b)//2 for a,b in zip(m1.color, m2.color))
            create_link(m1, m2, color, radius)

    session.models.add(marker_sets)
    return marker_sets

# -----------------------------------------------------------------------------
#
def register_meshmol_command(logger):
    from chimerax.core.commands import CmdDesc, register, create_alias, FloatArg
    from chimerax.core.commands import SurfacesArg
    desc = CmdDesc(
        required = [('surfaces', SurfacesArg)],
        optional= [('radius', FloatArg)],
        synopsis='Create a marker model from a mesh'
    )
    register('surface meshmol', desc, meshmol, logger=logger)
    create_alias('meshmol', 'surface meshmol $*', logger=logger)
