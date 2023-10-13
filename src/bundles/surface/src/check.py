# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2022 Regents of the University of California. All rights reserved.
# This software is provided pursuant to the ChimeraX license agreement, which
# covers academic and commercial uses. For more information, see
# <http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
#
# This file is part of the ChimeraX library. You can also redistribute and/or
# modify it under the GNU Lesser General Public License version 2.1 as
# published by the Free Software Foundation. For more details, see
# <https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html>
#
# This file is distributed WITHOUT ANY WARRANTY; without even the implied
# warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. This notice
# must be embedded in or attached to all copies, including partial copies, of
# the software or any revisions or derivations thereof.
# === UCSF ChimeraX Copyright ===

# -----------------------------------------------------------------------------
#
def surface_check(session, surfaces = None):

    if surfaces is None:
        from chimerax.core.models import Surface
        surfaces = session.models.list(type = Surface)
        
    lines = []
    for surface in surfaces:
        t = surface.triangles
        if t is not None:
            msg = surface_topology_check(t)
            if not msg:
                msg = 'ok'
            lines.append('%s #%s %d triangles %s'
                         % (surface.name, surface.id_string, len(t), msg))
    session.logger.info('\n'.join(lines))
    
# -----------------------------------------------------------------------------
#
def surface_topology_check(triangles):
    '''
    Check if surface is a manifold without boundary.
    This checks that each edge has exactly two triangles oriented in opposite directions
    (ie. the surface does not have a boundary and does not self-intersect),
    and that each vertex has a single fan of triangles (ie. that two parts of the
    surface do not contact at a point).
    '''
    vmap = {}
    for t,vt in enumerate(triangles):
        for v in vt:
            if v in vmap:
                vmap[v].append(t)
            else:
                vmap[v] = [t]

    edge_dup = 0
    boundary = 0
    vertex_contacts = 0
    for v, tlist in vmap.items():
        emap = {}
        for t in tlist:
            v1,v2,v3 = triangles[t]
            if v1 == v:
                va,vb = (v2,v3)
            elif v2 == v:
                va,vb = (v3,v1)
            elif v3 == v:
                va,vb = (v1,v2)
            if va in emap:
                # Two edges (v,va)
                edge_dup += 1
            emap[va] = vb
        for va,vb in emap.items():
            if vb not in emap:
                boundary += 1
        if _count_chains(emap) > 1:
            vertex_contacts += 1

    lines = []
    if edge_dup > 0:
        lines.append('%d edges intersections' % edge_dup)
    if boundary > 0:
        lines.append('%d boundary edges' % boundary)
    if vertex_contacts > 0:
        lines.append('%d point contacts' % vertex_contacts)
    return ', '.join(lines)
    
def _count_chains(emap):
    cmap = {}
    g = 1
    for va,vb in emap.items():
        ga, gb = cmap.get(va), cmap.get(vb)
        if ga and gb:
            if ga != gb:
                valist = [v for v,gv in cmap.items() if gv == ga]
                for v in valist:
                    cmap[v] = gb
        elif ga:
            cmap[vb] = ga
        elif gb:
            cmap[va] = gb
        else:
            cmap[va] = cmap[vb] = g
            g += 1
    return len(set(cmap.values()))

# -------------------------------------------------------------------------------------
#
def register_command(logger):
    from chimerax.core.commands import CmdDesc, register, SurfacesArg
    desc = CmdDesc(
        optional = [('surfaces', SurfacesArg)],
        synopsis = 'check surface topology')
    register('surface check', desc, surface_check, logger=logger)
