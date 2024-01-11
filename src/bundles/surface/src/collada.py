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

def read_collada_surfaces(session, path_or_stream, name = None, color = (200,200,200,255), **kw):
    '''Open a collada file.'''

    from collada import Collada
    if hasattr(path_or_stream, 'read'):
        stream = path_or_stream
        path = stream.name
        c = Collada(stream)
    else:
        path = path_or_stream
        c = Collada(path)

    if name is None:
        from os.path import basename
        name = basename(path)
    from chimerax.geometry import Place
    splist = surfaces_from_nodes(c.scene.nodes, color, Place(), {}, session)
    if len(splist) > 1:
        from chimerax.core.models import Model
        s = Model(name, session)
        s.add(splist)
    elif len(splist) == 1:
        s = splist[0]
        s.name = name
    else:
        from chimerax.core.errors import UserError
        raise UserError('Collada file has no TriangleSets: %s' % name)
    set_instance_positions_and_colors(s.all_drawings())

    ai = c.assetInfo
    if ai:
        s.collada_unit_name = ai.unitname
        s.collada_contributors = ai.contributors

    return [s], ('Opened collada file %s' % name)

def surfaces_from_nodes(nodes, color, place, instances, session):

    from collada.scene import GeometryNode, Node
    from chimerax.geometry import Place
    from chimerax.core.models import Surface
    splist = []
    for n in nodes:
        if isinstance(n, GeometryNode):
            materials = dict((m.symbol,m.target) for m in n.materials)
            g = n.geometry
            colors = g.sourceById
            if g.id in instances:
                add_geometry_instance(g.primitives, instances[g.id], place, color, materials)
            else:
                instances[g.id] = spieces = geometry_node_surfaces(g.primitives, place, color, materials, colors, session)
                splist.extend(spieces)
        elif isinstance(n, Node):
            pl = place * Place(n.matrix[:3,:])
            spieces = surfaces_from_nodes(n.children, color, pl, instances, session)
            name = n.xmlnode.get('name')
            if len(spieces) > 1:
                m = Surface(name, session)
                m.add(spieces)
                splist.append(m)
            elif len(spieces) == 1:
                s = spieces[0]
                s.name = name
                splist.append(s)
    return splist

def geometry_node_surfaces(primitives, place, color, materials, colors, session):

    from collada import polylist, triangleset

    splist = []
    for p in primitives:
        if isinstance(p, polylist.Polylist):
            p = p.triangleset()
        if not isinstance(p, triangleset.TriangleSet):
            continue        # Skip line sets.

        t = p.vertex_index            # N by 3 array of vertex indices for triangles
        t = t.copy()                  # array from pycollada is not contiguous.
        v = p.vertex                  # M by 3 array of floats for vertex positions
        ni = p.normal_index           # N by 3 array of normal indices for triangles
        n = p.normal		      # M by 3 array of floats for vertex normals

        # Collada allows different normals on the same vertex in different triangles,
        # but Hydra only allows one normal per vertex.
        if n is None:
            vn = None
        else:
            from numpy import empty
            vn = empty(v.shape, n.dtype)
            vn[t.ravel(),:] = n[ni.ravel(),:]

        vcolors = vertex_colors(p, t, len(v), colors)
        c = material_color(materials.get(p.material), color)

        name = '%d' % (len(splist) + 1)
        from chimerax.core.models import Surface
        sp = Surface(name, session)
        sp.SESSION_SAVE_DRAWING = True	# Save triangles in .cxs session files
        sp.set_geometry(v, vn, t)
        sp.color_list = [c]
        sp.position_list = [place]
        if not vcolors is None:
            sp.vertex_colors = vcolors
        sp.clip_cap = True

        splist.append(sp)

    return splist

def add_geometry_instance(primitives, spieces, place, color, materials):

    for p,sp in zip(primitives, spieces):
        c = material_color(materials.get(p.material), color)
        sp.position_list.append(place)
        sp.color_list.append(c)

def material_color(material, color):

    if material is None:
        return color
    e = material.effect
    if e is None:
        return color
    c = e.diffuse
    if c is None:
        return color
    c8bit = tuple(int(255*r) for r in c)
    return c8bit

def vertex_colors(triangle_set, tarray, nv, colors):

    carray = tuple((i,aname) for i,name,aname,x in triangle_set.getInputList().getList() if name == 'COLOR')
    if len(carray) == 0:
        return None
    ci,aname = carray[0]
    tc = triangle_set.indices[:,:,ci]    # color index for each of 3 triangle vertices
    colors = colors[aname[1:]].data      # Get colors array, remove leading "#" from array name.
    # Collada allows different colors on the same vertex in different triangles,
    # but Hydra only allows one color per vertex.
    from numpy import empty, uint8
    vc = empty((nv,4), uint8)
    vc[tarray.ravel(),:] = (colors[tc.ravel(),:]*255).astype(uint8)
    return vc

# For drawings with multiple instances make colors a numpy uint8 array.
def set_instance_positions_and_colors(drawings):
    from chimerax.geometry import Places
    for d in drawings:
        if hasattr(d, 'position_list') and hasattr(d, 'color_list'):
            clist = d.color_list
            from numpy import array, uint8
            d.colors = array(clist, uint8).reshape((len(clist),4))
            d.positions = Places(d.position_list)
