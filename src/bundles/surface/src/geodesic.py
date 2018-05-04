#
# Calculate and show piecewise linear geodesic (locally minimum length) paths on a triangulated surface.
# Also minimize path length for an existing path by moving nodes with ends fixed.
#

def geodesic(session, volume, length = 100, color = None, radius = 1, minimal = True):
    rgba = color.uint8x4() if color else (255,255,0,255)
    surface_path(session, volume.surfaces[0],
                 length = length, rgba = rgba, radius = radius, geodesic = minimal)

def surface_path(session, surface, length = 100, rgba = (255,255,0,255), radius = 1, geodesic = True):
    ep = edge_pairs(surface.triangles)
    from random import choice
    e = choice(tuple(ep.keys()))
#    e = first_element(ep.keys())
    vertices = surface.vertices
    if geodesic:
        tnormals = edge_triangle_normals(surface.vertices, surface.triangles)
        # Start in direction perpendicular to starting edge.
        eo = (e[1],e[0])
        ev = edge_vector(eo, vertices)
        from chimerax.core.geometry import cross_product, normalize_vector
        direction = normalize_vector(cross_product(tnormals[eo], ev))
    else:
        direction, tnormals = None
    points = make_surface_path(e, ep, length, vertices, direction, tnormals)
    from chimerax.markers import MarkerSet, create_link
    ms = MarkerSet(session, 'surface path')
    mprev = None
    for id, xyz in enumerate(points):
        m = ms.create_marker(xyz, rgba, radius, id)
        if mprev is not None:
            create_link(mprev, m, rgba = rgba, radius = radius)
        mprev = m
    session.models.add([ms])

def edge_pairs(triangles):
    '''
    Return dictionary mapping each edge to two other edges of triangle it belongs to.
    Edge is an ordered pair of vertex indices.
    '''
    ep = {}
    for v1,v2,v3 in triangles:
        ep[(v1,v2)] = ((v2,v3),(v3,v1))
        ep[(v2,v3)] = ((v3,v1),(v1,v2))
        ep[(v3,v1)] = ((v1,v2),(v2,v3))
    return ep

def edge_triangle_normals(vertices, triangles):
    tn = {}
    from chimerax.core.geometry import cross_product, normalize_vector
    for v1,v2,v3 in triangles:
        x1, x2, x3 = vertices[v1], vertices[v2], vertices[v3]
        n = normalize_vector(cross_product(x2-x1, x3-x1))
        tn[(v1,v2)] = tn[(v2,v3)] = tn[(v3,v1)] = n
    return tn

def edge_vector(e, vertices):
    return vertices[e[1]] - vertices[e[0]]

def first_element(iter):
    '''Return the next element from an iterator.'''
    for e in iter:
        return e

def make_surface_path(edge, edge_pairs, num_points, vertices,
                      direction = None, triangle_normals = None):
    points = []
    e = edge
    from random import random, choice
    f = random()
    for i in range(num_points):
        v1,v2 = e
        p = (1-f)*vertices[v1] + f*vertices[v2]
        points.append(p)
        edges = edge_pairs[(v2,v1)]
        if direction is None:
            e = choice(edges)
        else:
            e, f = next_edge(p, direction, edges, vertices)
            n1 = triangle_normals[e]
            n2 = triangle_normals[(e[1],e[0])]
            direction = next_direction(direction, e, vertices, n1, n2)
    return points

def next_edge(p, direction, edges, vertices):
    from chimerax.core.geometry import cross_product, inner_product
    for e in edges:
        ev1,ev2 = vertices[e[0]]-p, vertices[e[1]]-p
        n = cross_product(ev1, ev2)
        tn = cross_product(direction, n)
        i1, i2 = inner_product(tn, ev1), inner_product(tn, ev2)
        if (i1 < 0 and i2 > 0) or (i1 > 0 and i2 < 0):
            # Edge is split by geodesic direction
            f = i1 / (i1 - i2)
            p2 = (1-f)*vertices[e[0]] + f*vertices[e[1]]
            return e, f
    return None, None

def next_direction(direction, e, vertices, normal1, normal2):
    ev = edge_vector(e, vertices)
    from chimerax.core.geometry import cross_product, inner_product, normalize_vector
    en1, en2 = cross_product(ev, normal1), cross_product(ev, normal2)
    d = inner_product(direction, ev) * ev + inner_product(direction, en1) * en2
    d = normalize_vector(d)
    return d
    
def register_geodesic_command(session):
    from chimerax.core.commands import CmdDesc, register, IntArg, ColorArg, FloatArg
    from chimerax.map import MapArg
    desc = CmdDesc(
        required=[('volume', MapArg)],
        keyword=[('length', IntArg),
                 ('color', ColorArg),
                 ('radius', FloatArg)],
        synopsis='draw geodesic path on a volume surface'
    )
    register('geodesic', desc, geodesic, logger=session.logger)

register_geodesic_command(session)
