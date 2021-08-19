#
# Calculate and show piecewise linear geodesic (locally minimum length) paths on a triangulated surface
# starting from a specified point and direction.
#

def geodesic(session, surface, start_point = None, direction = None,
             length = 100, color = (255,255,0,255), radius = 1):
    '''
    Create a geodesic path on the surface, ie. one that has locally minimal path length.
    The returned path is a MarkerSet model and has a specified length, color and radius
    (it is made up of cylinder segements).
    If a start_point is specified it need not be on the surface and the starting
    point on the surface will be at the closest point on an edge of the triangulated
    surface.  If no starting point is given then a random edge is taken and a random
    point on that edge.  The direction gives an initial direction vector and will be projected
    onto a triangle containing the starting point point edge.  If no direction is given
    then it is taken perpendicular to the starting edge.
    '''
    va = surface.vertices
    ta = surface.joined_triangles if hasattr(surface, 'joined_triangles') else surface.triangles
    nedges = next_edges(ta)
    tnormals = edge_triangle_normals(va, ta)

    edge, f = edge_start_point(start_point, surface, nedges)
    sdir = start_direction(direction, edge, tnormals, surface)
        
    points = walk_surface(edge, f, sdir, length, nedges, va, tnormals)

    mset = marker_path(session, points, color, radius)
    session.models.add([mset])
    
    return mset

def next_edges(triangles):
    '''
    Return dictionary mapping each edge to two other edges of triangle it belongs to.
    Edge is an ordered pair of vertex indices.
    '''
    ne = {}
    for v1,v2,v3 in triangles:
        ne[(v1,v2)] = ((v2,v3),(v3,v1))
        ne[(v2,v3)] = ((v3,v1),(v1,v2))
        ne[(v3,v1)] = ((v1,v2),(v2,v3))
    return ne

def edge_triangle_normals(vertices, triangles):
    tn = {}
    from chimerax.geometry import cross_product, normalize_vector
    for v1,v2,v3 in triangles:
        x1, x2, x3 = vertices[v1], vertices[v2], vertices[v3]
        n = normalize_vector(cross_product(x2-x1, x3-x1))
        tn[(v1,v2)] = tn[(v2,v3)] = tn[(v3,v1)] = n
    return tn

def edge_start_point(start_point, surface, nedges):
    if start_point is None:
        from random import choice, uniform
        edge = choice(tuple(nedges.keys()))
        f = uniform(0,1)
    else:
        from chimerax.core.commands import Center
        if isinstance(start_point, Center):
            start_point = start_point.scene_coordinates()
        edges = tuple(nedges.keys())
        edge, f = closest_edge_point(start_point, surface, edges)
    return edge, f

def closest_edge_point(start_point, surface, edges):
    emin = fmin = dmin = None
    vertices = surface.vertices
    for edge in edges:
        d, f = edge_distance(start_point, edge, vertices)
        if dmin is None or d < dmin:
            emin = edge
            fmin = f
            dmin = d
    return emin, fmin

def edge_distance(start_point, edge, vertices):
    e0,e1 = edge
    v0,v1 = vertices[e0], vertices[e1]
    ev = v1-v0
    from chimerax.geometry import inner_product, distance
    se = inner_product(start_point-v0, ev)
    e2 = inner_product(ev, ev)
    if se <= 0:
        f = 0
        p = v0
    elif se >= e2:
        f = 1
        p = v1
    else:
        f = se/e2
        p = (1-f)*v0+f*v1
    d = distance(start_point, p)
    return d, f

def start_direction(direction, edge, tnormals, surface):
    eo = (edge[1],edge[0])
    tn = tnormals[eo]
    if direction is None:
        # Start in direction perpendicular to starting edge.
        ev = edge_vector(eo, surface.vertices)
        from chimerax.geometry import cross_product, normalize_vector
        sdir = normalize_vector(cross_product(tn, ev))
    else:
        from chimerax.core.commands import Axis
        if isinstance(direction, Axis):
            direction = direction.scene_coordinates()
        from chimerax.geometry import inner_product
        sdir = direction - inner_product(direction,tn)*tn
    return sdir

def edge_vector(e, vertices):
    return vertices[e[1]] - vertices[e[0]]

def walk_surface(edge, position, direction, length, next_edges, vertices, triangle_normals):
    points = []
    e = edge
    f = position  # Fraction of distance along edge.
    l = 0
    prev_p = None
    from chimerax.geometry import distance
    while l < length:
        v1,v2 = e
        p = (1-f)*vertices[v1] + f*vertices[v2]
        points.append(p)
        if prev_p is not None:
            l += distance(p, prev_p)
        prev_p = p
        edges = next_edges[(v2,v1)]
        e, f = next_edge(p, direction, edges, vertices)
        if e is None:
            return points
        n1 = triangle_normals[e]
        n2 = triangle_normals[(e[1],e[0])]
        direction = next_direction(direction, e, vertices, n1, n2)
    return points

def next_edge(p, direction, edges, vertices):
    from chimerax.geometry import cross_product, inner_product
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
    from chimerax.geometry import cross_product, inner_product, normalize_vector
    en1, en2 = cross_product(ev, normal1), cross_product(ev, normal2)
    d = inner_product(direction, ev) * ev + inner_product(direction, en1) * en2
    d = normalize_vector(d)
    return d

def marker_path(session, points, color, radius):
    from chimerax.markers import MarkerSet, create_link
    ms = MarkerSet(session, 'surface path')
    mprev = None
    for id, xyz in enumerate(points):
        m = ms.create_marker(xyz, color, radius, id)
        if mprev is not None:
            create_link(mprev, m, rgba = color, radius = radius)
        mprev = m
    return ms
    
def register_geodesic_command(session):
    from chimerax.core.commands import CmdDesc, register
    from chimerax.core.commands import SurfaceArg, CenterArg, AxisArg, IntArg, Color8Arg, FloatArg
    desc = CmdDesc(
        required=[('surface', SurfaceArg)],
        keyword=[('start_point', CenterArg),
                 ('direction', AxisArg),
                 ('length', FloatArg),
                 ('color', Color8Arg),
                 ('radius', FloatArg)],
        synopsis='draw geodesic path on a surface'
    )
    register('geodesic', desc, geodesic, logger=session.logger)

register_geodesic_command(session)
