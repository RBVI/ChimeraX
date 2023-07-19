# -----------------------------------------------------------------------------
# Produce hexagonal lattices on an icosahedron.  The hexagons are bent where
# they cross the edges of the icosahedron.
#
# These lattices are described at
#
#       http://viperdb.scripps.edu/icos_server.php?icspage=paradigm


def show_hk_lattice(session, h, k, radius, orientation='222',
                    color=(255, 255, 255, 255), sphere_factor=0,
                    edge_radius=None, mesh=False, replace=True, alpha='hexagonal'):

    varray, tarray, hex_edges = hk_icosahedron_lattice(h, k, radius, orientation, alpha)
    interpolate_with_sphere(varray, radius, sphere_factor)

    name = 'Icosahedron h = %d, k = %d' % (h,k)
    
    if mesh:
        model = sm = _cage_surface(session, name, replace)
        sm.set_geometry(varray, None, tarray)
        sm.color = color
        sm.display_style = sm.Mesh
        sm.edge_mask = hex_edges    # Hide spokes of hexagons.
        if sm.id is None:
            session.models.add([sm])
    else:
        # Make cage from markers.
        from chimerax.core.models import Surface
        sm = Surface(name, session)
        sm.set_geometry(varray, None, tarray)
        sm.color = color
        sm.display_style = sm.Mesh
        sm.edge_mask = hex_edges    # Hide spokes of hexagons.
        if edge_radius is None:
            edge_radius = .01 * radius
        mset = _cage_markers(session, name) if replace else None
        from chimerax.markers.cmd import markers_from_mesh
        model = markers_from_mesh(session, [sm], color = color,
                                  edge_radius = edge_radius, markers = mset)
        model.name = name
        if mset:
            mset._prev_markers.delete()

    model.hkcage = True

    return model

# -----------------------------------------------------------------------------
#
def _cage_markers(session, name):
    
    from chimerax.markers import MarkerSet
    mlist = [m for m in session.models.list(type = MarkerSet)
             if hasattr(m, 'hkcage')]
    if mlist:
        mset = mlist[0]
        mset._prev_markers = mset.atoms
        mset.name = name
        mset.hkcage = True
        return mset
    return None

# -----------------------------------------------------------------------------
#
def _cage_surface(session, name, replace):
    
    # Make new surface model or find an existing one.
    sm = None
    from chimerax.core.models import Surface
    if replace:
        mlist = [m for m in session.models.list(type = Surface)
                 if hasattr(m, 'hkcage')]
        if mlist:
            sm = mlist[0]
            sm.name = name
    if sm is None:
        sm = Surface(name, session)
    sm.hkcage = True
    return sm

# -----------------------------------------------------------------------------
#
def hk_icosahedron_lattice(h, k, radius, orientation = '222', alpha = 'hexagonal'):
    # Find triangles for the hk lattice covering one asymmetric unit equilateral triangle.
    # The asym unit triangle (corners) and hk lattice triangles are in the xy plane in 3-d.
    if alpha == 'hexagonal':
        corners, triangles, t_hex_edges = hk_triangle(h, k)
    elif alpha == 'hexagonal-dual':
        corners, triangles, t_hex_edges = hk_triangle_dual(h, k)
    elif alpha == 'trihex':
        corners, triangles, t_hex_edges = hk_triangle_trihex(h, k)
    elif alpha == 'trihex-dual':
        corners, triangles, t_hex_edges = hk_triangle_trihex_dual(h, k)
    elif alpha == 'snub':
        corners, triangles, t_hex_edges = hk_triangle_snub(h, k)
    elif alpha == 'snub-dual':
        corners, triangles, t_hex_edges = hk_triangle_snub_dual(h, k)
    elif alpha == 'rhomb':
        corners, triangles, t_hex_edges = hk_triangle_rhomb(h, k)
    elif alpha == 'rhomb-dual':
        corners, triangles, t_hex_edges = hk_triangle_rhomb_dual(h, k)

    from chimerax.geometry.icosahedron import icosahedron_geometry
    ivarray, itarray = icosahedron_geometry(orientation)

    # Map the 2d hk asymmetric unit triangles onto each face of an icosahedron
    tlist = []
    for i0,i1,i2 in itarray:
        face = ivarray[i0], ivarray[i1], ivarray[i2]
        tmap = triangle_map(corners, face)
        tlist.extend(map_triangles(tmap, triangles))

    # Convert from triangles defined by 3 vertex points, to an array of
    # unique vertices and triangles as 3 indices into the unique vertex list.
    va, ta = surface_geometry(tlist, tolerance = 1e-5)

    # Scale to requested radius
    from numpy import multiply, array, intc
    multiply(va, radius, va)
    
    # Compute the edge mask to show just the hexagon edges.
    hex_edges = array(t_hex_edges * len(itarray), intc)
    
    return va, ta, hex_edges

# -----------------------------------------------------------------------------
# Calculate a list of 2-d triangles that is the part of the hk lattice within
# a single asymmetric unit equilateral triangle in the plane.  Triangles that
# would cross the boundary of the asymmetric unit triangle are clipped.
#
# The returned corners are the 3 vertices of the asymmetric unit triangle.
# The returned asym unit and hk triangles are in the xy plane in 3 dimensions (z=0).
# A returned edge mask indicates which of the hk lattice triangle edges
# are hexagon edges.
#
def hk_triangle_snub(h, k):
    # Multiply h,k by 3 so hexagon corners have integer coordinates for
    # exact intersection calculations.
    corners2d = ((0, 0), (5 * h + k, 4 * k - h), (-4 * k + h, 5 * k + 4 * h))

    hex_corner_offset = ((2, -1), (1, 1), (-1, 2), (-2, 1), (-1, -1), (1, -2))
    tri_corner_offset = ((1, 1), (-1, 2), (-2, 1), (-1, -1), (1, -2), (2, -1))
    kmax = max(k, h + k)
    triangles2d = []
    hex_edges = []
    for k0 in range(kmax + 1):
        for h0 in range(-k0, h + 1):
            for c in range(6):
                h1o, k1o = hex_corner_offset[c]
                h1t, k1t = tri_corner_offset[c]
                h2o, k2o = hex_corner_offset[(c + 1) % 6]
                h2t, k2t = tri_corner_offset[(c + 1) % 6]
                tri = ((5 * h0 + k0, 4 * k0 - h0), (5 * h0 + k0 + h1o, 4 * k0 - h0 + k1o), (5 * h0 + k0 + h2o, 4 * k0 - h0 + k2o))
                ti, he = triangle_intersection(tri, corners2d, 2)
                triangles2d.extend(ti)
                hex_edges.extend(he)
                tri = ((5 * h0 + k0 + h1o, 4 * k0 - h0 + k1o), (5 * h0 + k0 + h1o + h1t, 4 * k0 - h0 + k1o + k1t), (5 * h0 + k0 + h1o + h2t, 4 * k0 - h0 + k1o + k2t))
                ti, he = triangle_intersection(tri, corners2d, 2)
                triangles2d.extend(ti)
                hex_edges.extend(he)
                tri = ((5 * h0 + k0 + h1o, 4 * k0 - h0 + k1o), (5 * h0 + k0 + h1o + h1t, 4 * k0 - h0 + k1o + k1t), (5 * h0 + k0 + h1o + h2t, 4 * k0 - h0 + k1o + k2t))
                ti, he = triangle_intersection(tri, corners2d, 3)
                triangles2d.extend(ti)
                hex_edges.extend(he)

    corners = hk3_to_xyz(corners2d)
    triangles = [hk3_to_xyz(t) for t in triangles2d]

    return corners, triangles, hex_edges


def hk_triangle_snub_dual(h, k):
    # Multiply h,k by 3 so hexagon corners have integer coordinates for
    # exact intersection calculations.
    corners2d = ((0, 0), (5 * h + k, 4 * k - h), (-4 * k + h, 5 * k + 4 * h))

    hex_corner_offset = ((2, 0), (0, 2), (-2, 2), (-2, 0), (0, -2), (2, -2))
    tri_start_offset = ((2, -1), (1, 1), (-1, 2), (-2, 1), (-1, -1), (1, -2))
    tri_corner_offset = ((1, -1), (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1))
    kmax = max(k, h + k)
    triangles2d = []
    hex_edges = []
    for k0 in range(kmax + 1):
        for h0 in range(-k0, h + 1):
            for c in range(6):
                h1o, k1o = hex_corner_offset[c]
                h2o, k2o = hex_corner_offset[(c + 1) % 6]
                hst, kst = tri_start_offset[c]
                h1t, k1t = tri_corner_offset[c]
                h2t, k2t = tri_corner_offset[(c + 1) % 6]
                h3t, k3t = tri_corner_offset[(c + 2) % 6]
                h4t, k4t = tri_corner_offset[(c + 3) % 6]
                tri = ((5 * h0 + k0, 4 * k0 - h0), (5 * h0 + k0 + h1o, 4 * k0 - h0 + k1o), (5 * h0 + k0 + h2o, 4 * k0 - h0 + k2o))
                ti, he = triangle_intersection(tri, corners2d, 1)
                triangles2d.extend(ti)
                hex_edges.extend(he)
                tri = ((5 * h0 + k0 + hst, 4 * k0 - h0 + kst), (5 * h0 + k0 + hst + h1t, 4 * k0 - h0 + kst + k1t), (5 * h0 + k0 + hst + h2t, 4 * k0 - h0 + kst + k2t))
                ti, he = triangle_intersection(tri, corners2d, 2)
                triangles2d.extend(ti)
                hex_edges.extend(he)
                tri = ((5 * h0 + k0 + hst, 4 * k0 - h0 + kst), (5 * h0 + k0 + hst + h2t, 4 * k0 - h0 + kst + k2t), (5 * h0 + k0 + hst + h3t, 4 * k0 - h0 + kst + k3t))
                ti, he = triangle_intersection(tri, corners2d, 2)
                triangles2d.extend(ti)
                hex_edges.extend(he)

    corners = hk3_to_xyz(corners2d)
    triangles = [hk3_to_xyz(t) for t in triangles2d]

    return corners, triangles, hex_edges


def hk_triangle_rhomb(h, k):
    # Multiply h,k by 3 so hexagon corners have integer coordinates for
    # exact intersection calculations.
    corners2d = ((0, 0), (5 * h, 5 * k), (-5 * k, 5 * (h + k)))

    hex_corner_offset = ((2, -1), (1, 1), (-1, 2), (-2, 1), (-1, -1), (1, -2))
    tri_corner_offset = ((2, -2), (2, 0), (0, 2), (-2, 2), (-2, 0), (0, -2))
    kmax = max(k, h + k)
    triangles2d = []
    hex_edges = []
    for k0 in range(kmax + 1):
        for h0 in range(-k0, h + 1):
            for c in range(6):
                h1o, k1o = hex_corner_offset[c]
                h1t, k1t = tri_corner_offset[c]
                h2o, k2o = hex_corner_offset[(c + 1) % 6]
                h2t, k2t = tri_corner_offset[(c + 1) % 6]
                tri = ((5 * h0, 5 * k0), (5 * h0 + h1o, 5 * k0 + k1o), (5 * h0 + h2o, 5 * k0 + k2o))
                ti, he = triangle_intersection(tri, corners2d, 2)
                triangles2d.extend(ti)
                hex_edges.extend(he)
                tri = ((5 * h0 + h1o, 5 * k0 + k1o), (5 * h0 + h1o + h1t, 5 * k0 + k1o + k1t), (5 * h0 + h1o + h2t, 5 * k0 + k1o + k2t))
                ti, he = triangle_intersection(tri, corners2d, 1)
                triangles2d.extend(ti)
                hex_edges.extend(he)

    corners = hk3_to_xyz(corners2d)
    triangles = [hk3_to_xyz(t) for t in triangles2d]

    return corners, triangles, hex_edges


def hk_triangle_rhomb_dual(h, k):
    # Multiply h,k by 3 so hexagon corners have integer coordinates for
    # exact intersection calculations.
    corners2d = ((0, 0), (6 * h, 6 * k), (-6 * k, 6 * (h + k)))

    hex_corner_offset = ((3, 0), (0, 3), (-3, 3), (-3, 0), (0, -3), (3, -3))
    tri_corner_offset = ((-1, 2), (-2, 1), (-1, -1), (1, -2), (2, -1), (1, 1))
    kmax = max(k, h + k)
    triangles2d = []
    hex_edges = []
    for k0 in range(kmax + 1):
        for h0 in range(-k0, h + 1):
            for c in range(6):
                h1o, k1o = hex_corner_offset[c]
                h1t, k1t = tri_corner_offset[c]
                h2o, k2o = hex_corner_offset[(c + 1) % 6]
                tri = ((6 * h0, 6 * k0), (6 * h0 + h1o, 6 * k0 + k1o), (6 * h0 + h2o, 6 * k0 + k2o))
                ti, he = triangle_intersection(tri, corners2d, 1)
                triangles2d.extend(ti)
                hex_edges.extend(he)
                tri = ((6 * h0 + h1o, 6 * k0 + k1o), (6 * h0 + h2o, 6 * k0 + k2o), (6 * h0 + h1o + h1t, 6 * k0 + k1o + k1t))
                ti, he = triangle_intersection(tri, corners2d, 2)
                triangles2d.extend(ti)
                hex_edges.extend(he)

    corners = hk3_to_xyz(corners2d)
    triangles = [hk3_to_xyz(t) for t in triangles2d]

    return corners, triangles, hex_edges


def hk_triangle(h, k):
    # Multiply h,k by 3 so hexagon corners have integer coordinates for
    # exact intersection calculations.
    corners2d = ((0, 0), (3 * h, 3 * k), (-3 * k, 3 * (h + k)))

    hex_corner_offset = ((2, -1), (1, 1), (-1, 2), (-2, 1), (-1, -1), (1, -2))
    kmax = max(k, h + k)
    triangles2d = []
    hex_edges = []
    for k0 in range(kmax + 1):
        for h0 in range(-k0, h + 1):
            for c in range(6):
                h1o, k1o = hex_corner_offset[c]
                h2o, k2o = hex_corner_offset[(c + 1) % 6]
                tri = ((3 * h0, 3 * k0), (3 * h0 + h1o, 3 * k0 + k1o), (3 * h0 + h2o, 3 * k0 + k2o))
                ti, he = triangle_intersection(tri, corners2d, 2)
                triangles2d.extend(ti)
                hex_edges.extend(he)

    corners = hk3_to_xyz(corners2d)
    triangles = [hk3_to_xyz(t) for t in triangles2d]

    return corners, triangles, hex_edges


def hk_triangle_dual(h, k):
    # Multiply h,k by 3 so hexagon corners have integer coordinates for
    # exact intersection calculations.
    corners2d = ((0,0), (3*h,3*k), (-3*k,3*(h+k)))

    hex_corner_offset = ((3,0), (0,3), (-3,3), (-3,0), (0,-3), (3,-3))
    kmax = max(k, h+k)
    triangles2d = []
    hex_edges = []
    for k0 in range(kmax+1):
        for h0 in range(-k0, h+1):
            for c in range(6):
                h1o, k1o = hex_corner_offset[c]
                h2o, k2o = hex_corner_offset[(c+1)%6]
                tri = ((3*h0,3*k0), (3*h0+h1o,3*k0+k1o), (3*h0+h2o,3*k0+k2o))
                ti, he = triangle_intersection(tri, corners2d, 2)
                triangles2d.extend(ti)
                hex_edges.extend(he)

    corners = hk3_to_xyz(corners2d)
    triangles = [hk3_to_xyz(t) for t in triangles2d]
    
    return corners, triangles, hex_edges

def hk_triangle_trihex_dual(h, k):
    # Multiply h,k by 3 so hexagon corners have integer coordinates for
    # exact intersection calculations.
    corners2d = ((0, 0), (3 * h, 3 * k), (-3 * k, 3 * (h + k)))

    hex_corner_offset = ((2, -1), (1, 1), (-1, 2), (-2, 1), (-1, -1), (1, -2))
    kmax = max(k, h + k)
    triangles2d = []
    hex_edges = []
    for k0 in range(kmax + 1):
        for h0 in range(-k0, h + 1):
            for c in range(6):
                h1o, k1o = hex_corner_offset[c]
                h2o, k2o = hex_corner_offset[(c + 1) % 6]
                tri = ((3 * h0, 3 * k0), (3 * h0 + h1o, 3 * k0 + k1o), (3 * h0 + h2o, 3 * k0 + k2o))
                ti, he = triangle_intersection(tri, corners2d, 1)
                triangles2d.extend(ti)
                hex_edges.extend(he)

    corners = hk3_to_xyz(corners2d)
    triangles = [hk3_to_xyz(t) for t in triangles2d]

    return corners, triangles, hex_edges


# -----------------------------------------------------------------------------
# Calculate a list of 2-d triangles that is the part of the hk lattice within
# a single asymmetric unit equilateral triangle in the plane.  Triangles that
# would cross the boundary of the asymmetric unit triangle are clipped.
#
# The returned corners are the 3 vertices of the asymmetric unit triangle.
# The returned asym unit and hk triangles are in the xy plane in 3 dimensions (z=0).
# A returned edge mask indicates which of the hk lattice triangle edges
# are hexagon edges.
#
def hk_triangle_trihex(h, k):
    # Multiply h,k by 3 so hexagon corners have integer coordinates for
    # exact intersection calculations.
    corners2d = ((0, 0), (2 * h, 2 * k), (-2 * k, 2 * (h + k)))

    hex_corner_offset = ((1, -1), (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1))
    kmax = max(k, h + k)
    triangles2d = []
    hex_edges = []
    for k0 in range(kmax + 1):
        for h0 in range(-k0, h + 1):
            for c in range(6):
                h1o, k1o = hex_corner_offset[c]
                h2o, k2o = hex_corner_offset[(c + 1) % 6]
                tri = ((2 * h0, 2 * k0), (2 * h0 + h1o, 2 * k0 + k1o), (2 * h0 + h2o, 2 * k0 + k2o))
                ti, he = triangle_intersection(tri, corners2d, 2)
                triangles2d.extend(ti)
                hex_edges.extend(he)
    corners = hk2_to_xyz(corners2d)
    triangles = [hk2_to_xyz(t) for t in triangles2d]
    return corners, triangles, hex_edges


# -----------------------------------------------------------------------------
# Triangulate the portion of triangle t1 inside t2.  The triangles are specified
# by 3 vertex points and are in 2 dimensions.  Only the cases that occur in the
# hk icosahedral grids are handled.
#
def triangle_intersection(t1, t2, edge_mask):

    interior_vertices = []
    exterior_vertices = []
    boundary_vertices = []
    locations = {-1:exterior_vertices, 0:boundary_vertices, 1:interior_vertices}
    for k in range(3):
        loc = vertex_in_triangle(t1[k], t2)
        locations[loc].append(k)

    iv = len(interior_vertices)
    ev = len(exterior_vertices)
    if iv == 0 and ev > 0:
        return [], []
    if ev == 0:
        return [t1], [edge_mask]

    # Have at least one exterior and one interior vertex.  Need new triangles.
    if iv == 1 and ev == 1:
        i = interior_vertices[0]
        b = boundary_vertices[0]
        e = exterior_vertices[0]
        thalf = list(t1)
        thalf[e] = cut_point(t1[i], t1[e], t2)
        return [thalf], [mask_edge(edge_mask, (e,b))]
    
    if iv == 1 and ev == 2:
        i = interior_vertices[0]
        e1 = exterior_vertices[0]
        e2 = exterior_vertices[1]
        tpiece = list(t1)
        tpiece[e1] = cut_point(t1[i], t1[e1], t2)
        tpiece[e2] = cut_point(t1[i], t1[e2], t2)
        return [tpiece], [mask_edge(edge_mask, (e1,e2))]

    if iv == 2 and ev == 1:
        i1 = interior_vertices[0]
        i2 = interior_vertices[1]
        e = exterior_vertices[0]
        tpiece1 = list(t1)
        tpiece1[e] = cut_point(t1[i1], t1[e], t2)
        em1 = mask_edge(edge_mask, (i2,e))
        tpiece2 = list(t1)
        tpiece2[e] = cut_point(t1[i2], t1[e], t2)
        tpiece2[i1] = tpiece1[e]
        em2 = mask_edge(edge_mask, (i1,e), (i1,i2))
        return [tpiece1, tpiece2], [em1, em2]
    

    raise ValueError('hkcage: Unexpected triangle intersection')

# -----------------------------------------------------------------------------
# Vertex and triangle are in two dimensions with triangle defined by 3 corner
# vertices.
#
def vertex_in_triangle(v, t):

    v0, v1, v2 = t
    v01p = (-v1[1]+v0[1], v1[0]-v0[0])
    v12p = (-v2[1]+v1[1], v2[0]-v1[0])
    v20p = (-v0[1]+v2[1], v0[0]-v2[0])
    from numpy import subtract, dot as inner_product
    sides = (inner_product(subtract(v,v0), v01p),
             inner_product(subtract(v,v1), v12p),
             inner_product(subtract(v,v2), v20p))
    inside = len([s for s in sides if s > 0])
    outside = len([s for s in sides if s < 0])
    if inside == 3:
        return 1     # Interior
    elif outside == 0:
        return 0     # Boundary
    
    return -1        # Outside

# -----------------------------------------------------------------------------
# Find intersection of segment u->v with triangle boundary.
# u and v must be an interior and an exterior point.
#
def cut_point(u, v, tri):

    for e in range(3):
        ip = segment_intersection(u, v, tri[e], tri[(e+1)%3])
        if ip:
            return ip
    raise ValueError('hkcage: No intersection %s %s %s' % (u, v, tri))

# -----------------------------------------------------------------------------
# Find intersection of segment ab with segment cd.
# Return (x,y) or None if no intersectin.
#
def segment_intersection(a, b, c, d):

    m11 = b[0]-a[0]
    m21 = b[1]-a[1]
    m12 = c[0]-d[0]
    m22 = c[1]-d[1]
    det = m11*m22 - m12*m21
    if det == 0:
        return None

    y1 = c[0]-a[0]
    y2 = c[1]-a[1]
    f = float(m22*y1 - m12*y2)/det
    g = float(-m21*y1 + m11*y2)/det
    if f < 0 or f > 1 or g < 0 or g > 1:
        return None

    p = (a[0] + m11*f, a[1] + m21*f)
    return p

# -----------------------------------------------------------------------------
# Mask out edges given by pair of vertex indices (0-2).  Bits 0, 1, and 2
# correspond to edges 0-1, 1-2, and 2-0 respectively.
#
def mask_edge(edge_mask, *edges):

    ebits = {(0,1):1, (1,0):1, (1,2):2, (2,1):2, (2,0):4, (0,2):4}
    emask = edge_mask
    for e in edges:
        emask &= ~ebits[e]
    return emask

# -----------------------------------------------------------------------------
# Shear transform 2d hk points to points on the xy plane in 3 dimensions (z=0).
#
def hk3_to_xyz(hklist):

    from math import sqrt
    hx = sqrt(3)/6
    hy = 0.5/3
    ky = 1.0/3
    xyz_list = [(h*hx,k*ky+h*hy,0) for h,k in hklist]
    return xyz_list

# -----------------------------------------------------------------------------
# Shear transform 2d hk points to points on the xy plane in 3 dimensions (z=0).
#
def hk2_to_xyz(hklist):

    from math import sqrt
    hx = 1/2.0
    hy = 1/(2*sqrt(3))
    ky = 1/sqrt(3)
    xyz_list = [(h*hx, k*ky+h*hy, 0) for h, k in hklist]
    return xyz_list

# -----------------------------------------------------------------------------
# Compute the 3 by 4 transform matrix mapping one 3-d triangle to another.
#
def triangle_map(tri1, tri2):

    from numpy import zeros, subtract, dot as matrix_multiply, float64

    f1 = zeros((3,3), float64)
    f1[:,0], f1[:,1] = subtract(tri1[1], tri1[0]), subtract(tri1[2], tri1[0])
    f1[:,2] = cross_product(f1[:,0], f1[:,1])

    f2 = zeros((3,3), float64)
    f2[:,0], f2[:,1] = subtract(tri2[1], tri2[0]), subtract(tri2[2], tri2[0])
    f2[:,2] = cross_product(f2[:,0], f2[:,1])
    
    from numpy.linalg import inv as inverse
    f1inv = inverse(f1)

    tmap = zeros((3,4), float64)
    tmap[:,:3] = matrix_multiply(f2, f1inv)
    tmap[:,3] = subtract(tri2[0], matrix_multiply(tmap[:,:3],tri1[0]))

    from chimerax.geometry import Place
    tf = Place(matrix = tmap)
    
    return tf
    
# -----------------------------------------------------------------------------
# Apply a 3x4 affine transformation to vertices of triangles.
#
def map_triangles(tmap, triangles):

    tri = [[tmap * v for v in t] for t in triangles]
    return tri
    
# -----------------------------------------------------------------------------
#
def cross_product(u, v):

    return (u[1]*v[2]-u[2]*v[1], u[2]*v[0]-u[0]*v[2], u[0]*v[1]-u[1]*v[0])
    
# -----------------------------------------------------------------------------
# Take a list of triangles where each triangle is specified by 3 xyz vertex
# positions and convert to a vertex and triangle array where the triangle
# array contains indices into the vertex array.  Vertices in the original
# triangle data that are close (within tolerance) are merged into a single
# vertex.
#
def surface_geometry(triangles, tolerance = 1e-5):

    from numpy import array, reshape, single as floatc, intc
    varray = reshape(triangles, (3*len(triangles),3)).astype(floatc)
    
    uindex = {}
    unique = []
    from chimerax.geometry import find_close_points
    for v in range(len(varray)):
        if not v in uindex:
            i1, i2 = find_close_points(varray[v:v+1,:], varray, tolerance)
            for i in i2:
                if not i in uindex:
                    uindex[i] = len(unique)
            unique.append(varray[v])

    uvarray = array(unique, floatc)
    tlist = [(uindex[3*t],uindex[3*t+1],uindex[3*t+2]) for t in range(len(triangles))]
    tarray = array(tlist, intc)
    
    return uvarray, tarray

# -----------------------------------------------------------------------------
# Radially interpolate vertex points a certain factor towards a sphere of
# given radius.
#
def interpolate_with_sphere(varray, radius, sphere_factor):

    if sphere_factor == 0:
        return

    from math import sqrt
    for v in range(len(varray)):
        x,y,z = varray[v]
        r = sqrt(x*x + y*y + z*z)
        if r > 0:
            ri = r * (1-sphere_factor) + radius*sphere_factor
            f = ri/r
            varray[v,:] = (f*x, f*y, f*z)
