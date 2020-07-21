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
# Edge length of unit icosahedron.
#
def icosahedron_edge_length():

    from math import sqrt
    e = sqrt(2 - 2/sqrt(5))
    return e

# -----------------------------------------------------------------------------
# Angles between 2-fold, 3-fold and 5-fold symmetry axes of an icosahedron.
#
def icosahedron_angles():

    e = icosahedron_edge_length()
    from math import asin, sqrt
    a25 = asin(e/2)
    a35 = asin(e/sqrt(3))
    a23 = asin(e/(2*sqrt(3)))
    return a23, a25, a35

# -----------------------------------------------------------------------------
# Return vertices and triangles for an icosahedron.
#
def icosahedron_geometry(orientation = '2n5'):

    a23, a25, a35 = icosahedron_angles()
    a = 2*a25                       # Angle spanned by edge from center
    
    # 5-fold symmetry axis along z
    from math import cos, sin, pi
    c5 = cos(2*pi/5)
    s5 = sin(2*pi/5)
    from . import Place
    tf5 = Place(((c5, -s5, 0, 0),
                 (s5, c5, 0, 0),
                 (0, 0, 1, 0)))

    # 2-fold symmetry axis along x
    tf2 = Place(((1, 0, 0, 0),
                 (0, -1, 0, 0),
                 (0, 0, -1, 0)))

    p = (0, 0, 1)
    p50 = (0, sin(a), cos(a))
    p51 = tf5 * p50
    p52 = tf5 * p51
    p53 = tf5 * p52
    p54 = tf5 * p53
    vertices = [p, p50, p51, p52, p53, p54]
    vertices.extend([tf2 * q for q in vertices])

    if orientation != '2n5':
        tf = coordinate_system_transform('2n5', orientation)
        vertices = [tf * p for p in vertices]
    
    #
    # Vertex numbering
    #
    #  Top   1          Bottom
    #      2   5        9   10
    #        0            6
    #      3   4        8   11
    #                     7
    # 20 triangles composing icosahedron.
    #
    triangles = ((0,1,2), (0,2,3), (0,3,4), (0,4,5), (0,5,1),
                 (6,7,8), (6,8,9), (6,9,10), (6,10,11), (6,11,7),
                 (1,9,2), (2,9,8), (2,8,3), (3,8,7), (3,7,4),
                 (4,7,11), (4,11,5), (5,11,10), (5,10,1), (1,10,9))

    return vertices, triangles

# -----------------------------------------------------------------------------
#
def icosahedron_triangulation(radius = 1, subdivision_levels = 0,
                              sphere_factor = 0, orientation = '222',
                              return_radii = False):

    vlist, tlist = icosahedron_geometry(orientation)
    from numpy import array, single as floatc, intc
    varray = array(vlist, floatc)
    tarray = array(tlist, intc)
    for level in range(subdivision_levels):
        from chimerax.surface import subdivide_triangles
        varray, tarray = subdivide_triangles(varray, tarray)
    icosahedral_radii = _radii(varray)
    if sphere_factor != 0:
        _interpolate_radii(varray, icosahedral_radii, 1, sphere_factor)
    _scale_radii(varray, radius)

    if return_radii:
        return varray, tarray, icosahedral_radii

    return varray, tarray

# -----------------------------------------------------------------------------
#
def _radii(points):

    from numpy import sqrt
    r = sqrt(points[:,0]*points[:,0] +
             points[:,1]*points[:,1] +
             points[:,2]*points[:,2])
    return r

# -----------------------------------------------------------------------------
#
def _interpolate_radii(varray, radii_0, radii_1, f):

    rarray = _radii(varray)
    scale = radii_0 * (1-f) + radii_1 * f
    from numpy import divide, multiply
    divide(scale, rarray, scale)
    for a in range(3):
        multiply(varray[:,a], scale, varray[:,a])

# -----------------------------------------------------------------------------
#
def _scale_radii(varray, scale):

    from numpy import multiply
    for a in range(3):
        multiply(varray[:,a], scale, varray[:,a])

# -----------------------------------------------------------------------------
# 60 icosahedral symmetry matrices.
#
def icosahedral_symmetry_matrices(orientation = '222', center = (0,0,0)):

    t = icosahedral_matrix_table()
    from . import recenter_symmetries
    tflist = recenter_symmetries(t.get(orientation, None), center)
    return tflist

# -----------------------------------------------------------------------------
# Compute icosahedral transformation matrices for different coordinate systems.
#
icos_matrices = {}      # Maps orientation name to 60 matrices.
def icosahedral_matrix_table():

    global icos_matrices
    if icos_matrices:
        return icos_matrices
        
    from math import cos, pi
    c = cos(2*pi/5)      #  .309016994
    c2 = cos(4*pi/5)      # -.809016994

    m222 = (

    ((1.0,   0.0,   0.0,     0.0),
     (0.0,   1.0,   0.0,     0.0),
     (0.0,   0.0,   1.0,     0.0)),

    ((c2,  -0.5,   c,       0.0),
     (-0.5,   c,  c2,       0.0),
     (c,   c2,  -0.5,       0.0)),

    ((0.0,  1.0,   0.0,      0.0),
     (0.0,  0.0,  -1.0,      0.0),
     (-1.0, 0.0,   0.0,      0.0)),

    ((-c2,  -0.5,  -c,      0.0),
     (-0.5,  -c,  c2,       0.0),
     (c,   -c2,  -0.5,      0.0)),

    ((0.5,   c,  c2,        0.0),
     (-c,  c2,  -0.5,       0.0),
     (c2,   0.5,  -c,       0.0)),

    ((-c,  c2,  -0.5,       0.0),
     (c2,   0.5,  -c,       0.0),
     (0.5,   c,  c2,        0.0)),

    ((c2,   0.5,  -c,       0.0),
     (0.5,   c,  c2,        0.0),
     (-c,  c2,  -0.5,       0.0)),

    ((c2,  -0.5,  -c,       0.0),
     (0.5,  -c,  c2,        0.0),
     (c,  c2,   0.5,        0.0)),

    ((-c,   -c2,  -0.5,     0.0),
     (c2,  -0.5,  -c,       0.0),
     (-0.5,   c,   -c2,     0.0)),

    ((0.5,  -c,  c2,        0.0),
     (-c,   -c2,  -0.5,     0.0),
     (-c2,   0.5,   c,      0.0)),

    ((0.0,   0.0,  -1.0,     0.0),
     (-1.0,   0.0,   0.0,    0.0),
     (0.0,   1.0,   0.0,     0.0)),

    ((-0.5,  -c,  c2,       0.0),
     (c,   -c2,  -0.5,      0.0),
     (-c2,  -0.5,  -c,      0.0)),

    ((-0.5,   c,  c2,       0.0),
     (c,  c2,  -0.5,        0.0),
     (c2,  -0.5,   c,       0.0)),

    ((-c,  c2,  -0.5,       0.0),
     (-c2,  -0.5,   c,      0.0),
     (-0.5,  -c,   -c2,     0.0)),

    ((c2,   0.5,  -c,       0.0),
     (-0.5,  -c,   -c2,     0.0),
     (c,   -c2,   0.5,      0.0)),

    ((0.5,   c,  c2,        0.0),
     (c,   -c2,   0.5,      0.0),
     (-c2,  -0.5,   c,      0.0)),

    ((-0.5,   c,  c2,       0.0),
     (-c,   -c2,   0.5,     0.0),
     (-c2,   0.5,  -c,      0.0)),

    ((0.0,   0.0,  -1.0,     0.0),
     (1.0,   0.0,   0.0,     0.0),
     (0.0,  -1.0,   0.0,     0.0)),

    ((-0.5,  -c,  c2,       0.0),
     (-c,  c2,   0.5,       0.0),
     (c2,   0.5,   c,       0.0)),

    ((0.0,  -1.0,   0.0,     0.0),
     (0.0,   0.0,   1.0,     0.0),
     (-1.0,   0.0,   0.0,    0.0)),

    ((c2,   0.5,   c,       0.0),
     (0.5,   c,   -c2,      0.0),
     (c,   -c2,  -0.5,      0.0)),

    ((-c2,   0.5,  -c,      0.0),
     (0.5,  -c,   -c2,      0.0),
     (c,  c2,  -0.5,        0.0)),

    ((-c,   -c2,  -0.5,     0.0),
     (-c2,   0.5,   c,      0.0),
     (0.5,  -c,  c2,        0.0)),

    ((0.5,  -c,  c2,        0.0),
     (c,  c2,   0.5,        0.0),
     (c2,  -0.5,  -c,       0.0)),

    ((c2,  -0.5,  -c,       0.0),
     (-0.5,   c,   -c2,     0.0),
     (-c,   -c2,  -0.5,     0.0)),

    ((-c,  c2,   0.5,       0.0),
     (c2,   0.5,   c,       0.0),
     (-0.5,  -c,  c2,       0.0)),

    ((-c,   -c2,   0.5,     0.0),
     (-c2,   0.5,  -c,      0.0),
     (-0.5,   c,  c2,       0.0)),

    ((1.0,   0.0,   0.0,     0.0),
     (0.0,  -1.0,   0.0,     0.0),
     (0.0,   0.0,  -1.0,     0.0)),

    ((c,   -c2,  -0.5,      0.0),
     (-c2,  -0.5,  -c,      0.0),
     (-0.5,  -c,  c2,       0.0)),

    ((c,  c2,  -0.5,        0.0),
     (c2,  -0.5,   c,       0.0),
     (-0.5,   c,  c2,       0.0)),

    ((-1.0,   0.0,   0.0,     0.0),
     (0.0,   1.0,   0.0,      0.0),
     (0.0,   0.0,  -1.0,      0.0)),

    ((-c2,   0.5,  -c,      0.0),
     (-0.5,   c,  c2,       0.0),
     (-c,   -c2,   0.5,     0.0)),

    ((0.0,  -1.0,   0.0,     0.0),
     (0.0,   0.0,  -1.0,     0.0),
     (1.0,   0.0,   0.0,     0.0)),

    ((c2,   0.5,   c,       0.0),
     (-0.5,  -c,  c2,       0.0),
     (-c,  c2,   0.5,       0.0)),

    ((-0.5,  -c,   -c2,     0.0),
     (-c,  c2,  -0.5,       0.0),
     (-c2,  -0.5,   c,      0.0)),

    ((c,   -c2,   0.5,      0.0),
     (c2,   0.5,  -c,       0.0),
     (-0.5,  -c,   -c2,     0.0)),

    ((-c2,  -0.5,   c,      0.0),
     (0.5,   c,  c2,        0.0),
     (c,   -c2,   0.5,      0.0)),

    ((-c2,   0.5,   c,      0.0),
     (0.5,  -c,  c2,        0.0),
     (-c,   -c2,  -0.5,     0.0)),

    ((c,  c2,   0.5,        0.0),
     (c2,  -0.5,  -c,       0.0),
     (0.5,  -c,  c2,        0.0)),

    ((-0.5,   c,   -c2,     0.0),
     (-c,   -c2,  -0.5,     0.0),
     (c2,  -0.5,  -c,       0.0)),

    ((0.0,   0.0,   1.0,     0.0),
     (-1.0,   0.0,   0.0,    0.0),
     (0.0,  -1.0,   0.0,     0.0)),

    ((0.5,   c,   -c2,      0.0),
     (c,   -c2,  -0.5,      0.0),
     (c2,   0.5,   c,       0.0)),

    ((0.5,  -c,   -c2,      0.0),
     (c,  c2,  -0.5,        0.0),
     (-c2,   0.5,  -c,      0.0)),

    ((c,   -c2,   0.5,      0.0),
     (-c2,  -0.5,   c,      0.0),
     (0.5,   c,  c2,        0.0)),

    ((-c2,  -0.5,   c,      0.0),
     (-0.5,  -c,   -c2,     0.0),
     (-c,  c2,  -0.5,       0.0)),

    ((-0.5,  -c,   -c2,     0.0),
     (c,   -c2,   0.5,      0.0),
     (c2,   0.5,  -c,       0.0)),

    ((0.5,  -c,   -c2,      0.0),
     (-c,   -c2,   0.5,     0.0),
     (c2,  -0.5,   c,       0.0)),

    ((0.0,   0.0,   1.0,     0.0),
     (1.0,   0.0,   0.0,     0.0),
     (0.0,   1.0,   0.0,     0.0)),

    ((0.5,   c,   -c2,      0.0),
     (-c,  c2,   0.5,       0.0),
     (-c2,  -0.5,  -c,      0.0)),

    ((0.0,   1.0,   0.0,     0.0),
     (0.0,   0.0,   1.0,     0.0),
     (1.0,   0.0,   0.0,     0.0)),

    ((-c2,  -0.5,  -c,      0.0),
     (0.5,   c,   -c2,      0.0),
     (-c,  c2,   0.5,       0.0)),

    ((c2,  -0.5,   c,       0.0),
     (0.5,  -c,   -c2,      0.0),
     (-c,   -c2,   0.5,     0.0)),

    ((c,  c2,   0.5,        0.0),
     (-c2,   0.5,   c,      0.0),
     (-0.5,   c,   -c2,     0.0)),

    ((-0.5,   c,   -c2,     0.0),
     (c,  c2,   0.5,        0.0),
     (-c2,   0.5,   c,      0.0)),

    ((-c2,   0.5,   c,      0.0),
     (-0.5,   c,   -c2,     0.0),
     (c,  c2,   0.5,        0.0)),

    ((c,   -c2,  -0.5,      0.0),
     (c2,   0.5,   c,       0.0),
     (0.5,   c,   -c2,      0.0)),

    ((c,  c2,  -0.5,        0.0),
     (-c2,   0.5,  -c,      0.0),
     (0.5,  -c,   -c2,      0.0)),

    ((-1.0,   0.0,   0.0,    0.0),
     (0.0,  -1.0,   0.0,     0.0),
     (0.0,   0.0,   1.0,     0.0)),

    ((-c,  c2,   0.5,       0.0),
     (-c2,  -0.5,  -c,      0.0),
     (0.5,   c,   -c2,      0.0)),

    ((-c,   -c2,   0.5,     0.0),
     (c2,  -0.5,   c,       0.0),
     (0.5,  -c,   -c2,      0.0)),

    )

    from . import Place, Places
    icos_matrices['222'] = Places([Place(m) for m in m222])
    for cs in coordinate_system_names:
        if cs != '222':
            t = coordinate_system_transform(cs, '222')
            icos_matrices[cs] = icos_matrices['222'].transform_coordinates(t)

    return icos_matrices

# -----------------------------------------------------------------------------
# Coordinates systems.
# '222'         2-fold symmetry along x, y, and z axes.
# '222r'        '222' with 90 degree rotation around z.
# '2n5'         2-fold symmetry along x and 5-fold along z.
# '2n5r'        '2n5' with 180 degree rotation about y.
# 'n25'         2-fold symmetry along y and 5-fold along z.
# 'n25r'        'n25' with 180 degree rotation about x.
# '2n3'         2-fold symmetry along x and 3-fold along z.
# '2n3r'        '2n3' with 180 degree rotation about y.
#
coordinate_system_names = ('222', '222r', '2n5', '2n5r', 'n25', 'n25r',
                           '2n3', '2n3r')

# -----------------------------------------------------------------------------
# Matrices for mapping between different icosahedron coordinate frames.
#
cst = {}
def coordinate_system_transform(from_cs, to_cs):

    global cst
    if cst:
        return cst[(from_cs, to_cs)]

    transform = cst

    e = icosahedron_edge_length() # Triangle edge length of unit icosahedron.

    from math import sqrt
    s25 = e/2                # Sin/Cos for angle between 2-fold and 5-fold axis
    c25 = sqrt(1 - s25*s25)
    s35 = e/sqrt(3)          # Sin/Cos for angle between 3-fold and 5-fold axis
    c35 = sqrt(1-s35*s35)

    from . import Place
    transform[('2n5','222')] = Place(((1,0,0,0),
                                      (0,c25,-s25,0),
                                      (0,s25,c25,0)))
    transform[('2n5','2n3')] = Place(((1, 0, 0, 0),
                                      (0, c35, s35, 0),
                                      (0, -s35, c35, 0)))

    # Axes permutations.
    transform[('222','222r')] = Place(((0,1,0,0),    # 90 degree rotation about z
                                       (-1,0,0,0),
                                       (0,0,1,0)))
    transform[('2n3','2n3r')] = \
    transform[('2n5','2n5r')] = Place(((-1,0,0,0),    # 180 degree rotation about y
                                       (0,1,0,0),
                                       (0,0,-1,0)))
    transform[('n25','n25r')] = Place(((1,0,0,0),     # 180 degree rotation about x
                                       (0,-1,0,0),
                                       (0,0,-1,0)))
    transform[('n25','2n5')] = Place(((0,1,0,0),      # x <-> y and z -> -z
                                      (1,0,0,0),
                                      (0,0,-1,0)))

    # Extend to all pairs of transforms.
    tlist = []
    while len(transform) > len(tlist):

        tlist = tuple(transform.keys())

        # Add inverse transforms
        for f,t in tlist:
            if not (t,f) in transform:
                transform[(t,f)] = transform[(f,t)].transpose()

        # Use transitivity
        for f1,t1 in tlist:
            for f2,t2 in tlist:
                if f2 == t1 and f1 != t2 and not (f1,t2) in transform:
                    transform[(f1,t2)] = transform[(f2,t2)] * transform[(f1,t1)]

    i = Place()
    for s in coordinate_system_names:
        transform[(s,s)] = i

    return transform[(from_cs, to_cs)]
