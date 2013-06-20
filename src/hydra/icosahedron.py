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
    tf5 = ((c5, -s5, 0, 0),
           (s5, c5, 0, 0),
           (0, 0, 1, 0))

    # 2-fold symmetry axis along x
    tf2 = ((1, 0, 0, 0),
           (0, -1, 0, 0),
           (0, 0, -1, 0))

    p = (0, 0, 1)
    p50 = (0, sin(a), cos(a))
    from .matrix import apply_matrix
    p51 = apply_matrix(tf5, p50)
    p52= apply_matrix(tf5, p51)
    p53 = apply_matrix(tf5, p52)
    p54 = apply_matrix(tf5, p53)
    vertices = [p, p50, p51, p52, p53, p54]
    vertices.extend([apply_matrix(tf2, q) for q in vertices])

    if orientation != '2n5':
        tf = coordinate_system_transform('2n5', orientation)
        vertices = [apply_matrix(tf, p) for p in vertices]
    
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

    from numpy import array, float32, uint32
    va = array(vertices, float32)
    ta = array(triangles, uint32)
    return va, ta

# -----------------------------------------------------------------------------
# 60 icosahedral symmetry matrices.
#
def icosahedral_symmetry_matrices(orientation = '222', center = (0,0,0)):

    t = icosahedral_matrix_table()
    import Symmetry as S
    tflist = S.recenter_symmetries(t.get(orientation, None), center)
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

    icos_matrices['222'] = (

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

    from .matrix import multiply_matrices
    for cs in coordinate_system_names:
        if cs != '222':
            t = coordinate_system_transform(cs, '222')
            tinv = coordinate_system_transform('222', cs)
            icos_matrices[cs] = [multiply_matrices(tinv, m, t)
                                 for m in icos_matrices['222']]
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

    transform[('2n5','222')] = ((1,0,0,0),
                                (0,c25,-s25,0),
                                (0,s25,c25,0))
    transform[('2n5','2n3')] = ((1, 0, 0, 0),
                                (0, c35, s35, 0),
                                (0, -s35, c35, 0))

    # Axes permutations.
    transform[('222','222r')] = ((0,1,0,0),    # 90 degree rotation about z
                                 (-1,0,0,0),
                                 (0,0,1,0))
    transform[('2n3','2n3r')] = \
    transform[('2n5','2n5r')] = ((-1,0,0,0),    # 180 degree rotation about y
                                 (0,1,0,0),
                                 (0,0,-1,0))
    transform[('n25','n25r')] = ((1,0,0,0),     # 180 degree rotation about x
                                 (0,-1,0,0),
                                 (0,0,-1,0))
    transform[('n25','2n5')] = ((0,1,0,0),      # x <-> y and z -> -z
                                 (1,0,0,0),
                                 (0,0,-1,0))

    # Extend to all pairs of transforms.
    tlist = []
    from .matrix import transpose_matrix, multiply_matrices, identity_matrix
    while len(transform) > len(tlist):

        tlist = transform.keys()

        # Add inverse transforms
        for f,t in tlist:
            if not (t,f) in transform:
                transform[(t,f)] = transpose_matrix(transform[(f,t)])

        # Use transitivity
        for f1,t1 in tlist:
            for f2,t2 in tlist:
                if f2 == t1 and f1 != t2 and not (f1,t2) in transform:
                    transform[(f1,t2)] = multiply_matrices(transform[(f2,t2)],
                                                           transform[(f1,t1)])

    i = identity_matrix()
    for s in coordinate_system_names:
        transform[(s,s)] = i

    return transform[(from_cs, to_cs)]
