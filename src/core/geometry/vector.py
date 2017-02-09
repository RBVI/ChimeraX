# vim: set expandtab shiftwidth=4 softtabstop=4:

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

'''
vector: Point and vector operations
===================================
These are convenience functions that act on points and vectors which are
one-dimensional numpy arrays of 3 elements, or for multiple points and vectors
two-dimensional numpy arrays of size N by 3.  Vectors of any dimensions
are allowed, not just 3-dimensional, except where noted.  Coordinate values
can be 32 or 64-bit floating point numbers.
'''

__all__ = [
    'vector_sum',
    'normalize_vectors',
    'normalize_vector',
    'cross_product',
    'norm',
    'vector_angle',
    'inner_product_64',
    'inner_product',
    'distance',
]

from .matrix import vector_sum
from .matrix import normalize_vectors
from .matrix import normalize_vector
from .matrix import cross_product
from .matrix import norm
from .matrix import vector_angle_radians as vector_angle

from ._geometry import distances_perpendicular_to_axis
from ._geometry import maximum_norm

'''Inner product of two vectors accumulated as a 64-bit float result.'''
from ._geometry import inner_product_64


def inner_product(u, v):
    '''Return the inner product of two vectors.'''
    return (u * v).sum()


def distance(p, q):
    '''Return the distance between two points.'''
    from math import sqrt
    if len(p) == 3:
        # Much faster than using numpy operations.
        dx,dy,dz = p[0]-q[0], p[1]-q[1], p[2]-q[2]
        return sqrt(dx*dx + dy*dy + dz*dz)
    d = sqrt(sum((pi-qi)*(pi-qi) for pi,qi in zip(p,q)))
    return d

def interpolate_points(p1, p2, f):
    '''Linearly interpolate from point p1 to p2 by fraction f (0 -> p1, 1 -> p2).'''
    return (1-f)*p1 + f*p2

def ray_segment(origin, direction, clip_planes):
    f0 = 0
    f1 = None
    for p, n in clip_planes:
        pon = inner_product(origin-p, n)
        dn = inner_product(direction, n)
        if dn != 0:
            f = -pon/dn
            if dn > 0:
                f0 = max(f0, f)
            else:
                f1 = f if f1 is None else min(f1, f)
    return f0, f1


def planes_as_4_vectors(triangles):
    '''
    Represent the plane of a triangle specified by 3 points
    as a 4-vector where the first 3 components are the normal
    and the 4th is an offset so that the equation of the plane
    is (v0,v1,v2,v3)*(x,y,z,1) = 0.
    '''
    n = len(triangles)
    from numpy import empty, float32
    p = empty((n,4), float32)
    for i, (a,b,c) in enumerate(triangles):
        n = normalize_vector(cross_product(b-a, c-a))
        p[i,:3] = n
        p[i,3] = -inner_product(n,a)
    return p

def angle(p0, p1, p2 = None):
    v0,v1 = (p0,p1) if p2 is None else (p0-p1,p2-p1)
    acc = inner_product(v0, v1)
    d0 = norm(v0)
    d1 = norm(v1)
    if d0 <= 0 or d1 <= 0:
        return 0
    acc /= (d0 * d1);
    if acc > 1:
        acc = 1
    elif acc < -1:
        acc = -1
    from math import acos, degrees
    return degrees(acos(acc))

def dihedral(p0, p1, p2, p3):
    v10 = p1 - p0
    v12 = p1 - p2
    v23 = p2 - p3
    t = cross_product(v10, v12)
    u = cross_product(v23, v12)
    v = cross_product(u, t);
    w = inner_product(v, v12)
    acc = angle(u, t)
    if w < 0:
        acc = -acc
    return acc

def dihedral_point(n1, n2, n3, dist, angle, dihed):
    '''Find dihedral point n0 with specified n0 to n1 distance,
    n0,n1,n2 angle, and n0,n1,n2,n3 dihedral (angles in degrees).'''

    v12 = n2 - n1
    v13 = n3 - n1
    v12 = normalize_vector(v12)
    x = normalize_vector(cross_product(v13, v12))
    y = normalize_vector(cross_product(v12, x))

    from math import radians, sin, cos
    radAngle = radians(angle)
    tmp = dist * sin(radAngle)
    radDihed = radians(dihed)
    pt = (tmp*sin(radDihed), tmp*cos(radDihed), dist*cos(radAngle))
    dp = pt[0]*x + pt[1]*y + pt[2]*v12 + n1
    return dp
