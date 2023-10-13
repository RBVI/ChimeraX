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
    'cross_products',
    'norm',
    'vector_angle',
    'inner_product_64',
    'inner_product',
    'distance',
    'distance_squared',
    'length',
]

from .matrix import vector_sum
from .matrix import normalize_vectors
from .matrix import normalize_vector
from .matrix import cross_product, cross_products
from .matrix import norm, length
from .matrix import vector_angle_radians as vector_angle

from ._geometry import distances_perpendicular_to_axis
from ._geometry import maximum_norm

'''Inner product of two vectors accumulated as a 64-bit float result.'''
from ._geometry import inner_product_64

from math import sqrt, acos, degrees, radians, sin, cos

def inner_product(u, v):
    '''Supported API. Return the inner product of two vectors.'''
    if len(u) == 3:
        # the below is considerably faster than even the C++ implementation,
        # as well as Python equivalents using zip() and sum()
        return u[0]*v[0] + u[1]*v[1] + u[2]*v[2]
    return inner_product_64(u, v)

def distance_squared(p, q):
    '''Supported API. Return the distance squared between two points.'''
    if len(p) == 3:
        # Much faster than using numpy operations.
        dx,dy,dz = p[0]-q[0], p[1]-q[1], p[2]-q[2]
        return dx*dx + dy*dy + dz*dz
    return sum((pi-qi)*(pi-qi) for pi,qi in zip(p,q))

def distance(p, q):
    '''Supported API. Return the distance between two points.'''
    return sqrt(distance_squared(p, q))

def interpolate_points(p1, p2, f):
    '''Supported API. Linearly interpolate from point p1 to p2 by fraction f (0 -> p1, 1 -> p2).'''
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
    '''
    Supported API.
    If p2 is not specified return the angle from origin spanned from point p0 to p1.
    If p2 is specified, return the angle made by segments p1 to p0 and p1 to p2.
    Returned angles are in degrees.
    '''
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
    return degrees(acos(acc))

def dihedral(p0, p1, p2, p3):
    '''
    Supported API.
    Return the dihedral angle defined by 4 points in degrees.
    '''
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

    radAngle = radians(angle)
    tmp = dist * sin(radAngle)
    radDihed = radians(dihed)
    pt = (tmp*sin(radDihed), tmp*cos(radDihed), dist*cos(radAngle))
    dp = pt[0]*x + pt[1]*y + pt[2]*v12 + n1
    return dp

def clip_segment(line, axis, offset1, offset2):
    '''
    Line defines a segment as two points that is to be clipped
    by slab offset1 <= (axis,p) <= offset2.
    The clipped line is returned.  If the clipped segment is empty
    then both returned segment endpoints are None.
    '''
    xyz1, xyz2 = line
    a1 = inner_product(axis, xyz1)
    a2 = inner_product(axis, xyz2)
    if (a1 < offset1 and a2 < offset1) or (a1 > offset2 and a2 > offset2):
        return None, None

    if a1 < offset1:
        cxyz1 = xyz1 + (offset1-a1)/(a2-a1) * (xyz2-xyz1)
    elif a1 > offset2:
        cxyz1 = xyz1 + (offset2-a1)/(a2-a1) * (xyz2-xyz1)
    else:
        cxyz1 = xyz1

    if a2 < offset1:
        cxyz2 = xyz2 + (offset1-a2)/(a1-a2) * (xyz1-xyz2)
    elif a2 > offset2:
        cxyz2 = xyz2 + (offset2-a2)/(a1-a2) * (xyz1-xyz2)
    else:
        cxyz2 = xyz2
        
    return cxyz1, cxyz2
