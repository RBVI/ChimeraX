# vi: set expandtab shiftwidth=4 softtabstop=4:
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
    d = p - q
    from math import sqrt
    return sqrt((d * d).sum())

def interpolate_points(p1, p2, f):
    '''Linearly interpolate from point p1 to p2 by fraction f (0 -> p1, 1 -> p2).'''
    return (1-f)*p1 + f*p2
