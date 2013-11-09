'''
Point and vector operations.
============================
These are convenience functions that act on points and vectors which are
one-dimensional numpy arrays of 3 elements, or for multiple points and vectors
two-dimensional numpy arrays of size N by 3.  Vectors of any dimensions
are allowed, not just 3-dimensional, except where noted.
'''

from .matrix import vector_sum
from .matrix import normalize_vectors
from .matrix import normalize_vector
from .matrix import cross_product
from .matrix import norm
from .matrix import vector_angle_radians as vector_angle

'''Inner product of two vectors accumulated as a 64-bit float result.'''
from .matrix import inner_product_64

def inner_product(u,v):
    '''Return the inner product of two vectors.'''
    return (u*v).sum()

def distance(p, q):
    '''Return the distance between two points.'''
    d = p-q
    from math import sqrt
    return sqrt((d*d).sum())
