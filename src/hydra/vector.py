from .matrix import vector_sum
from .matrix import normalize_vectors
from .matrix import normalize_vector
from .matrix import cross_product
from .matrix import norm
from .matrix import inner_product_64

def inner_product(u,v):
    return (u*v).sum()

def distance(p, q):
    d = p-q
    from math import sqrt
    return sqrt((d*d).sum())
