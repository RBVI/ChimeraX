"""
math3d: Support for coordinate-free geometry using numpy
========================================================

See `A Coordinate Free Geometry ADT <http://www.cs.uwaterloo.ca/research/tr/1997/15/CS-97-15.pdf>`_
and Nathan Litke's `Coordinate Free Geometric Programming <http://web.archive.org/web/*/http://www.cgl.uwaterloo.ca/~njlitke/geometry/>`_ for inspiration.

Points are "column" vectors.

TODO: Need to support general VRML transformations as well as
simple translation and pure rotation transformations.
"""

__all__ = [
    'Point', 'Vector', 'Xform', 'weighted_point',
    'Identity', 'Rotation', 'Translation',
    'transform',
    'look_at', 'ortho', 'frustum', 'perspective'
]

from numpy import ndarray, array, eye, dot, linalg, concatenate, allclose, isfortran, asfortranarray, sum
from math import sin, cos, tan, sqrt, radians

class Point(ndarray):
    """Floating point triplet representing a point"""

    def __new__(cls, data):
        p = ndarray.__new__(cls, (3,))
        p[0:3] = data
        return p

    def __add__(self, a):
        if isinstance(a, Point):
            raise ValueError("cannot add Points together")
        return ndarray.__add__(self, a)

    def __sub__(self, a):
        if isinstance(a, Point):
            v = ndarray.__sub__(self, a)
            v.__class__ = Vector
            return v
        return ndarray.__sub__(self, a)

    def __mul__(self, a):
        raise ValueError("cannot multiply Points")

    # TODO: add point operations

def weighted_point(points, weights=None):
    """return weighted sum of Points

    If no weights are given, then return average.
    """
    if weights is None:
        weight = len(points)
        p = sum(p.__array__() for p in points)
    else:
        from itertools import izip
        weight = sum(weights)
        p = sum(w * p.__array__() for w, p in izip(weights, points))
    p /= weight
    return Point(p)

class Vector(ndarray):
    """Floating point triplet representing a vector"""

    def __new__(cls, data):
        v = ndarray.__new__(cls, (3,))
        v[0:3] = data
        return v

    # TODO: add vector operations
    def __add__(self, a):
        if isinstance(a, Point):
            return ndarray.__add__(a, self)
        return ndarray.__add__(self, a)

    def __mul__(self, a):
        if isinstance(a, Vector):
            return dot(self, a)
        return Vector(self.__array__() * a)

    def length(self):
        """compute length of vector"""
        return sqrt(dot(self, self.conj()))

    def sqlength(self):
        """compute square length of vector"""
        return dot(self, self.conj())

    def normalize(self):
        """convert to unit length vector"""
        # faster than using numpy.linalg.norm()
        self /= sqrt(dot(self, self.conj()))

def cross(v1, v2):
    """Same as numpy.cross but return a Vector if arguments are Vectors"""
    from numpy import cross
    v = cross(v1, v2)
    if isinstance(v1, Vector) and isinstance(v2, Vector):
        return Vector(v)
    return v

class Xform:
    """4x4 matrix for homogeneous point and vector transformation
    
    When transformations are multiplied left-to-right, the right-most
    transformations transform the Point/Vector first, i.e., are more
    local.

    Member functions that make in-place modifications, left multiply
    the current matrix, i.e., make a global change.
    """

    def __init__(self, matrix, orthonormalize=True, _valid=False, _isIdentity=False, _pure=False, _projection=False):
        if isinstance(matrix, Xform):
            # C++ style copy constructor
            self._matrix = matrix._matrix.copy()
            self.isIdentity = matrix.isIdentity
            self._pure = matrix._pure
            self._projection = matrix._projection
            return
        self._matrix = matrix
        self.isIdentity = _isIdentity
        self._pure = _pure # True iff upper-left 3x3 is pure rotation
        self._projection = _projection
        if _valid:
            return
        if not isinstance(matrix, ndarray):
            matrix = array(matrix)
        if matrix.shape == (3, 4):
            matrix = concatenate((matrix, [[0, 0, 0, 1]]))
        assert matrix.shape == (4, 4), "not a 4x4 matrix"
        assert not matrix[3, 0:3].any() and matrix[3, 3] == 1, "perspective not supported yet"
        if isfortran(matrix):
            self._matrix = matrix
        else:
            self._matrix = asfortranarray(matrix)
        rot = matrix[0:3, 0:3]
        det = linalg.det(rot)
        if allclose(det, 0):
            raise TypeError("need non-degenerate rotation part")
        if allclose(det, 1):
            self._pure = True
        elif not orthonormalize:
            self._pure = False
        else:
            self._pure = False
            # TODO: orthonormalization

    def getOpenGLMatrix(self):
        m = self._matrix.astype('d')
        return m.flatten(order='f')

    def getOpenGLRotationMatrix(self):
        m = self._matrix[0:3, 0:3].astype('d')
        return m.flatten(order='f')

    def getWebGLMatrix(self):
        m = self._matrix.astype('f')
        return m.flatten(order='f')

    def getWebGLRotationMatrix(self):
        m = self._matrix[0:3, 0:3].astype('f')
        return m.flatten(order='f')

    def __mul__(self, xpv):
        """Apply transformation to Point or Vector

        Returns transformed Point or Vector.
        """
        # TODO: handle perspective?
        if isinstance(xpv, Xform):
            # TODO: multiple two xforms together
            if self.isIdentity:
                return Xform(xpv)
            if xpv.isIdentity:
                return Xform(self)
            m = dot(self._matrix, xpv._matrix)
            return Xform(m,
                    _pure = self._pure and xpv._pure,
                    _projection = self._projection or xpv._projection)
        if isinstance(xpv, Point):
            return Point(dot(self._matrix, concatenate((xpv, [1])))[0:3])
        if isinstance(xpv, Vector):
            return Vector(dot(self._matrix, concatenate((xpv, [0])))[0:3])
        if isinstance(xpv, ndarray) and xpv.shape == (4, 4):
            return Xform(dot(self._matrix, xpv))
        # TODO: handle arrays of Points and Vectors
        raise TypeError("expected a Xform, a Point, or a Vector")

    def rotate(self, axis, angle):
        """Further rotate"""
        rot = Rotation(axis, angle)
        self._matrix = dot(self._matrix, rot._matrix)

    def translate(self, offset):
        """Further translate"""
        rot = Translation(offset)
        self._matrix = dot(self._matrix, rot._matrix)

    #TODO: and more

def Identity():
    """Identify transformation"""
    return Xform(eye(4), _valid=True, _isIdentity=True)

def Rotation(axis, angle, inDegrees=False):
    """Build a rotation transformation"""
    assert(len(axis) == 3)
    if inDegrees:
        angle = radians(angle)
    sqlength = dot(axis, axis)
    if sqlength == 0:
        raise ValueError("can't rotate about zero vector")
    length = sqrt(sqlength)
    x = axis[0] / length
    y = axis[1] / length
    z = axis[2] / length
    s = sin(angle)
    c = cos(angle)
    t = 1 - c
    return Xform(array([
        [ t * x * x + c, t * x * y - s * z, t * x * z + s * y, 0 ],
        [ t * y * x + s * z, t * y * y + c, t * y * z - s * x, 0 ],
        [ t * z * x - s * y, t * z * y + s * x, t * z * z + c, 0 ],
        [ 0, 0, 0, 1 ]
    ]), _valid=True, _pure=True)

def Translation(offset):
    """Build a translation transformation"""
    assert(len(offset) == 3)
    t = eye(4)
    t[0:3, 3] = offset[0:3]
    return Xform(t, _valid=True, _pure=True)

def Scale(vector):
    """Build a scaling transformation"""
    assert(len(vector) == 3)
    s = eye(4)
    s[0][0] = vector[0]
    s[1][1] = vector[1]
    s[2][2] = vector[2]
    return Xform(s, _valid=True)

def transform(translation=None, center=None, rotation=None, scaleOrientation=None, scale=None):
    """X3D transformation specification into single transformation

    All angles must be in radians.
    """
    #    xf = t * c * r * sr * s * -sr * -c
    if scale is None:
        xf = Identity(4)
    elif scaleOrientation is not None:
        sc = Scale(scale)
        orient = Rotation(*scaleOrientation)
        invOrient = Rotation(scaleOrientation[0], -scaleOrientation[1])
        xf = orient * sc * invOrient
    else:
        xf = Scale(scale)
    if rotation is not None:
        xf = xf.rotate(*rotation)
        if center is not None:
            c = Translation(center)
            invC = Translation(-center)
            xf = c * xf * invC
    if translation is not None:
        xf = xf.translate(*translation)
    return xf

# look_at is like the OpenGL gluLookAt, it puts p0 at the origin,
# p1 on the negative Z axis, and p2 on the Y axis.

def look_at(p0, p1, p2):
    # Compute the rotational part of lookat matrix
    f = p1 - p0
    try:
        f.normalize()
    except ZeroDivisionError:
        raise ValueError("colinear points")
    up = p2 - p0
    s = cross(f, up)
    try:
        s.normalize()
    except ZeroDivisionError:
        raise ValueError("colinear points")
    u = cross(s, f)
    try:
        u.normalize()
    except ZeroDivisionError:
        raise ValueError("colinear points")

    xf = Xform(array([
            [s[0], s[1], s[2], 0],
            [u[0], u[1], u[2], 0],
            [-f[0], -f[1], -f[2], 0]
        ]), _pure=True)

    # Compute the translation component of matrices
    xlate = xf * p0
    xf._matrix[0:3, 3] = -xlate
    return xf

def ortho(left, right, bottom, top, hither, yon):
    """return 4x4 orthographic transformation"""
    # Matrices from OpenGL Programming Guide, 2nd Edition, p. 598
    # glOrtho(left, right, bottom, top, hither, yon);
    xf = Xform(array([
            [2 / (right - left), 0, 0, - (right + left) / (right - left)],
            [0, 2 / (top - bottom), 0, - (top + bottom) / (top - bottom)],
            [0, 0, -2 / (yon - hither), - (yon + hither) / (yon - hither)],
            [0, 0, 0, 1]
        ]), _valid=True, _projection=True)
    return xf

def frustum(left, right, bottom, top, hither, yon):
    """return 4x4 perspective transformation"""
    # Matrices from OpenGL Programming Guide, 2nd Edition, p. 598
    # glFrustum(left, right, bottom, top, hither, yon);
    xf = Xform(array([
            [2 * hither / (right - left), 0,
                    (right + left) / (right - left), 0],
            [0, 2 * hither / (top - bottom),
                    (top + bottom) / (top - bottom), 0],
            [0, 0, - (yon + hither) / (yon - hither),
                    -2 * yon * hither / (yon - hither)],
            [0, 0, -1, 0]
        ]), _valid=True, _projection=True)
    return xf

def perspective(fov_y, aspect, z_near, z_far):
    """return 4x4 perspective transformation"""
    f = 1 / tan(fov_y / 2)
    xf = Xform(array([
            [f / aspect, 0, 0, 0],
            [0, f, 0, 0],
            [0, 0, (z_far + z_near) / (z_near - z_far),
                    2 * z_far * z_near / (z_near - z_far)],
            [0, 0, -1, 0]
        ]), _valid=True, _projection=True)
    return xf
