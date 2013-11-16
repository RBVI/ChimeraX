# vi:set shiftwidth=4 expandtab:
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
    'Point', 'Vector', 'Xform', 'weighted_point', 'cross',
    'Identity', 'Rotation', 'Translation', 'Scale',
    'transform',
    'look_at', 'ortho', 'frustum', 'perspective',
    'camera_orientation',
    'BBox',
]

from numpy import (
    ndarray, array, eye, dot, linalg, concatenate, allclose, isfortran,
    asfortranarray, sum, float32, float64, amin, amax, array_equal,
)
from math import sin, cos, tan, acos, sqrt, radians, pi

EPSILON = 1.0e-10

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

    def __eq__(self, a):
        return array_equal(self, a)

    def close(self, a, **kw): # rtol=1e-05, atol=1e-08
        """Return if Point is close to another Point using numpy.allclose"""
        return allclose(self, a, **kw)

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
        length = sqrt(dot(self, self.conj()))
        if 0 - EPSILON <= length <= 0 + EPSILON:
            raise ArithmeticError("can not normalize zero length vector")
        self /= length

    def __eq__(self, a):
        return array_equal(self, a)

    def close(self, a, **kw): # rtol=1e-05, atol=1e-08
        """Return if Vector is close to another Vector using numpy.allclose"""
        return allclose(self, a, **kw)

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
            try:
                self.make_orthonormal()
            except:
                self._pure = False
                raise

    def getOpenGLMatrix(self):
        m = self._matrix.astype(float64)
        return m.flatten(order='F')

    def getOpenGLRotationMatrix(self):
        m = self._matrix[0:3, 0:3].astype(float64)
        return m.flatten(order='F')

    def getWebGLMatrix(self):
        m = self._matrix.astype(float32)
        return m.flatten(order='F')

    def getWebGLRotationMatrix(self):
        m = self._matrix[0:3, 0:3].astype(float32)
        return m.flatten(order='F')

    def get_translation(self):
        return Vector(self._matrix[0:3, 3])

    def get_rotation(self):
        rot = self._matrix[0:3, 0:3]
        cos_theta = (rot[0][0] + rot[1][1] + rot[2][2] - 1) / 2
        if 1 - EPSILON <= cos_theta <= 1 + EPSILON:
            angle = 0
            axis = Vector([0, 0, 1])
            return axis, angle

        if not (-1 - EPSILON <= cos_theta <= -1 + EPSILON):
            angle = acos(cos_theta)
            sin_theta = sqrt(1 - cos_theta * cos_theta)
            axis = Vector([
                (rot[2][1] - rot[1][2]) / 2 / sin_theta,
                (rot[0][2] - rot[2][0]) / 2 / sin_theta,
                (rot[1][0] - rot[0][1]) / 2 / sin_theta
            ])
            axis.normalize()
            return axis, angle

        angle = pi         # 180 degrees
        if rot[0][0] >= rot[1][1]:
            if rot[0][0] >= rot[2][2]:
                # rot00 is maximal diagonal term
                x = sqrt(rot[0][0] - rot[1][1] - rot[2][2] + 1) / 2
                half_inv = 1 / (2 * x)
                axis = Vector([x, half_inv * rot[0][1], half_inv * rot[0][2]])
            else:
                # rot22 is maximal diagonal term
                z = sqrt(rot[2][2] - rot[0][0] - rot[1][1] + 1) / 2
                half_inv = 1 / (2 * z)
                axis = Vector([half_inv * rot[0][2], half_inv * rot[1][2], z])
        else:
            if rot[1][1] >= rot[2][2]:
                # rot11 is maximal diagonal term
                y = sqrt(rot[1][1] - rot[0][0] - rot[2][2] + 1) / 2
                half_inv = 1 / (2 * y)
                axis = Vector([half_inv * rot[0][1], y, half_inv * rot[1][2]])
            else:
                # rot22 is maximal diagonal term
                z = sqrt(rot[2][2] - rot[0][0] - rot[1][1] + 1) / 2
                half_inv = 1 / (2 * z)
                axis = Vector([half_inv * rot[0][2], half_inv * rot[1][2], z])
        axis.normalize()
        return axis, angle

    def __mul__(self, arg):
        """Apply transformation to Point or Vector

        Returns transformed Point or Vector.
        """
        # TODO: handle perspective?
        if isinstance(arg, Xform):
            # TODO: multiple two xforms together
            if self.isIdentity:
                return Xform(arg)
            if arg.isIdentity:
                return Xform(self)
            m = dot(self._matrix, arg._matrix)
            return Xform(m,
                    _pure = self._pure and arg._pure,
                    _projection = self._projection or arg._projection)
        if isinstance(arg, Point):
            return Point(dot(self._matrix, concatenate((arg, [1])))[0:3])
        if isinstance(arg, Vector):
            return Vector(dot(self._matrix, concatenate((arg, [0])))[0:3])
        if isinstance(arg, ndarray) and arg.shape == (4, 4):
            return Xform(dot(self._matrix, arg))
        if isinstance(arg, BBox):
            from copy import copy
            bbox = copy(arg)
            bbox.xform(self)
            return bbox
        # TODO: handle arrays of Points and Vectors
        raise TypeError("expected a Xform, a Point, or a Vector")

    def rotate(self, axis, angle):
        """Further rotate"""
        rot = Rotation(axis, angle)
        self._matrix = dot(self._matrix, rot._matrix)
        self.isIdentity = False

    def translate(self, offset):
        """Further translate"""
        tran = Translation(offset)
        self._matrix = dot(self._matrix, tran._matrix)
        self.isIdentity = False

    def scale(self, vector):
        """Further scale"""
        sc = Scale(vector)
        self._matrix = dot(self._matrix, sc._matrix)
        self.isIdentity = False

    def invert(self):
        if self.isIdentity:
            return
        if self._projection:
            raise NotImplemented("can't invert projection matrices")

        if not self._pure:
            from numpy.linalg import inv
            self._matrix = inv(self._matrix)
        else:
            # swap off-diagonal section of rotation part
            self._matrix[0, 1], self._matrix[1, 0] = self._matrix[1, 0], self._matrix[0, 1]
            self._matrix[0, 2], self._matrix[2, 0] = self._matrix[2, 0], self._matrix[0, 2]
            self._matrix[1, 2], self._matrix[2, 1] = self._matrix[2, 1], self._matrix[1, 2]
            # reverse translation part
            self._matrix[0:3, 3] = - dot(self._matrix[0:3, 0:3], self._matrix[0:3, 3])

    def inverse(self):
        xf = Xform(self)
        xf.invert()
        return xf

    def make_orthonormal(self):
        """orthonormalize rotation part of transformation"""
        # Gram-Schmidt orthonormalization from Foley & van Dam, 2nd edition
        from numpy import linalg
        det = linalg.det(self._matrix)
        if 1 - EPSILON <= det <= 1 + EPSILON:
            self._pure = True
            return
        if 0 - EPSILON <= det <= 0 + EPSILON:
            raise ValueError("unable to orthonormalize rotation")

        try:
            r0 = Vector(self._matrix[0, 0:3])
            r0.normalize()
            r1 = Vector(self._matrix[1, 0:3])
            r1 = r1 - dot(r0, r1) * r0
            r1.normalize()
        except ArithmeticError:
            raise ValueError("unable to orthonormalize rotation")
        r2 = cross(r0, r1)
        self._matrix[0, 0:3] = r0
        self._matrix[1, 0:3] = r1
        self._matrix[2, 0:3] = r2
        self._pure = True

def Identity():
    """Identify transformation"""
    return Xform(eye(4), _valid=True, _isIdentity=True, _pure=True)

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

def look_at(p0, p1, up):
    """Compute viewing transformation

    Place :param p0: at origin, :param p1: on the negative z axis,
    and orient so :param up: is in the same half-plane as the postive y axis
    and z axis.  :param up: can either be a Point or a Vector.
    """
    # Compute the rotational part of lookat matrix
    f = p1 - p0
    try:
        f.normalize()
    except ZeroDivisionError:
        raise ValueError("colinear points")
    if isinstance(up, Point):
        up = up - p0
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

def camera_orientation(p0, p1, up):
    """Like look_at but p1 is at origin

    Place :param p0: on positive z axis origin, :param p1: at the origin,
    and orient so :param up: is in the same half-plane as the postive y axis
    and z axis.  :param up: can either be a Point or a Vector.
    """
    # Compute the rotational part of lookat matrix
    f = p1 - p0
    try:
        f.normalize()
    except ZeroDivisionError:
        raise ValueError("colinear points")
    if isinstance(up, Point):
        up = up - p1
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
    xlate = xf * p1
    xf._matrix[0:3, 3] = -xlate
    return xf

def ortho(left, right, bottom, top, hither, yon):
    """return 4x4 orthographic transformation"""
    # Matrices from OpenGL Programming Guide, 2nd Edition, p. 598
    # glOrtho(left, right, bottom, top, hither, yon)
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
    # glFrustum(left, right, bottom, top, hither, yon)
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

class BBox:
    """right-handed axis-aligned bounding box

    If either :py:attr:`BBox.llb` or :py:attr:`BBox.urf` are None,
    then the bounding box is uninitialized.
    """

    __slots__ = ['llb', 'urf']

    def __init__(self, llb=None, urf=None):
        self.llb = None	#: lower-left-back corner coordinates, a :py:class:`~chimera2.math3d.Point`
        self.urf = None	#: upper-right-front corner coordinates, a :py:class:`~chimera2.math3d.Point`
        if llb is not None:
            self.llb = Point(llb)
        if urf is not None:
            self.urf = Point(urf)

    def add(self, pt):
        """expand bounding box to encompass given point

        :param pt: a :py:class:`~chimera2.math3d.Point or other XYZ-tuple`
        """
        if self.llb is None:
            self.llb = Point(pt)
            self.urf = Point(pt)
            return
        for i in range(3):
            if pt[i] < self.llb[i]:
                self.llb[i] = pt[i]
            elif pt[i] > self.urf[i]:
                self.urf[i] = pt[i]

    def add_bbox(self, box):
        """expand bounding box to encompass given bounding box
        
        :param box: a :py:class:`BBox`
        """
        if self.llb is None:
            self.llb = box.llb
            self.urf = box.urf
            return
        for i in range(3):
            if box.llb[i] < self.llb[i]:
                self.llb[i] = box.llb[i]
            if box.urf[i] > self.urf[i]:
                self.urf[i] = box.urf[i]

    def bulk_add(self, pts):
        """expand bounding box to encompass all given points

        :param pts: a numpy array of XYZ coordinates
        """
        mi = amin(pts, axis=0)
        ma = amax(pts, axis=0)
        if self.llb is None:
            self.llb = Point(mi)
            self.urf = Point(ma)
            return
        for i in range(3):
            if mi[i] < self.llb[i]:
                self.llb[i] = mi[i]
            if ma[i] > self.urf[i]:
                self.urf[i] = ma[i]

    def center(self):
        """return center of bounding box
        
        :rtype: a :py:class:`~chimera2.math3d.Point`
        """
        if self.llb is None:
            raise ValueError("empty bounding box")
        return weighted_point([self.llb, self.urf])

    def size(self):
        """return length of sides of bounding box
        
        :rtype: a :py:class:`~chimera2.math3d.Vector`
        """
        if self.llb is None:
            raise ValueError("empty bounding box")
        return self.urf - self.llb

    def xform(self, xf):
        """transform bounding box in place"""
        if xf.isIdentity:
            return
        b = BBox([0., 0., 0.], [0., 0., 0.])
        for i in range(3):
            b.llb[i] = b.urf[i] = xf._matrix[i][3]
            for j in range(3):
                coeff = xf._matrix[i][j]
                if coeff == 0:
                    continue
                if coeff > 0:
                    b.llb[i] += self.llb[j] * coeff
                    b.urf[i] += self.urf[j] * coeff
                else:
                    b.llb[i] += self.urf[j] * coeff
                    b.urf[i] += self.llb[j] * coeff
        self.llb = b.llb
        self.urf = b.urf

    def merge(self, bbox):
        if self.llb is None:
            if bbox.llb is None:
                return
            self.llb = Point(bbox.llb)
            self.urf = Point(bbox.urf)
            return
        if bbox.llf[0] < self.llf[0]:
            self.llf[0] = bbox.llf[0]
        if bbox.llf[1] < self.llf[1]:
            self.llf[1] = bbox.llf[1]
        if bbox.llf[2] < self.llf[2]:
            self.llf[2] = bbox.llf[2]
        if bbox.urb[0] > self.urb[0]:
            self.urb[0] = bbox.urb[0]
        if bbox.urb[1] > self.urb[1]:
            self.urb[1] = bbox.urb[1]
        if bbox.urb[2] > self.urb[2]:
            self.urb[2] = bbox.urb[2]
