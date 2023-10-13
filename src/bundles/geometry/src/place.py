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
place: Coordinate systems
=========================

A coordinate system is represented by a
Place object which defines an origin and three axes specified relative
to another coordinate system.  A Place can specify the position and
orientation of a model in a scene, defining the local coordinate
system of the model relative to the global coordinate system of the
scene.  A Place can also represent the position and orientation
of a camera within a scene.  A Place can also be thought of as a
coordinate transformation mapping Place coordinates to the other
coordinate system.  The transform consists of a linear part (often a
rotation, but more generally a 3 by 3 matrix) followed by a shift
along the 3 axes.  Place objects use 64-bit coordinates for
axes and origin.


A point or vector is a one-dimensional numpy array of 3 floating point
values.  Multiple points or vectors are represented as two-dimensional
numpy arrays of size N by 3. Points and vectors can have 32-bit or 64-bit
floating point coordinates.

'''


from . import matrix as m34
from . import _geometry


class Place:
    '''
    The Place class gives the origin and axes vectors for a model in
    the global coordinate system.  A Place is often thought of as a
    coordinate transformation from local to global coordinates.

    The axes can be specified as a sequence of three axis vectors. The
    origin can be specified as a point.  Or both axes and origin can be
    specified as a 3 by 4 array where the first 3 columns are the axes
    and the last column is the origin.

    The axes do not need to be orthogonal, for instance a crystallographic
    density map may use skewed axes, and the axis vectors need not be of
    unit length.  The axes and origin are represented as 64-bit floating
    point values.
    '''
    def __init__(self, matrix=None, axes=None, origin=None):
        '''
        Supported API.
        Coordinate axes and origin can be specified as a 3 by 4 array
        where the first 3 columns are the axes vectors and the fourth
        column is the origin.  Alternatively axes can be specified as
        a list of axes vectors and the origin can be specifed as a vector.
        '''
        from numpy import array, float64, transpose
        if matrix is None:
            m = array(((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0)), float64)
            if axes is not None:
                m[:, :3] = transpose(axes)
            if origin is not None:
                m[:, 3] = origin
        else:
            m = array(matrix, float64, order = 'C')

        self._matrix = m
        '''
        Supported API.
        3 by 4 numpy 64-bit float array, first 3 columns are axes, last column is origin.
        '''

        self._is_identity = None # Cached boolean value whether matrix is identity
        self._inverse = None    # Cached inverse.
        self._m44 = None	# Cached 4x4 opengl matrix
        
    def copy(self):
        return Place(self._matrix)

    @property
    def matrix(self):
        '''Supported API. Returns a copy of the 3x4 float64 transformation matrix as a numpy array.'''
        return self._matrix.copy()
    
    def __eq__(self, p):
        '''Supported API.  Are matrix values of this Place equal to matrix values of another Place.'''
        return p is self or _geometry.same_matrix(p._matrix, self._matrix)

    def __mul__(self, p):
        '''
        Supported API.
        Multiplication of a Place and a point transforms from local
        point coordinates to global coordinates, the result being a
        new point.  Multiplication of a Place by another Place composes
        the coordinate transforms acting in right to left order producing
        a new Place object.
        '''
        
        if isinstance(p, Place):
            sp = _reuse_place()
            _geometry.multiply_matrices(self._matrix, p._matrix, sp._matrix)
            return sp
        elif isinstance(p, Places):
            return Places([self]) * p

        from numpy import ndarray
        if isinstance(p, (ndarray, tuple, list)):
            return m34.apply_matrix(self._matrix, p)

        from tinyarray import ndarray_int, ndarray_float
        if isinstance(p, (ndarray_float, ndarray_int)):
            return m34.apply_matrix(self._matrix, p)

        raise TypeError('Cannot multiply Place times "%s"' % str(p))

    def transform_points(self, xyz, in_place = False):
        '''
        Supported API.
        Returned transformed array of points. Makes a copy of points
        if place is not identity or in_place is False.
        '''
        if in_place:
            m34.transform_points(xyz, self._matrix)
            return xyz
        if self.is_identity():
            return xyz
        else:
            cxyz = xyz.copy()
            m34.transform_points(cxyz, self._matrix)
            return cxyz

    def transform_vector(self, v):
        '''
        Supported API.
        Apply the linear part (3x3 matrix) of the transform without the shift.
        '''
        return m34.apply_matrix_without_translation(self._matrix, v)
    apply_without_translation = transform_vector
    
    def transform_vectors(self, v, in_place = False):
        '''
        Supported API.
        Apply the linear part (3x3 matrix) of the transform without the shift.
        A new copy of the numpy Nx3 array of vectors is returned unless in_place = True.
        '''
        if in_place:
            m34.transform_vectors(v, self._matrix)
            return v

        return m34.apply_matrix_without_translation(self._matrix, v)

    def transform_normals(self, xyz, in_place = False, is_rotation = False):
        '''
        Supported API.
        Apply inverse transpose of transform with zero shift to an array of normal vectors,
        modifying the vectors in place.  If is_rotation is true then the transform is assumed
        to be a rotation with no scaling or skewing so that the inverse transpose is the same
        as the original transform, allowing for faster computation.
        '''
        if not in_place:
            xyz = xyz.copy()
        if is_rotation:
            m34.transform_vectors(xyz, self._matrix)
        else:
            m34.transform_normals(xyz, self._matrix)
        return xyz

    def inverse(self, is_orthonormal = False):
        '''
        Supported API.
        Return the inverse transform. If is_orthonormal  is True then the transform
        is assumed to be orthonormal so the inverse is the transpose, allowing faster computation.
        '''
        if self._inverse is None:
            if is_orthonormal:
                self._inverse = result = _reuse_place()
                _geometry.invert_orthonormal(self._matrix, result._matrix)
            else:
                self._inverse = Place(m34.invert_matrix(self._matrix))
        return self._inverse

    def transpose(self):
        '''
        Supported API.
        Return a copy of the transform with the linear part transposed.
        '''
        m = self._matrix.copy()
        m[:, :3] = self._matrix[:, :3].transpose()
        return Place(m)

    def zero_translation(self):
        '''Supported API. Return a copy of the transform with zero shift.'''
        m = self._matrix.copy()
        m[:, 3] = 0
        return Place(m)

    def scale_translation(self, s):
        '''Return a copy of the transform with scaled shift.'''
        m = self._matrix.copy()
        m[:, 3] *= s
        return Place(m)

    def scale_factor(self):
        '''
        Scale factor assuming a rotation times a scale factor.
        '''
        m = self._matrix
        from math import sqrt
        s = sqrt(m[0,0]*m[0,0] + m[1,0]*m[1,0] + m[2,0]*m[2,0])
        return s

    def remove_scale(self):
        '''
        Return a copy of the transform with unit scale factor
        assuming it is a rotation times a scale factor.
        '''
        s = self.scale_factor()
        m = self._matrix.copy()
        m[:3,:3] *= 1/s
        return Place(m)
        
    def opengl_matrix(self):
        '''
        Supported API.
        Return a numpy float32 4x4 array which is the transformation matrix
        in OpenGL order (columns major).
        '''
        m = self._m44
        if m is None:
            self._m44 = m = _geometry.opengl_matrix(self._matrix)  # float32
        return m

    def interpolate(self, tf, center, frac):
        '''
        Supported API.
        Interpolate from this transform to the specified one by
        a specified fraction.  A fraction of 0 gives this transform
        while a fraction 1 gives transform tf.  The interpolation is
        done by finding the transform mapping to tf, treating it as a
        rotation about the specified center point followed by a shift,
        then perform the specified fraction of the full rotation, and
        specified fraction of the shift.  When the interpolated places
        are thought of as coordinate systems positions, the center
        point is in local coordinates.
        '''
        return Place(m34.interpolate_transforms(self._matrix, center, tf._matrix, frac))

    def rotation_angle(self):
        '''
        Supported API.
        Return the rotation angle of the transform, or equivalently
        the rotation of the coordinate axes from the global axes.
        The return values is in radians from 0 to pi.  The rotation is
        about the unique axis that takes the local to global coordinates.
        This assumes the transform is a rotation, or equivalently that
        the coordinate axes are orthonormal and right handed.
        '''
        m = self._matrix
        tr = m[0][0] + m[1][1] + m[2][2]
        cosa = .5 * (tr - 1)
        if cosa > 1:
            cosa = 1
        elif cosa < -1:
            cosa = -1
        from math import acos
        a = acos(cosa)
        return a

    def rotation_axis_and_angle(self):
        '''Supported API. Return the rotation axis and angle (degrees) of the transform.'''
        return m34.rotation_axis_angle(self._matrix)

    def shift_and_angle(self, center):
        '''
        Supported API.
        Return the shift distance and rotation angle for the transform.
        The rotation angle is in radians the same as returned by
        rotation_angle(), and the shift is the distance the given center
        point is moved by the transformation.
        '''
        return m34.shift_and_angle(self._matrix, center)

    def axis_center_angle_shift(self):
        '''
        Supported API.
        Parameterize the transform as a rotation about a point around
        an axis and shift along that axis.  Return the rotation axis,
        a point on the axis, rotation angle (in degrees), and shift
        distance parallel to the axis.  This assumes the transformation
        linear part is a rotation.
        '''
        return m34.axis_center_angle_shift(self._matrix)

    def rotation_quaternion(self):
        '''Supported API. Return the quaternion of the rotation part of the transform.'''
        axis, angle = self.rotation_axis_and_angle()
        from math import pi, sin, cos
        a = angle * pi/180
        sa2 = sin(a/2)
        ca2 = cos(a/2)
        from numpy import array, float64
        q = array((ca2, sa2*axis[0], sa2*axis[1], sa2*axis[2]), float64)
        return q

    def translation(self):
        '''
        Supported API.
        Return the transformation shift vector, or equivalently the
        coordinate system origin.
        '''
        return self._matrix[:, 3].copy()

    def origin(self):
        '''
        Supported API.
        Return the transformation shift vector, or equivalently the
        coordinate system origin.
        '''
        return self._matrix[:, 3].copy()

    def axes(self):
        '''Supported API. Return the coordinate system axes.'''
        return self._matrix[:, :3].transpose().copy()

    def axes_lengths(self):
        '''Supported API. Return the lengths of the axes vectors.'''
        m = self._matrix
        from . import norm
        return [norm(m[:,a]) for a in (0,1,2)]

    def axis_length(self, axis_number):
        '''Supported API. Return the length of the axix (0,1 or 2 for x,y,z).'''
        m = self._matrix
        from . import norm
        return norm(m[:,axis_number])

    def z_axis(self):
        '''Supported API. Return the coordinate system z axis.'''
        return self._matrix[:, 2].copy()

    def determinant(self):
        '''Supported API. Return the determinant of the linear part of the transformation.'''
        return m34.determinant(self._matrix)

    def _polar_decomposition(self):
        '''
        Don't use. This code won't handle degenerate or near degenerate transforms.
        Return 4 transforms, a translation, rotation, scaling, orthonormal scaling axes
        whose product equals this transformation.  Needed for X3D output.
        '''
        from numpy import linalg
        a = self.zero_translation()
        at = a.transpose()
        ata = at*a
        eval, evect = linalg.eigh(ata._matrix[:3,:3])
        from math import sqrt
        s = scale([sqrt(e) for e in eval])
        sinv = s.inverse()
        ot = Place(axes = evect)
        o = ot.transpose()
        r = a*o*sinv
        t = translation(self.translation())
        return t, r, s, ot
        
    def description(self):
        '''
        Supported API.
        Return a text description of the transformation including
        the 3 by 4 matrix, and the decomposition as a rotation about a
        point through an axis and shift along that axis.
        '''
        return m34.transformation_description(self._matrix)

    def same(self, p, angle_tolerance=0, shift_tolerance=0):
        '''
        Supported API.
        Is this transform the same as the given one to within a
        rotation angle tolerance (degrees), and shift tolerance (distance)
        '''
        return m34.same_transform(self._matrix, p._matrix,
                                  angle_tolerance, shift_tolerance)

    def is_identity(self, tolerance=0):
        '''
        Supported API.
        Is the transform the identity transformation?  Tests if each
        of the 3 by 4 matrix elements is within the specified tolerance
        of the identity transform.
        '''
        if tolerance == 0:
            ii = self._is_identity
            if ii is None:
                self._is_identity = ii = _geometry.is_identity_matrix(self._matrix, tolerance)
        else:
            ii = _geometry.is_identity_matrix(self._matrix, tolerance)
        return ii

    def _reuse(self):
        self._is_identity = None
        self._inverse = None
        self._m44 = None


'''
The following routines create Place objects representing specific
transformations.
'''

def translation(v):
    '''Supported API. Return a transform which is a shift by vector v.'''
    p = _reuse_place(create = False)
    if p:
        _geometry.set_translation_matrix(v, p._matrix)
    else:
        p = Place(origin=v)
    return p


def rotation(axis, angle, center=(0, 0, 0)):
    '''
    Supported API.
    Return a transform which is a rotation about the specified center
    and axis by the given angle (degrees).
    '''
    return Place(m34.rotation_transform(axis, angle, center))


def quaternion_rotation(q, shift=(0, 0, 0)):
    '''
    Supported API.

    Return a transform which is a rotation for quaternion (q0, q1, q2, q3)
    followed by a translation.
    '''
    a,b,c,d = q
    a2,b2,c2,d2 = a*a,b*b,c*c,d*d
    bc,da,bd,ca,cd,ba = b*c,d*a,b*d,c*a,c*d,b*a
    from math import sqrt
    s2 = 2 / sqrt(a2+b2+c2+d2)
    sx,sy,sz = shift
    m = ((1 - s2*(c2 + d2), s2*(bc - da), s2*(bd + ca), sx),
         (s2*(bc + da), 1 - s2*(b2 + d2), s2*(cd - ba), sy),
         (s2*(bd - ca), s2*(cd + ba), 1 - s2*(b2 + c2), sz))
    return Place(m)


def vector_rotation(u, v):
    '''
    Supported API. 
    Return a rotation transform taking vector u to vector v by rotation
    about an axis perpendicular to u and v.  The vectors can have any
    length and the transform maps the direction of u to the direction
    of v.
    '''
    return Place(m34.vector_rotation_transform(u, v))


def scale(s):
    '''Supported API. Return a transform which is a scale by factor s.'''
    p = _reuse_place()
    v = (s,s,s) if isinstance(s, (float, int)) else s
    _geometry.set_scale_matrix(v, p._matrix)
    return p


def orthonormal_frame(zaxis, ydir=None, xdir=None, origin=None):
    '''
    Supported API. 
    Return a Place object with the specified z axis.  Any rotation
    about that z axis is allowed, unless a vector ydir is given in which
    case the y axis will be in the plane define by the z axis and ydir.
    If the zaxis is 0 then (0,0,1) is used.
    '''
    axes = m34.orthonormal_frame(zaxis, ydir, xdir)
    result = _reuse_place()
    o0,o1,o2 = (0,0,0) if origin is None else origin
    result._matrix[:] = ((axes[0][0], axes[1][0], axes[2][0], o0),
                        (axes[0][1], axes[1][1], axes[2][1], o1),
                        (axes[0][2], axes[1][2], axes[2][2], o2))
    return result


def skew_axes(cell_angles):
    '''
    Supported API. 
    Return a Place object representing the skewing with cell angles
    alpha, beta and gamma (degrees).  The first axis is (1, 0, 0), the
    second axis makes angle gamma with the first and is in the xy plane,
    and the third axis makes angle beta with the first and angle alpha
    with the second axes.  The axes are unit length and form a right
    handed coordinate system.  The angles alpha, beta and gamma are the
    angles between yz, xz, and xy axes.
    '''
    return Place(axes=m34.skew_axes(cell_angles))


def cross_product_matrix(u):
    '''
    Return the transform representing vector crossproduct with vector u
    (on the left) and zero shift.
    '''
    return Place(((0, -u[2], u[1], 0),
                  (u[2], 0, -u[0], 0),
                  (-u[1], u[0], 0, 0)))


_identity_place = None
def identity():
    '''Supported API. Return the identity transform.'''
    global _identity_place
    if _identity_place is None:
        _identity_place = Place()
    return _identity_place

def product(plist):
    '''Supported API. Product of a sequence of Place transforms.'''
    p = plist[0]
    for p2 in plist[1:]:
        if not p2.is_identity():
            p = p*p2
    return p

def interpolate_rotation(place1, place2, fraction):
    '''
    Supported API. 
    Interpolate the rotation parts of place1 and place2.
    The rotation axis taking one coordinate frame to the other
    is linearly interpolated. The translations are ignored
    and the returned Place has translation set to zero.
    '''
    r1, r2 = place1.zero_translation(), place2.zero_translation()
    center = (0,0,0)
    return r1.interpolate(r2, center, fraction)

# look_at is called a lot when finding hbonds and having the import inside the
# function is actually a non-trivial cost
from ._geometry import look_at as  c_look_at
def look_at(from_pt, to_pt, up):
    '''
    Supported API. 
    Return a Place object that represents looking from 'from_pt' to 'to_pt'
    with 'up' pointing up.
    '''
    return Place(matrix=c_look_at(from_pt, to_pt, up))

def z_align(pt1, pt2):
    '''
    Supported API. 
    Return a Place object that puts the pt1->pt2 vector on the Z axis
    '''
    a, b, c = pt2 - pt1
    l = a * a + c * c
    d = l + b * b
    epsilon = 1e-10
    if abs(d) < epsilon:
        raise ValueError("z_align endpoints must be distinct")
    from math import sqrt
    l = sqrt(l)
    d = sqrt(d)

    from numpy import array, float64
    xf = array(((0, 0, 0), (0, 0, 0), (0, 0, 0)), float64)
    xf[1][1] = l / d
    if abs(l) < epsilon:
        xf[0][0] = 1.0
        xf[2][1] = -b / d
    else:
        xf[0][0] = c / l
        xf[2][0] = -a / l
        xf[0][1] = -(a * b) / (l * d)
        xf[2][1] = -(b * c) / (l * d)
    xf[0][2] = a / d
    xf[1][2] = b / d
    xf[2][2] = c / d

    import numpy
    xlate = numpy.negative(numpy.dot(numpy.transpose(xf), pt1))
    return Place(axes=xf, origin=xlate)

def transform_planes(coord_sys, planes):
    '''Planes are given by 4 vectors v defining plane v0*x + v1*y + v2*z + v3 = 0.
    Returns planes in new coordinate system.'''
    if coord_sys.is_identity():
        return planes
    cp = planes.copy()
    ct = coord_sys.transpose()
    t = coord_sys.translation()
    for p in range(len(planes)):
        v = planes[p,:3]
        cp[p,:3] = ct.transform_vector(v)
        cp[p,3] = planes[p,3] + (t * v).sum()
    return cp

from sys import getrefcount
_recent_place_instances = []
_max_recent_place_instances = 10
def _reuse_place(create = True):
    '''
    Keep a cache of Place instances and reuse them if they are no longer used.
    This is to improve performance when using many temporary matrices, so it is not
    required to allocate to numpy arrays and Place instances.
    '''
    global _recent_place_instances
    for p in _recent_place_instances:
        if getrefcount(p) == 3:
            p._reuse()
            return p
    if create:
        p = Place()
        _recent_place_instances.insert(0, p)
        if len(_recent_place_instances) > _max_recent_place_instances:
            _recent_place_instances = _recent_place_instances[:_max_recent_place_instances]
    else:
        p = None
    return p

    
class Places:
    '''
    The Places class represents a list of 0 or more Place objects.
    The advantage of Places over using a list of Place objects is that
    it doesn't need to create a separate Python Place object for each
    position, instead it is able to represent for example 10,000 atom
    positions as a numpy array of positioning matrices.  So this class is
    primarily to allow efficient handling of large numbers of positions.
    '''
    def __init__(self, places=None, place_array=None, shift_and_scale=None,
                 opengl_array=None):
        '''
        Supported API. 
        Places can be specified as a list of Place instances, or a numpy N x 3 x 4 array,
        or a shift and scale N x 4 array, or a 4 x 4 numpy array (opengl format).
        '''
        if (place_array is not None or shift_and_scale is not None
                or opengl_array is not None):
            pl = None
        elif places is None:
            pl = [Place()]
        else:
            pl = list(places)
        self._place_list = pl
        self._place_array = place_array
        self._opengl_array = opengl_array
        self._shift_and_scale = shift_and_scale

        # Check that arrays are the right type.
        if place_array is not None:
            self._check_array('place_array', place_array, double=True)
        if opengl_array is not None:
            self._check_array('opengl_array', opengl_array)
        if shift_and_scale is not None:
            self._check_array('shift_and_scale', shift_and_scale)

    def _check_array(self, name, array, double = False, contiguous = True):
        from numpy import ndarray, float32, float64
        dtype = float64 if double else float32
        if not isinstance(array, ndarray):
            raise ValueError('Places %s argument must be a numpy array, got %s' % (name, str(type(array))))
        if array.dtype != dtype:
            raise ValueError('Places %s argument must have type %s, got %s' % (name, str(dtype), str(array.dtype)))
        if contiguous and not array.flags['C_CONTIGUOUS']:
            raise ValueError('Places %s argument must be contiguous' % name)
        
    def place_list(self):
        '''Supported API. Return a list of Place instances.'''
        pl = self._place_list
        if pl is None:
            if self._place_array is not None:
                pl = tuple(Place(m) for m in self._place_array)
            elif self._shift_and_scale is not None:
                pl = tuple(Place(((s[3], 0, 0, s[0]),
                                  (0, s[3], 0, s[1]),
                                  (0, 0, s[3], s[2])))
                           for s in self._shift_and_scale)
            elif self._opengl_array is not None:
                pl = tuple(Place(m.transpose()[:3, :])
                           for m in self._opengl_array)
            else:
                pl = []
            self._place_list = pl
        return pl

    def array(self):
        '''Supported API. Return a numpy float64 N x 3 x 4 array.'''
        pa = self._place_array
        if pa is None:
            from numpy import empty, float64
            pa = empty((len(self),3,4), float64)
            if self._place_list is not None:
                for i,p in enumerate(self._place_list):
                    pa[i,:,:] = p._matrix
            elif self._shift_and_scale is not None:
                sas = self._shift_and_scale
                pa[:] = 0
                pa[:,:,3] = sas[:,:3]
                pa[:,0,0] = pa[:,1,1] = pa[:,2,2] = sas[:,3]
            elif self._opengl_array is not None:
                pa[:] = self._opengl_array.transpose((0,2,1))[:,:3,:]
            self._place_array = pa
        return pa

    def masked(self, mask):
        '''
        Supported API.
        Return a Places instance containing only the places
        where boolean mask array is true.  The mask array must have length equal
        to the number of places.
        '''
        if mask is None:
            return self
        sas = self._shift_and_scale
        if not sas is None:
            p = Places(shift_and_scale = sas[mask])
        else:
            oa = self._opengl_array
            if oa is None:
                p = Places(place_array=self.array()[mask])
            else:
                p = Places(opengl_array = oa[mask])
        return p

    def shift_and_scale_array(self):
        '''
        Supported API.
        Only valid if this Places was specified using a shift and scale matrix,
        otherwise None is returned.
        '''
        return self._shift_and_scale

    def opengl_matrices(self):
        '''
        Supported API.
        Return array of 4x4 float32 matrices with column-major order.
        '''
        m = self._opengl_array
        if m is None:
            m = _geometry.opengl_matrices(self.array(), len(self))
            self._opengl_array = m
        return m

    def __getitem__(self, i):
        return self.place_list()[i]

    def __setitem__(self, i, p):
        self._place_list[i] = p

    def __len__(self):
        '''Supported API. Number of places.'''
        if self._place_list is not None:
            n = len(self._place_list)
        elif self._place_array is not None:
            n = len(self._place_array)
        elif self._shift_and_scale is not None:
            n = len(self._shift_and_scale)
        elif self._opengl_array is not None:
            n = len(self._opengl_array)
        return n

    def __iter__(self):
        '''Supported API. Iterator returning Place instances.'''
        return self.place_list().__iter__()
    
    def __eq__(self, p):
        '''
        Supported API.
        Test if these places identical in order and matrix elements.
        '''
        if p is self:
            return True
        if len(p) != len(self):
            return False
        for p0,p in zip(self, p):
            if p != p0:
                return False
        return True

    def __mul__(self, places_or_vector):
        '''
        Supported API.
        Can multiply by another Places which produces all-by-all products of matrices,
        or can multiply by a single Place, or can multiply by a single vector or array of vectors.
        '''
        if isinstance(places_or_vector, Places):
            places = places_or_vector
            r = _geometry.multiply_matrix_lists(self.array(), len(self),
                                                places.array(), len(places))
            return Places(place_array = r)
        elif isinstance(places_or_vector, Place):
            place = places_or_vector
            r = _geometry.multiply_matrix_lists(self.array(), len(self),
                                                place._matrix.reshape((1,3,4)), 1)
            return Places(place_array = r)
        else:
            from numpy import array, float32, dot, empty
            a = self.array()
            if len(a) == 0:
                return empty((0,3),float32)
            v = places_or_vector
            if v.ndim == 1:
                v4 = array((v[0], v[1], v[2], 1.0), float32)
                pv = dot(a, v4)
            elif v.ndim == 2:
                pv2 = dot(v, a[:,:,:3].transpose(0,2,1))
                pv2 += a[:,:,3]
                pv = pv2.reshape((len(a)*len(v), 3))
            else:
                raise ValueError('Multiplication of Places times array shape %s not supported'
                                 % ','.join('%d' % s for s in v.shape))
            return pv

    def is_identity(self):
        '''
        Supported API.
        Whether this Places contains just one place which is the identity.
        '''
        return len(self) == 1 and self[0].is_identity()

    def transform_coordinates(self, csys):
        '''
        Supported API.
        Transform these places to a new coordinate system.
        Argument csys maps new coordinates to old coordinates.
        '''
        if csys.is_identity():
            return self
        csys_inv = csys.inverse()
        return Places([csys_inv*p*csys for p in self])

    def _resize(self, n):
        if len(self) == n:
            return
        from numpy import empty, float64
        self._place_array = empty((n,3,4), float64)
        self._place_list = None
        self._opengl_array = None
        self._shift_and_scale = None

    def _matrices_changed(self):
        self._place_list = None
        self._shift_and_scale = None
        oa = self._opengl_array
        if oa is not None:
            _geometry.opengl_matrices(self.array(), len(self), oa)

    
def multiply_transforms(tf1, tf2, result = None):
    '''
    Experimental API.
    Multiply Place or Places times another Place or Places.
    Returns tf1 * tf2.  If the result argument is specified (Place or Places)
    it modifies the matrices of that instance. This violates the assumption that
    Place and Places are immutable, and is only done for optimization purposes.
    Use of this method is discouraged.  Instead use tf1 * tf2.
    '''
    if result is None:
        return tf1 * tf2

    # Set result transform.
    if isinstance(tf1, Place) and isinstance(tf2, Place):
        _geometry.multiply_matrices(tf1._matrix, tf2._matrix, result._matrix)
        result._reuse()
    elif isinstance(tf1, Places) and isinstance(tf2, Place):
        n = len(tf1)
        result._resize(n)
        _geometry.multiply_matrix_lists(tf1.array(), n, tf2._matrix.reshape((1,3,4)), 1, result.array())
        result._matrices_changed()
    elif isinstance(tf1, Place) and isinstance(tf2, Places):
        n = len(tf2)
        result._resize(n)
        _geometry.multiply_matrix_lists(tf1._matrix.reshape((1,3,4)), 1, tf2.array(), n, result.array())
        result._matrices_changed()
    elif isinstance(tf1, Places) and isinstance(tf2, Places):
        n1, n2 = len(tf1), len(tf2)
        result._resize(n1*n2)
        _geometry.multiply_matrix_lists(tf1.array(), n1, tf2.array(), n2, result.array())
        result._matrices_changed()
    else:
        raise ValueError('Arguments must be Place or Places.  Got %s and %s.'
                         % (str(type(tf1)), str(type(tf2))))

    return result
