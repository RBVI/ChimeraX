#
# Placement operations involving rotations and translations.
#
from . import matrix as m34
class Place:
    def __init__(self, matrix = None, axes = None, shift = None):
        from numpy import array, float64, transpose
        if matrix is None:
            m = array(((1,0,0,0), (0,1,0,0), (0,0,1,0)), float64)
            if not axes is None:
                m[:,:3] = transpose(axes)
            if not shift is None:
                m[:,3] = shift
        else:
            m = array(matrix, float64)
        self.matrix = m

    def __mul__(self, p):
        if isinstance(p, Place):
            return Place(m34.multiply_matrices(self.matrix, p.matrix))

        from numpy import ndarray
        if isinstance(p, (ndarray, tuple, list)):
            return m34.apply_matrix(self.matrix, p)

        raise TypeError('Cannot multiply Place times "%s" %s' % str(p))

    def apply_without_translation(self, v):
        return m34.apply_matrix_without_translation(self.matrix, v)

    def move(self, xyz):
        m34.transform_points(xyz, self.matrix)

    def inverse(self):
        return Place(m34.invert_matrix(self.matrix))

    def transpose(self):
        m = self.matrix.copy()
        m[:,:3] = self.matrix[:,:3].transpose()
        return Place(m)

    def zero_translation(self):
        m = self.matrix.copy()
        m[:,3] = 0
        return Place(m)

    def opengl_matrix(self):
        return m34.opengl_matrix(self.matrix)

    def interpolate(self, tf, center, frac):
        return Place(m34.interpolate_transforms(self.matrix, center, tf.matrix, frac))

    #
    # Returns value in range 0 to pi.
    #
    def rotation_angle(self):

        m = self.matrix
        tr = m[0][0] + m[1][1] + m[2][2]
        cosa = .5 * (tr - 1)
        if cosa > 1:
            cosa = 1
        elif cosa < -1:
            cosa = -1
        from math import acos
        a = acos(cosa)
        return a

    def shift_and_angle(self, center):
        return m34.shift_and_angle(self.matrix, center)

    def axis_center_angle_shift(self):
        return m34.axis_center_angle_shift(self.matrix)

    def translation(self):
        return self.matrix[:,3]

    def axes(self):
        return self.matrix[:,:3].transpose()

    def z_axis(self):
        return self.matrix[:,2]

    def determinant(self):
        return m34.determinant(self.matrix)

    def description(self):
        return m34.transformation_description(self.matrix)

    def same(self, p, angle_tolerance = 0, shift_tolerance = 0):
        return m34.same_transform(self.matrix, p.matrix,
                                  angle_tolerance, shift_tolerance)

    def is_identity(self, tolerance = 1e-6):
        return m34.is_identity_matrix(self.matrix)

def translation(v):
    return Place(shift = v)

def rotation(axis, angle, center = None):
    return Place(m34.rotation_transform(axis, angle, center))

def vector_rotation(u,v):
    return Place(m34.vector_rotation_transform(u,v))

def orthonormal_frame(zaxis, ydir = None):
    return Place(axes = m34.orthonormal_frame(zaxis, ydir))

def skew_axes(cell_angles):
    return Place(axes = m34.skew_axes(cell_angles))

def cross_product(u):
    return Place(((0, -u[2], u[1], 0),
                  (u[2], 0, -u[0], 0),
                  (-u[1], u[0], 0, 0)))

def identity():
    return Place()
