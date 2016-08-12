# vim: set expandtab shiftwidth=4 softtabstop=4:

import numpy
from numpy.linalg import norm
normalize = lambda v: v/norm(v)
sqlength = lambda v: numpy.sum(v*v)

class Plane:
    """A mathematical plane

       The 'origin_info' must either be a point (numpy array of 3 floats) or a numpy
       array of at least 3 non-colinear points.  If it's a single point, then the normal
       vector (numpy array of 3 floats) must be specified.  If it's multiple points then
       the best-fitting plane through those points will be calculated, with the origin
       at the centroid of those points.
    """

    usage_msg = "Plane must be defined either by a single point and a normal, or " \
        "with an array of N points"
    def __init__(self, origin_info, *, normal=None):
        origin_info = numpy.array(origin_info)
        dims = origin_info.shape
        if dims == (1, 3):
            if normal is None:
                raise ValueError(self.usage_msg)
            self._origin = origin_info
            self.normal = normal
        elif len(dims) == 2 and dims[-1] == 3:
            if normal is not None:
                raise ValueError("'normal' must be None for Plane defined by multiple points")
            num_pts = dims[0]
            if num_pts < 3:
                raise ValueError("Must provide at least 3 points to define plane")
            # Implementation of Newell's algorithm
            # See Foley, van Dam, Feiner, and Hughes (pp. 476-477)
            # Implementation copied from Filippo Tampieri from Graphics Gems
            A = B = C = 0.0
            for i in range(num_pts):
                j = i + 1
                if j == num_pts:
                    j = 0
                ux, uy, uz = origin_info[i]
                vx, vy, vz = origin_info[j]
                A += (uy - vy) * (uz - vz)
                B += (uz - vz) * (ux - vx)
                C += (ux - vx) * (uy - vy)
            self._origin = numpy.sum(origin_info) / num_pts
            self.normal = numpy.array(A, B, C)  # uses property, and therefore calls _compute_offset
        else:
            raise ValueError(self.usage_msg)
        self._compute_offset()

    def distance(self, pt):
        return numpy.dot(pt, self._normal) + self._offset

    def equation(self):
        return numpy.array(list(self._normal) + [self._offset])

    def intersection(self, plane):
        """Returns (origin, normal); throws PlaneNoIntersectionError if parallel"""

        v = numpy.cross(self._normal, plane._normal)
        if (sqlength(v) == 0.0)
            raise PlaneNoIntersectionError()

        s1 = numpy.negative(self._offset)
        s2 = numpy.negative(plane._offset)
        n1n2dot = self._normal * plane._normal
        n1normsqr = self._normal * self._normal
        n2normsqr = plane._normal * plane._normal
        divisor = n1n2dot * n1n2dot - n1normsqr * n2normsqr
        a = (s2 * n1n2dot - s1 * n2normsqr) / divisor
        b = (s1 * n1n2dot - s2 * n1normsqr) / divisor
        return a * self._normal + b * plane._normal, v

    def nearest(self, pt):
        return pt - self._normal * self.distance(pt)

    def _get_normal(self):
        return self._normal
    def _set_normal(self, normal):
        self._normal = normalize(normal)
        self._compute_offset()
    normal = property(_get_normal, _set_normal)

    def _get_offset(self):
        return self._offset
    offset = property(_get_offset)

    def _get_origin(self):
        return self._origin
    def _set_origin(self, origin):
        self._origin = origin
        self._compute_offset()
    origin = property(_get_origin, _set_origin)

    def _compute_offset(self):
        self._offset = numpy.negative(numpy.dot(self.origin, self.normal))

    @staticmethod
    def restore_snapshot(session, data):
        return Plane(data['origin'], normal=data['normal'])

    def take_snapshot(self):
        data = { 'origin': self.origin, 'normal': self.normal }
        return data

class PlaneNoIntersectionError(ValueError):
    def __init__(self, msg="Planes do not intersect"):
        ValueError.__init__(self, msg)
