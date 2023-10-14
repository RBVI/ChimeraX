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

import numpy
from numpy.linalg import norm, eig, svd, eigh
normalize = lambda v: v/norm(v)
sqlength = lambda v: numpy.sum(v*v)

from chimerax.core.state import State
class Plane(State):
    """A mathematical plane

       The 'origin_info' must either be a point (numpy array of 3 floats) or an array/list
       of at least 3 non-colinear points.  If it's a single point, then the normal vector
       (numpy array of 3 floats) must be specified.  If it's multiple points then
       the best-fitting plane through those points will be calculated, with the origin
       at the centroid of those points.
    """

    usage_msg = "Plane must be defined either by a single point and a normal, or " \
        "with an array of N points"
    def __init__(self, origin_info, *, normal=None):
        origin_info = numpy.array(origin_info)
        dims = origin_info.shape
        if dims == (3,):
            if normal is None:
                raise ValueError(self.usage_msg)
            self._origin = origin_info
            # sets 'normal' property, and therefore calls _compute_offset
            self.normal = normal
        elif len(dims) == 2 and dims[-1] == 3:
            if normal is not None:
                raise ValueError("'normal' must be None for Plane defined by multiple points")
            num_pts = dims[0]
            if num_pts < 3:
                raise ValueError("Must provide at least 3 points to define plane")
            xyzs = origin_info
            centroid = xyzs.mean(0)
            centered = xyzs - centroid
            ignore, vals, vecs = svd(centered, full_matrices=False)
            self._origin = centroid
            # sets 'normal' property, and therefore calls _compute_offset
            self.normal = vecs[numpy.argmin(vals)]
        else:
            raise ValueError(self.usage_msg)

    def distance(self, pt):
        return numpy.dot(pt, self._normal) + self._offset

    def equation(self):
        return numpy.array(list(self._normal) + [self._offset])

    def intersection(self, plane):
        """Returns a line in the form (origin, vector); throws PlaneNoIntersectionError if parallel"""

        v = numpy.cross(self._normal, plane._normal)
        if sqlength(v) == 0.0:
            raise PlaneNoIntersectionError()

        s1 = numpy.negative(self._offset)
        s2 = numpy.negative(plane._offset)
        n1n2dot = numpy.dot(self._normal, plane._normal)
        n1normsqr = numpy.dot(self._normal, self._normal)
        n2normsqr = numpy.dot(plane._normal, plane._normal)
        divisor = n1n2dot * n1n2dot - n1normsqr * n2normsqr
        a = (s2 * n1n2dot - s1 * n2normsqr) / divisor
        b = (s1 * n1n2dot - s2 * n1normsqr) / divisor
        return a * self._normal + b * plane._normal, v

    plane_intersection = intersection

    def line_intersection(self, origin, direction, *, epsilon=1e-6):
        # Cribbed from https://rosettacode.org/wiki/Find_the_intersection_of_a_line_with_a_plane
        n_dot_d = self.normal.dot(direction)
        if abs(n_dot_d) < epsilon:
            raise PlaneNoIntersectionError("Line does not intersect plane or lies in plane")

        w = origin - self.origin
        si = -self.normal.dot(w) / n_dot_d
        return w + si * direction + self.origin

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
        self._offset = -numpy.dot(self.origin, self.normal)

    @staticmethod
    def restore_snapshot(session, data):
        return Plane(data['origin'], normal=data['normal'])

    def take_snapshot(self, session, flags):
        data = { 'origin': self.origin, 'normal': self.normal }
        return data

class PlaneNoIntersectionError(ValueError):
    def __init__(self, msg="Planes do not intersect"):
        ValueError.__init__(self, msg)
