# vi: set expandtab shiftwidth=4 softtabstop=4:

EPSILON = 1e-6

class Ribbon:

    FRONT = 1
    BACK = 2

    def __init__(self, coords, guides):
        if guides is None or len(coords) != len(guides):
            raise ValueError("different number of coordinates and guides")
        self.coefficients = []
        for i in range(3):
            self._compute_coefficients(coords, i)
        self.normals = []
        for i in range(len(coords) - 1):
            n = guides[i] - coords[i]
            t = self.tangent(i)
            normal = normalize(get_orthogonal_component(n, t))
            self.normals.append(normal)
        # Have to do last normal differently because
        # the tangent is computed differently
        n = guides[-1] - coords[-1]
        t = self.last_tangent()
        normal = normalize(get_orthogonal_component(n, t))
        self.normals.append(normal)

    def _compute_coefficients(self, coords, n):
        # Matrix from http://mathworld.wolfram.com/CubicSpline.html
        # Set b[0] and b[-1] to 1 to match TomG code in VolumePath
        import numpy
        size = len(coords)
        a = numpy.ones((size,), float)
        b = numpy.ones((size,), float) * 4
        #b[0] = b[-1] = 2
        b[0] = b[-1] = 1
        c = numpy.ones((size,), float)
        d = numpy.zeros((size,), float)
        d[0] = coords[1][n] - coords[0][n]
        for i in range(1, size - 1):
            d[i] = 3 * (coords[i + 1][n] - coords[i - 1][n])
        d[-1] = 3 * (coords[-1][n] - coords[-2][n])
        D = tridiagonal(a, b, c, d)
        coeffs = []
        for i in range(0, size - 1):
            c_a = coords[i][n]
            c_b = D[i]
            delta = coords[i + 1][n] - coords[i][n]
            c_c = 3 * delta - 2 * D[i] - D[i + 1]
            c_d = 2 * -delta + D[i] + D[i + 1]
            coeffs.append((c_a, c_b, c_c, c_d))
        self.coefficients.append(coeffs)

    @property
    def num_segments(self):
        return len(self.coefficients[0])

    @property
    def has_normals(self):
        return hasattr(self, "normals")

    def coordinate(self, n):
        from numpy import array
        xc = self.coefficients[0][n]
        yc = self.coefficients[1][n]
        zc = self.coefficients[2][n]
        return array((xc[0], yc[0], zc[0]))

    @property
    def coordinates(self):
        from numpy import array
        xcv = self.coefficients[0]
        ycv = self.coefficients[1]
        zcv = self.coefficients[2]
        return array([[xcv[n], ycv[n], zcv[n]]
                      for n in range(self.num_segments)])

    def tangent(self, n):
        from numpy import array
        xc = self.coefficients[0][n]
        yc = self.coefficients[1][n]
        zc = self.coefficients[2][n]
        return normalize(array((xc[1], yc[1], zc[1])))

    def last_tangent(self):
        from numpy import array
        xc = self.coefficients[0][-1]
        yc = self.coefficients[1][-1]
        zc = self.coefficients[2][-1]
        return normalize(array((xc[1] + 2 * xc[2] + 3 * xc[3],
                                yc[1] + 2 * yc[2] + 3 * yc[3],
                                zc[1] + 2 * zc[2] + 3 * zc[3])))

    @property
    def tangents(self):
        from numpy import array
        xcv = self.coefficients[0]
        ycv = self.coefficients[1]
        zcv = self.coefficients[2]
        return array([normalize(array([xcv[n], ycv[n], zcv[n]]))
                      for n in range(self.num_segments)])

    def normal(self, n):
        return self.normals[n]

    def segment(self, seg, side, divisions, last=False):
        xc = self.coefficients[0][seg]
        yc = self.coefficients[1][seg]
        zc = self.coefficients[2][seg]
        coords = []
        tangents = []
        from numpy import array
        coords = [array([xc[0], yc[0], zc[0]])]
        tangents = [normalize(array([xc[1], yc[1], zc[1]]))]
        d = float(divisions)
        for i in range(divisions):
            t = (i + 1) / d
            t2 = t * t
            t3 = t2 * t
            x = xc[0] + xc[1] * t + xc[2] * t2 + xc[3] * t3
            y = yc[0] + yc[1] * t + yc[2] * t2 + yc[3] * t3
            z = zc[0] + zc[1] * t + zc[2] * t2 + zc[3] * t3
            coords.append(array([x, y, z]))
            x = xc[1] + 2 * xc[2] * t + 3 * xc[3] * t2
            y = yc[1] + 2 * yc[2] * t + 3 * yc[3] * t2
            z = zc[1] + 2 * zc[2] * t + 3 * zc[3] * t2
            tangents.append(normalize(array([x, y, z])))
        coords = array(coords)
        tangents = array(tangents)
        ns = self.normals[seg]
        ne = self.normals[seg + 1]
        normals, flipped = constrained_normals(tangents, ns, ne)
        if flipped:
            self.normals[seg + 1] = -ne
        # divisions = number of segments = number of vertices + 1
        if side is self.FRONT:
            start = 0
            end = (divisions // 2) + 1
        else:
            start = (divisions + 1) // 2
            if last:
                end = divisions + 1
            else:
                end = divisions
        return coords[start:end], tangents[start:end], normals[start:end]


from collections import namedtuple
ExtrudeValue = namedtuple("ExtrudeValue", ["vertices", "normals", "triangles", "colors",
                                           "front_band", "back_band"])


class XSection:

    def __init__(self, coords, coords2=None, normals=None, normals2=None, faceted=False):
        # XSection coordinates are 2D and counterclockwise
        import numpy
        self.xs_coords = numpy.array(coords)
        if coords2 is not None:
            self.xs_coords2 = numpy.array(coords2)
        else:
            self.xs_coords2 = None
        if normals is None:
            self._generate_normals(faceted)
        elif normals2 is None:
            self.xs_normals = numpy.array(normals)
            self.normalize_normals(self.xs_normals)
            self.extrude = self._extrude_smooth
            self.blend = self._blend_smooth
        else:
            self.xs_normals = numpy.array(normals)
            self.normalize_normals(self.xs_normals)
            self.xs_normals2 = numpy.array(normals2)
            self.normalize_normals(self.xs_normals2)
            self.extrude = self._extrude_faceted
            self.blend = self._blend_faceted
        self.tessellation = tessellate(self.xs_coords)

    def _generate_normals(self, faceted):
        import numpy
        if not faceted:
            num_coords = len(self.xs_coords)
            self.xs_normals = numpy.zeros((num_coords, 2), float)
            for i in range(num_coords):
                ci = self.xs_coords[i]
                j = (i + 1) % num_coords
                cj = self.xs_coords[j]
                k = (i + 2) % num_coords
                ck = self.xs_coords[k]
                n = ck - ci
                if is_concave(ci, cj, ck):
                    x = -n[1]
                    y = n[0]
                else:
                    x = n[1]
                    y = -n[0]
                self.xs_normals[j][:] = (x, y)
            self.normalize_normals(self.xs_normals)
            self.extrude = self._extrude_smooth
            self.blend = self._blend_smooth
        else:
            num_coords = len(self.xs_coords)
            self.xs_normals = numpy.zeros((num_coords, 2), float)
            self.xs_normals2 = numpy.zeros((num_coords, 2), float)
            from math import sqrt
            for i in range(num_coords):
                j = (i + 1) % num_coords
                dx = self.xs_coords[j][0] - self.xs_coords[i][0]
                dy = self.xs_coords[j][1] - self.xs_coords[i][1]
                d = sqrt(dx * dx + dy * dy)
                n = (dy / d, -dx / d)
                self.xs_normals[i] = n
                self.xs_normals2[j] = n
            self.normalize_normals(self.xs_normals)
            self.normalize_normals(self.xs_normals2)
            self.extrude = self._extrude_faceted
            self.blend = self._blend_faceted

    def normalize_normals(self, v):
        # normalize an array of vectors
        import numpy
        lens = numpy.sqrt(v[:,0]**2 + v[:,1]**2)
        v[:,0] /= lens
        v[:,1] /= lens

    def _extrude_smooth(self, centers, tangents, normals, color, cap_front, cap_back, offset):
        from numpy import cross, concatenate, array, zeros
        binormals = cross(tangents, normals)
        # Generate spline coordinates
        num_splines = len(self.xs_coords)
        num_pts_per_spline = len(centers)
        num_vertices = num_splines * num_pts_per_spline
        if cap_front:
            num_vertices += num_splines
        if cap_back:
            num_vertices += num_splines
        ca = zeros((num_vertices, 4), float)
        ca[:] = color
        va = zeros((num_vertices, 3), float)
        na = zeros((num_vertices, 3), float)
        vindex = 0
        for i in range(num_splines):
            # xc, xn = extrusion coordinates and normals
            if self.xs_coords2 is None:
                n, b = self.xs_coords[i]
            else:
                n1, b1 = self.xs_coords[i]
                n2, b2 = self.xs_coords2[i]
                steps = len(centers)
                from numpy import linspace
                n = linspace(n1, n2, steps)
                b = linspace(b1, b2, steps)
            xc = centers + normals * n + binormals * b
            n, b = self.xs_normals[i]
            xn = normals * n + binormals * b
            va[vindex:vindex + num_pts_per_spline] = xc
            na[vindex:vindex + num_pts_per_spline] = xn
            # XXX: These normals are not quite right for an arrow because
            # they should be slanted proportionally to the arrow angle.
            # However, to compute them correctly , we would need to compute
            # the path length and width rates in order to get the correct
            # proportion and the difference visually is not great.
            # So we ignore the problem for now.
            vindex += num_pts_per_spline
        # Generate triangle list for sides
        num_triangles = num_splines * (num_pts_per_spline - 1) * 2
        if cap_front:
            num_triangles += len(self.tessellation)
        if cap_back:
            num_triangles += len(self.tessellation)
        ta = zeros((num_triangles, 3), int)
        tindex = 0
        front_band = []
        back_band = []
        for s in range(num_splines):
            i_start = s * num_pts_per_spline + offset
            front_band.append(i_start)
            back_band.append(i_start + num_pts_per_spline - 1)
            j = (s + 1) % num_splines
            j_start = j * num_pts_per_spline + offset
            for k in range(num_pts_per_spline - 1):
                ta[tindex] = (i_start + k + 1, i_start + k, j_start + k)
                ta[tindex + 1] = (i_start + k + 1, j_start + k, j_start + k + 1)
                tindex += 2
                # Skip second triangle for "reptile" mode
        # Generate caps
        offset += num_splines * num_pts_per_spline
        if cap_front:
            for i in range(num_splines):
                va[vindex + i] = va[i * num_pts_per_spline]
            na[vindex:vindex + num_splines] = -tangents[0]
            for i, j, k in self.tessellation:
                ta[tindex] = (k + offset, j + offset, i + offset)
                tindex += 1
            offset += num_splines
            vindex += num_splines
        if cap_back:
            for i in range(num_splines):
                va[vindex + i] = va[i * num_pts_per_spline + num_pts_per_spline - 1]
            na[vindex:vindex + num_splines] = tangents[-1]
            for i, j, k in self.tessellation:
                ta[tindex] = (i + offset, j + offset, k + offset)
                tindex += 1
        return ExtrudeValue(va, na, ta, ca, front_band, back_band)

    def _blend_smooth(self, back_band, front_band):
        size = len(back_band)
        if len(front_band) != size:
            raise ValueError("blending non-identical cross sections")
        from numpy import zeros
        ta = zeros((size * 2, 3), int)
        for i in range(size):
            j = (i + 1) % size
            ta[i * 2] = (back_band[i], back_band[j], front_band[i])
            ta[i * 2 + 1] = (front_band[i], back_band[j], front_band[j])
        return ta

    def _extrude_faceted(self, centers, tangents, normals, color, cap_front, cap_back, offset):
        from numpy import cross, concatenate, array, zeros
        sc = array([color] * len(centers))
        binormals = cross(tangents, normals)
        # Generate spline coordinates
        num_splines = len(self.xs_coords)
        num_pts_per_spline = len(centers)
        num_vertices = num_splines * num_pts_per_spline * 2
        if cap_front:
            num_vertices += num_splines
        if cap_back:
            num_vertices += num_splines
        ca = zeros((num_vertices, 4), float)
        ca[:] = color
        va = zeros((num_vertices, 3), float)
        na = zeros((num_vertices, 3), float)
        vindex = 0
        for i in range(num_splines):
            # xc, xn = extrusion coordinates and normals
            if self.xs_coords2 is None:
                n, b = self.xs_coords[i]
            else:
                n1, b1 = self.xs_coords[i]
                n2, b2 = self.xs_coords2[i]
                steps = len(centers)
                front = steps // 2
                from numpy import ones, linspace
                n = ones(steps, float) * n2
                b = ones(steps, float) * b2
                n[:front] = linspace(n1, n2, front)
                b[:front] = linspace(b1, b2, front)
                n.shape = (steps, 1)
                b.shape = (steps, 1)
            xc = centers + normals * n + binormals * b
            # append vertex twice for different normals
            n, b = self.xs_normals[i]
            xn = normals * n + binormals * b
            va[vindex:vindex + num_pts_per_spline] = xc
            na[vindex:vindex + num_pts_per_spline] = xn
            vindex += num_pts_per_spline
            n, b = self.xs_normals2[i]
            xn = normals * n + binormals * b
            va[vindex:vindex + num_pts_per_spline] = xc
            na[vindex:vindex + num_pts_per_spline] = xn
            vindex += num_pts_per_spline
        # Generate triangle list
        num_triangles = num_splines * (num_pts_per_spline - 1) * 2
        if cap_front:
            num_triangles += len(self.tessellation)
        if cap_back:
            num_triangles += len(self.tessellation)
        ta = zeros((num_triangles, 3), int)
        tindex = 0
        front_band = []
        back_band = []
        for i in range(num_splines):
            i_start = (i * 2) * num_pts_per_spline + offset
            front_band.append(i_start)
            front_band.append(i_start + num_pts_per_spline)
            back_band.append(i_start + num_pts_per_spline - 1)
            back_band.append(i_start + 2 * num_pts_per_spline - 1)
            j = (i + 1) % num_splines
            j_start = (j * 2 + 1) * num_pts_per_spline + offset
            for k in range(num_pts_per_spline - 1):
                ta[tindex] = (i_start + k + 1, i_start + k, j_start + k)
                # Comment out next statement for "reptile" mode
                ta[tindex + 1] = (i_start + k + 1, j_start + k, j_start + k + 1)
                tindex += 2
        # Generate caps
        offset += num_splines * num_pts_per_spline * 2
        if cap_front:
            for i in range(num_splines):
                va[vindex + i] = va[i * 2 * num_pts_per_spline]
            na[vindex:vindex + num_splines] = -tangents[0]
            for i, j, k in self.tessellation:
                ta[tindex] = (k + offset, j + offset, i + offset)
                tindex += 1
            offset += num_splines
            vindex += num_splines
        if cap_back:
            for i in range(num_splines):
                va[vindex + i] = va[i * 2 * num_pts_per_spline + num_pts_per_spline - 1]
            na[vindex:vindex + num_splines] = tangents[-1]
            for i, j, k in self.tessellation:
                ta[tindex] = (i + offset, j + offset, k + offset)
                tindex += 1
        return ExtrudeValue(va, na, ta, ca, front_band, back_band)

    def _blend_faceted(self, back_band, front_band):
        size = len(back_band)
        if len(front_band) != size:
            raise ValueError("blending non-identical cross sections")
        num_vertices = size // 2
        from numpy import zeros
        ta = zeros((num_vertices * 2, 3), int)
        for n in range(0, num_vertices):
            i = n * 2
            j = (i + 3) % size
            ta[n * 2] = (back_band[i], back_band[j], front_band[i])
            ta[n * 2 + 1] = (front_band[i], back_band[j], front_band[j])
        return ta


def normalize(v):
    # normalize a single vector
    from numpy.linalg import norm
    d = norm(v)
    if d < EPSILON:
        return v
    return v / d


def tridiagonal(a, b, c, d):
    '''
    TDMA solver, a b c d can be NumPy array type or Python list type.
    refer to http://en.wikipedia.org/wiki/Tridiagonal_matrix_algorithm
    Hacked source from
    http://ofan666.blogspot.com/2012/02/tridiagonal-matrix-algorithm-solver-in.html
    '''
    nf = len(a)     # number of equations
    for i in range(1, nf):
        mc = a[i] / b[i - 1]
        b[i] = b[i] - mc * c[i - 1] 
        d[i] = d[i] - mc * d[i - 1]
    xc = a
    xc[-1] = d[-1] / b[-1]
    for i in range(nf - 2, -1, -1):
        xc[i] = (d[i] - c[i] * xc[i + 1]) / b[i]
    return xc


def get_orthogonal_component(v, ref):
    from numpy import inner
    from numpy.linalg import norm
    d = inner(v, ref)
    ref_len = norm(ref)
    return v + ref * (-d / ref_len)


def _rotate_around(n, c, s, v):
    c1 = 1 - c
    m00 = c + n[0] * n[0] * c1
    m01 = n[0] * n[1] * c1 - s * n[2]
    m02 = n[2] * n[0] * c1 + s * n[1]
    m10 = n[0] * n[1] * c1 + s * n[2]
    m11 = c + n[1] * n[1] * c1
    m12 = n[2] * n[1] * c1 - s * n[0]
    m20 = n[0] * n[2] * c1 - s * n[1]
    m21 = n[1] * n[2] * c1 + s * n[0]
    m22 = c + n[2] * n[2] * c1
    from numpy import array
    return array([m00 * v[0] + m01 * v[1] + m02 * v[2],
                  m10 * v[0] + m11 * v[1] + m12 * v[2],
                  m20 * v[0] + m21 * v[1] + m22 * v[2]])


def parallel_transport_normals(tangents, n0):
    from numpy import cross, inner, array
    from numpy.linalg import norm
    from math import sqrt
    normals = [ n0 ]
    n = n0
    for i in range(len(tangents) - 1):
        b = cross(tangents[i], tangents[i + 1])
        if norm(b) < EPSILON:
            normals.append(n)
        else:
            b_hat = normalize(b)
            cos_theta = inner(tangents[i], tangents[i + 1])
            sin_theta = sqrt(1 - cos_theta * cos_theta)
            n = _rotate_around(b_hat, cos_theta, sin_theta, n)
            normals.append(n)
    return array(normals)


def constrained_normals(tangents, n_start, n_end):
    from numpy import cross, inner
    from math import acos, pi, sin, cos
    normals = parallel_transport_normals(tangents, n_start)
    n = normals[-1]
    other_end = n_end
    twist = acos(inner(n, n_end))
    flipped = False
    if twist > pi / 2:
        other_end = -n_end
        twist = acos(inner(n, other_end))
        flipped = True
    delta = twist / (len(normals) - 1)
    if inner(cross(n, other_end), tangents[-1]) < 0:
        delta = -delta
    for i in range(1, len(normals)):
        c = cos(i * delta)
        s = sin(i * delta)
        normals[i] = _rotate_around(tangents[i], c, s, normals[i])
    return normals, flipped

def tessellate(coords):
    tess = []
    if False:
        for i in range(1, len(coords) - 1):
            tess.append((0, i, i + 1))
    # "indices" is the array of vertex indices remaining in the tessellation
    # the vertices are ordered counterclockwise
    indices = list(range(len(coords)))
    while len(indices) > 3:
        num_indices = len(indices)
        for i in range(num_indices):
            ni = indices[i]
            nj = indices[(i + 1) % num_indices]
            nk = indices[(i + 2) % num_indices]
            # Consider whether we can form triangle from the
            # consecutive sequence of 3 vertices
            ci = coords[ni]
            cj = coords[nj]
            ck = coords[nk]
            # Compute cross product to check whether vertices
            # result in a triangle on the inside of the polygon
            if is_concave(ci, cj, ck):
                continue
            # Check if any segments intersect ci-ck
            for m in range(num_indices):
                # Skip edges from candidate triangle
                if m >= i and m <= i + 2:
                    continue
                cm = coords[indices[m]]
                cn = coords[indices[(m + 1) % num_indices]]
                if intersects(ci, ck, cm, cn):
                    break
            else:
                # No intersections, triangle should be okay
                tess.append((ni, nj, nk))
                # i and k are still part of polygon to be tessellated, but j is done
                del indices[(i + 1) % num_indices]
                break
        else:
            raise RuntimeError("cannot tessellate cross section")
    tess.append((indices[0], indices[1], indices[2]))
    return tess

def is_concave(c0, c1, c2):
    u01 = c0 - c1
    u21 = c2 - c1
    return (u01[0] * u21[1] - u01[1] * u21[0]) >= 0

def intersects(p0, p1, q0, q1):
    # From http://geomalgorithms.com/a05-_intersect-1.html
    from numpy import array
    u = p1 - p0
    v = q1 - q0
    w = p0 - q0
    #u_p = array([-u[1], u[0])    # u perpendicular
    #v_p = array([-v[1], v[0])    # v perpendicular
    #s_i = -(v_p[0] * w[0] + v_p[1] * w[1]) / (v_p[0] * u[0] + v_p[1] * u[1])
    #t_i = (u_p[0] * w[0] + u_p[1] * w[1]) / (u_p[0] * v[0] + u_p[1] * v[1])
    det = (-u[1] * v[0] + u[0] * v[1])
    if det < EPSILON:
        return False
    s_i = (v[0] * w[1] - v[1] * w[0]) / det
    t_i = (u[0] * w[0] - u[1] * w[0]) / det
    return s_i > 0 and s_i < 1 and t_i > 0 and t_i < 1
