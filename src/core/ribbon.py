# vi: set expandtab shiftwidth=4 softtabstop=4:

EPSILON = 1e-6

class Ribbon:

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
        b[0] = b[-1] = 2
        c = numpy.ones((size,), float)
        d = numpy.zeros((size,), float)
        d[0] = coords[1][n] - coords[0][n]
        for i in range(1, size - 1):
            d[i] = 3 * (coords[i + 1][n] - coords[i - 1][n])
        d[-1] = coords[-1][n] - coords[-2][n]
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

    def segment(self, seg, divisions):
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
        ns = self.normal(seg)
        ne = self.normal(seg + 1)
        normals = constrained_normals(tangents, ns, ne)
        return coords, tangents, normals


class XSection:

    FRONT = 0x1
    BACK = 0x2
    BOTH = FRONT | BACK

    def __init__(self, coords, normals=None, normals2=None, faceted=False):
        import numpy
        self.xs_coords = numpy.array(coords)
        if normals is None:
            self._generate_normals(faceted)
        elif normals2 is None:
            self.xs_normals = numpy.array(normals)
            self.normalize_normals(self.xs_normals)
            self.extrude = self._extrude_smooth
        else:
            self.xs_normals = numpy.array(normals)
            self.normalize_normals(self.xs_normals)
            self.xs_normals2 = numpy.array(normals2)
            self.normalize_normals(self.xs_normals2)
            self.extrude = self._extrude_faceted

    def _generate_normals(self, faceted):
        import numpy
        if not faceted:
            self.xs_normals = numpy.array(self.xs_coords)
            self.normalize_normals(self.xs_normals)
            self.extrude = self._extrude_smooth
        else:
            num_coords = len(self.xs_coords)
            xs_normals = [None] * num_coords
            xs_normals2 = [None] * num_coords
            from math import sqrt
            for i in range(num_coords):
                j = (i + 1) % num_coords
                dx = self.xs_coords[j][0] - self.xs_coords[i][0]
                dy = self.xs_coords[j][1] - self.xs_coords[i][1]
                d = sqrt(dx * dx + dy * dy)
                n = (dy / d, -dx / d)
                xs_normals[i] = n
                xs_normals2[j] = n
            self.xs_normals = numpy.array(xs_normals)
            self.normalize_normals(self.xs_normals)
            self.xs_normals2 = numpy.array(xs_normals2)
            self.normalize_normals(self.xs_normals2)
            self.extrude = self._extrude_faceted

    def normalize_normals(self, v):
        # normalize an array of vectors
        import numpy
        lens = numpy.sqrt(v[:,0]**2 + v[:,1]**2)
        v[:,0] /= lens
        v[:,1] /= lens

    def _extrude_smooth(self, centers, tangents, normals, show, cap, offset):
        from numpy import cross, concatenate, array
        import numpy
        if show == self.FRONT:
            end = len(centers) // 2 + 1
            centers = centers[:end]
            tangents = tangents[:end]
            normals = normals[:end]
        elif show == self.BACK:
            start = len(centers) // 2 + 1
            centers = centers[start:]
            tangents = tangents[start:]
            normals = normals[start:]
        binormals = cross(tangents, normals)
        # Generate spline coordinates
        num_splines = len(self.xs_coords)
        vertex_list = []
        normal_list = []
        for i in range(num_splines):
            # xc, xn = extrusion coordinates and normals
            n, b = self.xs_coords[i]
            xc = centers + normals * n + binormals * b
            vertex_list.append(xc)
            n, b = self.xs_normals[i]
            xn = normals * n + binormals * b
            normal_list.append(xn)
        va = concatenate(vertex_list)
        na = concatenate(normal_list)
        # Generate triangle list
        num_pts_per_spline = len(centers)
        triangle_list = []
        for s in range(num_splines):
            i_start = s * num_pts_per_spline + offset
            j = (s + 1) % num_splines
            j_start = j * num_pts_per_spline + offset
            for k in range(num_pts_per_spline - 1):
                triangle_list.append((i_start + k + 1, i_start + k,
                                      j_start + k))
                # Comment out next statement for "reptile" mode
                triangle_list.append((i_start + k + 1, j_start + k,
                                      j_start + k + 1))
        ta = array(triangle_list)
        return va, na, ta

    def _extrude_faceted(self, centers, tangents, normals, show, cap, offset):
        from numpy import cross, concatenate, array
        import numpy
        if show == self.FRONT:
            end = len(centers) // 2 + 1
            centers = centers[:end]
            tangents = tangents[:end]
            normals = normals[:end]
        elif show == self.BACK:
            start = len(centers) // 2 + 1
            centers = centers[start:]
            tangents = tangents[start:]
            normals = normals[start:]
        binormals = cross(tangents, normals)
        # Generate spline coordinates
        num_splines = len(self.xs_coords)
        vertex_list = []
        normal_list = []
        for i in range(num_splines):
            # xc, xn = extrusion coordinates and normals
            n, b = self.xs_coords[i]
            xc = centers + normals * n + binormals * b
            # append vertex twice for different normals
            vertex_list.append(xc)
            vertex_list.append(xc)
            n, b = self.xs_normals[i]
            xn = normals * n + binormals * b
            normal_list.append(xn)
            n, b = self.xs_normals2[i]
            xn = normals * n + binormals * b
            normal_list.append(xn)
        va = concatenate(vertex_list)
        na = concatenate(normal_list)
        # Generate triangle list
        num_pts_per_spline = len(centers)
        triangle_list = []
        for i in range(num_splines):
            i_start = (i * 2) * num_pts_per_spline + offset
            j = (i + 1) % num_splines
            j_start = (j * 2 + 1) * num_pts_per_spline + offset
            for k in range(num_pts_per_spline - 1):
                triangle_list.append((i_start + k + 1, i_start + k,
                                      j_start + k))
                # Comment out next statement for "reptile" mode
                triangle_list.append((i_start + k + 1, j_start + k,
                                      j_start + k + 1))
        ta = array(triangle_list)
        return va, na, ta


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
    if twist > pi / 2:
        other_end = -n_end
        twist = acos(inner(n, other_end))
    delta = twist / (len(normals) - 1)
    if inner(cross(n, other_end), tangents[-1]) < 0:
        delta = -delta
    for i in range(1, len(normals)):
        c = cos(i * delta)
        s = sin(i * delta)
        normals[i] = _rotate_around(tangents[i], c, s, normals[i])
    return normals
