# vim: set expandtab shiftwidth=4 softtabstop=4:

EPSILON = 1e-6

class Ribbon:

    FRONT = 1
    BACK = 2

    def __init__(self, coords, guides):
        if guides is None or len(coords) != len(guides):
            raise ValueError("different number of coordinates and guides")
        # Extend the coordinates at start and end to make sure the
        # ribbon is straight on either end.  Compute the spline
        # coefficients for each axis.  Then throw away the
        # coefficients for the fake ends.
        from numpy import zeros
        c = zeros((len(coords) + 2, 3), float)
        c[0] = coords[0] + (coords[0] - coords[1])
        c[1:-1] = coords
        c[-1] = coords[-1] + (coords[-1] - coords[-2])
        self.coefficients = []
        for i in range(3):
            self._compute_coefficients(c, i)
        # Compute spline normals from guide atom positions.
        self.normals = self._compute_normals_numpy(coords, guides)
        # Initialize segment cache
        self._seg_cache = {}

    def _compute_normals_numpy(self, coords, guides):
        from numpy import zeros, array
        from sys import __stderr__ as stderr
        t = self.get_tangents()
        n = guides - coords
        normals = zeros((len(coords), 3), float)
        for i in range(len(coords)):
            normals[i,:] = get_orthogonal_component(n[i], t[i])
        return normalize_vector_array(normals)

    def _compute_coefficients(self, coords, n):
        # Matrix from http://mathworld.wolfram.com/CubicSpline.html
        # Set b[0] and b[-1] to 1 to match TomG code in VolumePath
        import numpy
        size = len(coords)
        a = numpy.ones((size,), float)
        b = numpy.ones((size,), float) * 4
        b[0] = b[-1] = 2
        #b[0] = b[-1] = 1
        c = numpy.ones((size,), float)
        d = numpy.zeros((size,), float)
        d[0] = coords[1][n] - coords[0][n]
        d[1:-1] = 3 * (coords[2:,n] - coords[:-2,n])
        d[-1] = 3 * (coords[-1][n] - coords[-2][n])
        D = tridiagonal(a, b, c, d)
        from numpy import array
        c_a = coords[:-1,n]
        c_b = D[:-1]
        delta = coords[1:,n] - coords[:-1,n]
        c_c = 3 * delta - 2 * D[:-1] - D[1:]
        c_d = 2 * -delta + D[:-1] + D[1:]
        tcoeffs = array([c_a, c_b, c_c, c_d]).transpose()
        self.coefficients.append(tcoeffs[1:-1])

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

    def get_tangents(self):
        xcv = self.coefficients[0]
        ycv = self.coefficients[1]
        zcv = self.coefficients[2]
        nc = len(xcv)
        from numpy import array
        t = [[xcv[n][1], ycv[n][1], zcv[n][1]] for n in range(nc)]
        xc = xcv[-1]
        yc = ycv[-1]
        zc = zcv[-1]
        t.append((xc[1] + 2 * xc[2] + 3 * xc[3],
                   yc[1] + 2 * yc[2] + 3 * yc[3],
                   zc[1] + 2 * zc[2] + 3 * zc[3]))
        return normalize_vector_array(array(t))

    def normal(self, n):
        return self.normals[n]

    def segment(self, seg, side, divisions, last=False):
        try:
            coords, tangents, normals = self._seg_cache[seg]
        except KeyError:
            coords, tangents = self._segment_path(seg, divisions)
            tangents = normalize_vector_array(tangents)
            ns = self.normals[seg]
            ne = self.normals[seg + 1]
            normals, flipped = constrained_normals(tangents, ns, ne)
            if flipped:
                self.normals[seg + 1] = -ne
            self._seg_cache[seg] = (coords, tangents, normals)
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

    def _segment_path(self, seg, divisions):
        # Compute coordinates by multiplying spline parameter vector
        # (1, t, t**2, t**3) by the spline coefficients, and
        # compute tangents by multiplying spline parameter vector
        # (0, 1, 2*t, 3*t**2) by the same spline coefficients
        from numpy import array, zeros, ones, linspace, dot
        spline = array([self.coefficients[0][seg],
                        self.coefficients[1][seg],
                        self.coefficients[2][seg]]).transpose()
        nc = divisions + 1
        t = linspace(0.0, 1.0, nc)
        t2 = t * t
        coeff_shape = (nc, 4)
        ct = ones(coeff_shape, float)    # coords coefficients
        ct[:,1] = t
        ct[:,2] = t2
        ct[:,3] = t * t2
        coords = dot(ct, spline)
        tt = zeros(coeff_shape, float)   # tangent coefficients
        tt[:,1] = 1.0
        tt[:,2] = 2.0 * t
        tt[:,3] = 3 * t2
        tangents = dot(tt, spline)
        return coords, tangents


def normalize(v):
    # normalize a single vector
    from numpy.linalg import norm
    d = norm(v)
    if d < EPSILON:
        return v
    return v / d


def normalize_vector_array(a):
    from numpy.linalg import norm
    import numpy
    d = norm(a, axis=1)
    d[d < EPSILON] = 1
    n = a / d[:, numpy.newaxis]
    return n


def normalize_vector_array_inplace(a):
    from numpy.linalg import norm
    import numpy
    d = norm(a, axis=1)
    d[d < EPSILON] = 1
    a /= d[:, numpy.newaxis]
    return a


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


def constrained_normals(tangents, n_start, n_end):
    from .molc import c_function
    import ctypes
    f = c_function("constrained_normals",
                   args=(ctypes.py_object, ctypes.py_object, ctypes.py_object),
                   ret = ctypes.py_object)
    return f(tangents, n_start, n_end)




# Code for debugging moving code from Python to C++
# 
# DebugCVersion = False
# 
def _debug_compare(label, test, ref, verbose=False):
    from sys import __stderr__ as stderr
    if isinstance(ref, list):
        from numpy import array
        ref = array(ref)
    from numpy import allclose
    try:
        if not allclose(test, ref, atol=1e-4):
            raise ValueError("not same")
    except ValueError:
        print(label, "--- not same!", test.shape, ref.shape, file=stderr)
        print(test, file=stderr)
        print(ref, file=stderr)
        print(test - ref, file=stderr)
    else:
        if verbose:
            print(label, "-- okay", file=stderr)
