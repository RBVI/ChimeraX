# vim: set expandtab shiftwidth=4 softtabstop=4:

EPSILON = 1e-6

# Must match values in molc.cpp
FLIP_MINIMIZE = 0
FLIP_PREVENT = 1
FLIP_FORCE = 2

class Ribbon:

    FRONT = 1
    BACK = 2

    def __init__(self, coords, guides, orient):
        # Extend the coordinates at start and end to make sure the
        # ribbon is straight on either end.  Compute the spline
        # coefficients for each axis.  Then throw away the
        # coefficients for the fake ends.
        from .structure import Structure
        from numpy import empty, zeros
        c = empty((len(coords) + 2, 3), float)
        c[0] = coords[0] + (coords[0] - coords[1])
        c[1:-1] = coords
        c[-1] = coords[-1] + (coords[-1] - coords[-2])
        self.coefficients = []
        for i in range(3):
            self._compute_coefficients(c, i)
        self.flipped = zeros(len(coords), bool)
        if orient == Structure.RIBBON_ORIENT_ATOMS:
            self.normals = self._compute_normals_from_control_points(coords)
            self.ignore_flip_mode = True
        elif orient == Structure.RIBBON_ORIENT_CURVATURE:
            self.normals = self._compute_normals_from_curvature(coords)
            self.ignore_flip_mode = True
        else:
            # RIBBON_ORIENT_GUIDES and default case
            if guides is None or len(coords) != len(guides):
                self.normals = self._compute_normals_from_control_points(coords)
                self.ignore_flip_mode = True
            else:
                self.normals = self._compute_normals_from_guides(coords, guides)
                self.ignore_flip_mode = False
        # Initialize segment cache
        self._seg_cache = {}

    def _compute_normals_from_guides(self, coords, guides):
        from numpy import zeros, array
        from sys import __stderr__ as stderr
        t = self.get_tangents()
        n = guides - coords
        normals = zeros((len(coords), 3), float)
        for i in range(len(coords)):
            normals[i,:] = get_orthogonal_component(n[i], t[i])
        return normalize_vector_array(normals)

    def _compute_normals_from_curvature(self, coords):
        from numpy import empty, array, cross, inner
        from numpy.linalg import norm
        tangents = self.get_tangents()
        raw_normals = empty(coords.shape, float)
        xcv = self.coefficients[0]
        ycv = self.coefficients[1]
        zcv = self.coefficients[2]
        # Curvature for the N-1 points are at the start of a segment
        # but the last one is at the end of the last segment and has
        # to be treated specially
        for seg in range(len(raw_normals) - 1):
            curvature = array((xcv[seg][2], ycv[seg][2], zcv[seg][2]))
            raw_normals[seg] = cross(curvature, tangents[seg])
        xc = xcv[-1]
        yc = ycv[-1]
        zc = zcv[-1]
        curvature = array((2 * xc[2] + 6 * xc[3],
                           2 * yc[2] + 6 * yc[3],
                           2 * zc[2] + 6 * zc[3]))
        raw_normals[-1] = cross(curvature, tangents[-1])
        #
        # The normals are NOT normalized.  We check for any near-zero
        # normals and replace them with vectors computed from well-defined
        # normals of adjacent control points
        normals = empty(raw_normals.shape, float)
        lengths = norm(raw_normals, axis=1)
        last_valid = None
        last_normal = None
        for i in range(len(lengths)):
            if lengths[i] > EPSILON:
                n = raw_normals[i] / lengths[i]
                normals[i] = n
                if last_valid is None:
                    # At start of spline, so we propagate this normal
                    # backwards (if needed)
                    for j in range(i):
                        normals[j] = n
                else:
                    # There was a previous normal defined so we interpolate
                    # for intermediates (if needed)
                    dv = n - last_normal
                    ncp = i - last_valid
                    for j in range(1, ncp):
                        f = j / ncp
                        rn = dv * f + last_normal
                        d = norm(rn)
                        int_n = rn / d
                        normals[last_valid+j] = int_n
                last_valid = i
                last_normal = n
        # Check whether there are control points at the end that have
        # no normals.  It could actually be all control points if the
        # spline is really a straight line.
        if last_valid is None:
            # No valid control point normals assigned.
            # Just pick a random one.
            for i in range(3):
                v = [0.0, 0.0, 0.0]
                v[i] = 1.0
                rn = cross(array(v), tangents[0])
                d = norm(rn)
                if d > EPSILON:
                    n = rn / d
                    normals[:] = n
                    return normals
            # Really, we only need range(2) for i since the spline line can only be
            # collinear with one of the axis, but we put 3 for esthetics.
            # If we try all three and fail, something is seriously wrong.
            raise RuntimeError("spline normal computation for straight line")
        else:
            for j in range(last_valid + 1, len(lengths)):
                normals[j] = last_normal
        return normals

    def _compute_normals_from_control_points(self, coords):
        # This version ignores the guide atom positions and computes normals
        # at each control point by making it perpendicular to the vectors pointing
        # to control points on either side.  The two ends and any collinear control
        # points are handled afterwards.  We also assume that ribbon cross sections
        # are symmetric in both x- and y-axis so that a twist by more than 90 degrees
        # is equvalent to an opposite twist of (180 - original_twist) degrees.
        from numpy import cross, empty, array, dot
        from numpy.linalg import norm
        #
        # Compute normals by cross-product of vectors to prev and next control points.
        # Normals are for range [1:-1] since 0 and -1 are missing prev and next
        # control points.  Normal index is therefore off by one.
        tangents = self.get_tangents()
        dv = coords[:-1] - coords[1:]
        raw_normals = cross(dv[:-1], -dv[1:])
        for i in range(len(raw_normals)):
            raw_normals[i] = get_orthogonal_component(raw_normals[i], tangents[i+1])
        #
        # Assign normal for first control point.  If there are collinear control points
        # at the beginning, assign same normal for all of them.  If there is no usable
        # normal across the entire "spline" (all control points collinear), just pick
        # a random vector normal to the line.
        normals = empty(coords.shape, float)
        lengths = norm(raw_normals, axis=1)
        for i in range(len(raw_normals)):
            if lengths[i] > EPSILON:
                # Normal for this control point is propagated to all previous ones
                prev_normal = raw_normals[i] / lengths[i]
                prev_index = i
                # Use i+2 because we need to assign one normal for the first control point
                # which has no corresponding raw_normal and (i + 1) for all the control
                # points up to and including this one.
                normals[:i+2] = prev_normal
                break
        else:
            # All control points collinear
            for i in range(3):
                v = [0.0, 0.0, 0.0]
                v[i] = 1.0
                rn = cross(array(v), tangents[0])
                d = norm(rn)
                if d > EPSILON:
                    n = rn / d
                    normals[:] = n
                    return normals
            # Really, we only need range(2) for i since the spline line can only be
            # collinear with one of the axis, but we put 3 for esthetics.
            # If we try all three and fail, something is seriously wrong.
            raise RuntimeError("spline normal computation for straight line")
        #
        # Now we have at least one normal assigned.  This is the anchor and we
        # look for the next control point that has a non-zero raw normal.
        # If we do not find one, then this is the last anchor and we just assign
        # the normal to the remainder of the control points.  Otherwise, we
        # have two normals (perpendicular to the same straight line) for 2 or
        # more control points.  The first normal is from the previous control
        # point whose normal we already set, and the second normal is for the
        # last control point in our range of 2 or more.  If there are more than
        # 2 control points, we interpolate the two normals to get the intermediate
        # normals.
        while i < len(raw_normals):
            if lengths[i] > EPSILON:
                # Our control points run from prev_index to i
                n = raw_normals[i] / lengths[i]
                # First we check whether we should flip it due to too much twist
                if dot(n, prev_normal) < 0:
                    n = -n
                # Now we compute normals for intermediate control points (if any)
                # Instead of slerp, we just use linear interpolation for simplicity
                ncp = i - prev_index
                dv = n - prev_normal
                for j in range(1, ncp):
                    f = j / ncp
                    rn = dv * f + prev_normal
                    d = norm(rn)
                    int_n = rn / d
                    normals[prev_index+j] = int_n
                # Finally, we assign normal for this control point
                normals[i+1] = n
                prev_normal = n
                prev_index = i
            i += 1
        # This is the last part of the spline, so assign the remainder of
        # the normals.
        normals[prev_index+1:] = prev_normal
        return normals

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

    def segment(self, seg, side, divisions, no_twist,
                flip_mode=FLIP_MINIMIZE, prev_normal=None, last=False):
        try:
            coords, tangents, normals = self._seg_cache[seg]
        except KeyError:
            coeffs = [self.coefficients[0][seg],
                      self.coefficients[1][seg],
                      self.coefficients[2][seg]]
            coords, tangents = self._segment_path(coeffs, 0, 1, divisions)
            tangents = normalize_vector_array(tangents)
            ns = self.normals[seg]
            ne = self.normals[seg + 1]
            # import sys
            # print("ns, ne", ns, ne, file=sys.__stderr__); sys.__stderr__.flush()
            if self.ignore_flip_mode:
                flip_mode = FLIP_MINIMIZE
            normals, flipped = constrained_normals(tangents, ns, ne, flip_mode,
                                                   self.flipped[seg], self.flipped[seg + 1], no_twist)
            if flipped:
                self.flipped[seg + 1] = not self.flipped[seg + 1]
                self.normals[seg + 1] = -ne
            #normals = curvature_to_normals(curvature, tangents, prev_normal)
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

    def _segment_path(self, coeffs, tmin, tmax, divisions):
        # Compute coordinates by multiplying spline parameter vector
        # (1, t, t**2, t**3) by the spline coefficients, and
        # compute tangents by multiplying spline parameter vector
        # (0, 1, 2*t, 3*t**2) by the same spline coefficients
        from numpy import array, zeros, ones, linspace, dot
        spline = array(coeffs).transpose()
        nc = divisions + 1
        t = linspace(tmin, tmax, nc)
        t2 = t * t
        st = ones((nc, 4), float)    # spline multiplier matrix
        # st[:,0] is 1.0        # 1
        st[:,1] = t             # t
        st[:,2] = t2            # t^2
        st[:,3] = t * t2        # t^3
        coords = dot(st, spline)
        # st[:,0] stays at 1.0  # 1
        st[:,1] *= 2.0          # 2t
        st[:,2] *= 3.0          # 3t^2
        tangents = dot(st[:,:-1], spline[1:])
        #st[:,0] = 2.0           # 2
        #st[:,1] *= 3.0;         # 6t
        #curvature = dot(st[:,:-2], spline[2:])
        #return coords, tangents, curvature
        return coords, tangents

    def lead_segment(self, divisions):
        coeffs = [self.coefficients[0][0],
                  self.coefficients[1][0],
                  self.coefficients[2][0]]
        # We do not want to go from -0.5 to 0 because the
        # first residue will already have the "0" coordinates
        # as part of its ribbon.  We want to connect to that
        # coordinate smoothly.
        step = 0.5 / (divisions + 1)
        coords, tangents = self._segment_path(coeffs, -0.3, -step, divisions)
        tangents = normalize_vector_array(tangents)
        n = self.normals[0]
        normals, flipped = constrained_normals(tangents, n, n, FLIP_MINIMIZE, False, False, True)
        #normals = curvature_to_normals(curvature, tangents, None)
        return coords, tangents, normals

    def trail_segment(self, divisions, prev_normal=None):
        coeffs = [self.coefficients[0][-1],
                  self.coefficients[1][-1],
                  self.coefficients[2][-1]]
        # We do not want to go from 1 to 1.5 because the
        # last residue will already have the "1" coordinates
        # as part of its ribbon.  We want to connect to that
        # coordinate smoothly.
        step = 0.5 / (divisions + 1)
        coords, tangents = self._segment_path(coeffs, 1 + step, 1.3, divisions)
        tangents = normalize_vector_array(tangents)
        n = self.normals[-1]
        normals, flipped = constrained_normals(tangents, n, n, FLIP_MINIMIZE, False, False, True)
        #normals = curvature_to_normals(curvature, tangents, prev_normal)
        return coords, tangents, normals


class XSectionManager:
    """XSectionManager keeps track of ribbon cross sections used in an AtomicStructure instance.

    Constants:
      Residue classes:
        RC_NUCLEIC - all nucleotides
        RC_COIL - coil
        RC_SHEET_START - first residue in sheet
        RC_SHEET_MIDDLE - middle residue in sheet
        RC_SHEET_END - last residue in sheet
        RC_HELIX_START - first residue in helix
        RC_HELIX_MIDDLE - middle residue in helix
        RC_HELIX_END - last residue in helix
      Cross section styles:
        STYLE_SQUARE - four-sided cross section with right angle corners and faceted lighting
        STYLE_ROUND - round cross section with no corners
        STYLE_PIPING - flat cross section with piping on either end
      Ribbon cross sections:
        RIBBON_NUCLEIC - Use style for nucleotides
        RIBBON_COIL - Use style for coil
        RIBBON_SHEET - Use style for sheet
        RIBBON_SHEET_ARROW - Use style for sheet scaled into an arrow
        RIBBON_HELIX - Use style for helix
        RIBBON_HELIX_ARROW - Use style for helix scaled into an arrow
    """

    # Class constants
    (RC_NUCLEIC, RC_COIL,
     RC_SHEET_START, RC_SHEET_MIDDLE, RC_SHEET_END,
     RC_HELIX_START, RC_HELIX_MIDDLE, RC_HELIX_END) = range(8)
    RC_ANY_SHEET = set([RC_SHEET_START, RC_SHEET_MIDDLE, RC_SHEET_END])
    RC_ANY_HELIX = set([RC_HELIX_START, RC_HELIX_MIDDLE, RC_HELIX_END])
    (STYLE_SQUARE, STYLE_ROUND, STYLE_PIPING) = range(3)
    (RIBBON_NUCLEIC, RIBBON_SHEET, RIBBON_SHEET_ARROW,
     RIBBON_HELIX, RIBBON_HELIX_ARROW, RIBBON_COIL) = range(6)

    def __init__(self, structure):
        import weakref
	# 0.21 is slightly bigger than the default stick radius
	# so ends of stick will be completely hidden by ribbon
	# instead of sticking out partially on the other side
        self.structure = weakref.ref(structure)
        self.scale_helix = (1.0, 0.2)
        self.scale_helix_arrow = ((2.0, 0.2), (0.2, 0.2))
        self.scale_sheet = (1.0, 0.2)
        self.scale_sheet_arrow = ((2.0, 0.2), (0.2, 0.2))
        self.scale_coil = (0.2, 0.2)
        self.scale_nucleic = (0.2, 1.0)
        self.style_helix = self.STYLE_ROUND
        self.style_sheet = self.STYLE_SQUARE
        self.style_coil = self.STYLE_ROUND
        self.style_nucleic = self.STYLE_SQUARE
        self.arrow_helix = False
        self.arrow_sheet = True
        self.params = {
            self.STYLE_ROUND: {
                "sides": 12,
                "faceted": False,
            },
            self.STYLE_SQUARE: {
                # No parameters yet for square style
            },
            self.STYLE_PIPING: {
                "sides": 12,
                "ratio": 0.5,
                "faceted": False,
            },
        }
        self.transitions = {
            # SHEET_START in the middle
            (self.RC_COIL, self.RC_SHEET_START, self.RC_SHEET_MIDDLE):
                (self.RIBBON_COIL, self.RIBBON_SHEET),
            (self.RC_COIL, self.RC_SHEET_START, self.RC_SHEET_END):
                (self.RIBBON_COIL, self.RIBBON_SHEET),
            (self.RC_HELIX_END, self.RC_SHEET_START, self.RC_SHEET_MIDDLE):
                (self.RIBBON_SHEET, self.RIBBON_SHEET),
            (self.RC_HELIX_END, self.RC_SHEET_START, self.RC_SHEET_END):
                (self.RIBBON_SHEET, self.RIBBON_SHEET),
            (self.RC_SHEET_END, self.RC_SHEET_START, self.RC_SHEET_MIDDLE):
                (self.RIBBON_SHEET, self.RIBBON_SHEET),
            (self.RC_SHEET_END, self.RC_SHEET_START, self.RC_SHEET_END):
                (self.RIBBON_SHEET, self.RIBBON_SHEET),
            # SHEET_END in the middle
            (self.RC_SHEET_START, self.RC_SHEET_END, self.RC_COIL):
                (self.RIBBON_SHEET_ARROW, self.RIBBON_COIL),
            (self.RC_SHEET_MIDDLE, self.RC_SHEET_END, self.RC_COIL):
                (self.RIBBON_SHEET_ARROW, self.RIBBON_COIL),
            (self.RC_SHEET_START, self.RC_SHEET_END, self.RC_HELIX_START):
                (self.RIBBON_SHEET, self.RIBBON_SHEET_ARROW),
            (self.RC_SHEET_MIDDLE, self.RC_SHEET_END, self.RC_HELIX_START):
                (self.RIBBON_SHEET, self.RIBBON_SHEET_ARROW),
            (self.RC_SHEET_START, self.RC_SHEET_END, self.RC_SHEET_START):
                (self.RIBBON_SHEET, self.RIBBON_SHEET_ARROW),
            (self.RC_SHEET_MIDDLE, self.RC_SHEET_END, self.RC_SHEET_START):
                (self.RIBBON_SHEET, self.RIBBON_SHEET_ARROW),
            # HELIX_START in the middle
            (self.RC_COIL, self.RC_HELIX_START, self.RC_HELIX_MIDDLE):
                (self.RIBBON_COIL, self.RIBBON_HELIX),
            (self.RC_COIL, self.RC_HELIX_START, self.RC_HELIX_END):
                (self.RIBBON_COIL, self.RIBBON_HELIX),
            (self.RC_HELIX_END, self.RC_HELIX_START, self.RC_HELIX_MIDDLE):
                (self.RIBBON_HELIX, self.RIBBON_HELIX),
            (self.RC_HELIX_END, self.RC_HELIX_START, self.RC_HELIX_END):
                (self.RIBBON_HELIX, self.RIBBON_HELIX),
            (self.RC_SHEET_END, self.RC_HELIX_START, self.RC_HELIX_MIDDLE):
                (self.RIBBON_HELIX, self.RIBBON_HELIX),
            (self.RC_SHEET_END, self.RC_HELIX_START, self.RC_HELIX_END):
                (self.RIBBON_HELIX, self.RIBBON_HELIX),
            # HELIX_END in the middle
            (self.RC_HELIX_START, self.RC_HELIX_END, self.RC_COIL):
                (self.RIBBON_HELIX_ARROW, self.RIBBON_COIL),
            (self.RC_HELIX_MIDDLE, self.RC_HELIX_END, self.RC_COIL):
                (self.RIBBON_HELIX_ARROW, self.RIBBON_COIL),
            (self.RC_HELIX_START, self.RC_HELIX_END, self.RC_HELIX_START):
                (self.RIBBON_HELIX, self.RIBBON_HELIX_ARROW),
            (self.RC_HELIX_MIDDLE, self.RC_HELIX_END, self.RC_HELIX_START):
                (self.RIBBON_HELIX, self.RIBBON_HELIX_ARROW),
            (self.RC_HELIX_START, self.RC_HELIX_END, self.RC_SHEET_START):
                (self.RIBBON_HELIX, self.RIBBON_HELIX_ARROW),
            (self.RC_HELIX_MIDDLE, self.RC_HELIX_END, self.RC_SHEET_START):
                (self.RIBBON_HELIX, self.RIBBON_HELIX_ARROW),
        }

        self._xs_helix = None
        self._xs_helix_arrow = None
        self._xs_sheet = None
        self._xs_sheet_arrow = None
        self._xs_coil = None
        self._xs_nucleic = None

    def assign(self, rc0, rc1, rc2):
        """Return front and back cross sections for the middle residue.

        rc0, rc1 and rc2 are residue classes of three consecutive residues.
        The return value is a 2-tuple of XSection instances.
        The first is for the half-segment between rc0 and rc1, ending at rc1.
        The second is for the half-segment between rc1 and rc2, starting at rc1."""
        if rc1 is self.RC_NUCLEIC:
            return self.xs_nucleic, self.xs_nucleic
        if rc1 is self.RC_SHEET_MIDDLE:
            return self.xs_sheet, self.xs_sheet
        if rc1 is self.RC_HELIX_MIDDLE:
            return self.xs_helix, self.xs_helix
        if rc1 is self.RC_COIL:
            return self.xs_coil, self.xs_coil
        try:
            r_front, r_back = self.transitions[(rc0, rc1, rc2)]
            return self._xs_ribbon(r_front), self._xs_ribbon(r_back)
        except KeyError:
            print("unsupported transition %d-%d-%d" % (rc0, rc1, rc2))
            return self.xs_coil, self.xs_coil

    def is_compatible(self, xs0, xs1):
        """Return if the two cross sections can be blended."""
        return xs0 is xs1

    def set_helix_scale(self, x, y):
        """Set scale factors for helix ribbon cross section."""
        v = (x, y)
        if self.scale_helix != v:
            self.scale_helix = v
            self._xs_helix = None
            self._set_gc_ribbon()

    def set_helix_arrow_scale(self, x1, y1, x2, y2):
        """Set scale factors for helix arrow ribbon cross section."""
        v = ((x1, y1), (x2, y2))
        if self.scale_helix_arrow != v:
            self.scale_helix_arrow = v
            self._xs_helix_arrow = None
            self._set_gc_ribbon()

    def set_sheet_scale(self, x, y):
        """Set scale factors for sheet ribbon cross section."""
        v = (x, y)
        if self.scale_sheet != v:
            self.scale_sheet = v
            self._xs_sheet = None
            self._set_gc_ribbon()

    def set_sheet_arrow_scale(self, x1, y1, x2, y2):
        """Set scale factors for sheet arrow ribbon cross section."""
        v = ((x1, y1), (x2, y2))
        if self.scale_sheet_arrow != v:
            self.scale_sheet_arrow = v
            self._xs_sheet_arrow = None
            self._set_gc_ribbon()

    def set_coil_scale(self, x, y):
        """Set scale factors for coil ribbon cross section."""
        v = (x, y)
        if self.scale_coil != v:
            self.scale_coil = v
            self._xs_coil = None
            self._set_gc_ribbon()

    def set_nucleic_scale(self, x, y):
        """Set scale factors for nucleic ribbon cross section."""
        v = (x, y)
        if self.scale_nucleic != v:
            self.scale_nucleic = v
            self._xs_nucleic = None
            self._set_gc_ribbon()

    def set_helix_style(self, s):
        """Set style for helix ribbon cross section."""
        if self.style_helix != s:
            self.style_helix = s
            self._xs_helix = None
            self._xs_helix_arrow = None
            self._set_gc_ribbon()

    def set_sheet_style(self, s):
        """Set style for sheet ribbon cross section."""
        if self.style_sheet != s:
            self.style_sheet = s
            self._xs_sheet = None
            self._xs_sheet_arrow = None
            self._set_gc_ribbon()

    def set_coil_style(self, s):
        """Set style for coil ribbon cross section."""
        if self.style_coil != s:
            self.style_coil = s
            self._xs_coil = None
            self._set_gc_ribbon()

    def set_nucleic_style(self, s):
        """Set style for helix ribbon cross section."""
        if self.style_nucleic != s:
            self.style_nucleic = s
            self._xs_nucleic = None
            self._set_gc_ribbon()

    def set_transition(self, rc0, rc1, rc2, rf, rb):
        """Set transition for ribbon cross section across residue classes.

        The "assign" method converts residue classes for three consecutive
        residues into two cross sections.  This method defines the values
        that are used by "assign" when the residue classes are different.
        If rc1 is RC_NUCLEIC, RC_COIL, RC_SHEET_MIDDLE, RC_HELIX_MIDDLE,
        the two returned cross sections are the same and match the residue
        class.  In all other cases, a lookup dictionary is used with the
        3-tuple key of (rc0, rc1, rc2).  For this method, the values to
        be inserted into the lookup dictionary are the RIBBON_* constants."""
        key = (rc0, rc1, rc2)
        if key not in self.transitions:
            raise ValueError("transition %d-%d-%d is never used" % key)
        v = (rf, rb)
        if self.transitions[key] != v:
            self.transitions[key] = v
            self._set_gc_ribbon()

    def set_helix_end_arrow(self, b):
        if self.arrow_helix != b:
            self.arrow_helix = b
            self._set_gc_ribbon()

    def set_sheet_end_arrow(self, b):
        if self.arrow_sheet != b:
            self.arrow_sheet = b
            self._set_gc_ribbon()

    def set_params(self, style, **kw):
        param = self.params[style]
        for k in kw.keys():
            if k not in param:
                raise ValueError("unknown parameter %s" % k)
        any_changed = False
        for k, v in kw.items():
            if param[k] != v:
                param[k] = v
                any_changed = True
                self._set_gc_ribbon()
        if any_changed:
            if self.style_helix == style:
                self._xs_helix = None
                self._xs_helix_arrow = None
            if self.style_sheet == style:
                self._xs_sheet = None
                self._xs_sheet_arrow = None
            if self.style_coil == style:
                self._xs_coil = None
            if self.style_nucleic == style:
                self._xs_nucleic = None

    @property
    def xs_helix(self):
        if self._xs_helix is None:
            self._xs_helix = self._make_xs(self.style_helix, self.scale_helix)
        return self._xs_helix

    @property
    def xs_helix_arrow(self):
        if self._xs_helix_arrow is None:
            base = self._make_xs(self.style_helix, (1.0, 1.0))
            self._xs_helix_arrow = base.arrow(self.scale_helix_arrow)
        return self._xs_helix_arrow

    @property
    def xs_sheet(self):
        if self._xs_sheet is None:
            self._xs_sheet = self._make_xs(self.style_sheet, self.scale_sheet)
        return self._xs_sheet

    @property
    def xs_sheet_arrow(self):
        if self._xs_sheet_arrow is None:
            base = self._make_xs(self.style_sheet, (1.0, 1.0))
            self._xs_sheet_arrow = base.arrow(self.scale_sheet_arrow)
        return self._xs_sheet_arrow

    @property
    def xs_coil(self):
        if self._xs_coil is None:
            self._xs_coil = self._make_xs(self.style_coil, self.scale_coil)
        return self._xs_coil

    @property
    def xs_nucleic(self):
        if self._xs_nucleic is None:
            self._xs_nucleic = self._make_xs(self.style_nucleic, self.scale_nucleic)
        return self._xs_nucleic

    def _make_xs(self, style, scale):
        if style is self.STYLE_ROUND:
            return self._make_xs_round(scale)
        elif style is self.STYLE_SQUARE:
            return self._make_xs_square(scale)
        elif style is self.STYLE_PIPING:
            return self._make_xs_piping(scale)
        else:
            raise ValueError("unknown style %s" % style)

    def _make_xs_round(self, scale):
        from numpy import array
        from .molobject import RibbonXSection as XSection
        coords = []
        normals = []
        param = self.params[self.STYLE_ROUND]
        sides = param["sides"]
        from numpy import linspace, cos, sin, stack
        from math import pi
        angles = linspace(0, 2 * pi, sides, endpoint=False)
        ca = cos(angles)
        sa = sin(angles)
        circle = stack((ca, sa), axis=1)
        coords = circle * array(scale)
        if param["faceted"]:
            return XSection(coords, faceted=True)
        else:
            normals = circle * array((scale[1], scale[0]))
            return XSection(coords, normals=normals, faceted=False)

    def _make_xs_square(self, scale):
        from numpy import array
        from .molobject import RibbonXSection as XSection
        coords = array(((1.0, 1.0), (-1.0, 1.0), (-1.0, -1.0), (1.0, -1.0))) * array(scale)
        return XSection(coords, faceted=True)

    def _make_xs_piping(self, scale):
        from numpy import array
        from math import pi, cos, sin, asin
        from .molobject import RibbonXSection as XSection
        if scale[0] > scale[1]:
            flipped = False
            delta = scale[0] - scale[1]
            radius = scale[1]
        else:
            flipped = True
            delta = scale[1] - scale[0]
            radius = scale[0]
        coords = []
        normals = []
        param = self.params[self.STYLE_PIPING]
        # The total number of sides is param["sides"]
        # We subtract the two connectors and then divide
        # by two while rounding up
        sides = (param["sides"] - 1) // 2
        ratio = param["ratio"]
        theta = asin(ratio)
        side_angle = 2.0 * (pi - theta) / sides
        # Generate the vertices for the two piping on either side.
        # The first and last points are used to connect the two.
        for start_angle, offset in ((theta - pi, delta), (theta, -delta)):
            for i in range(sides + 1):
                angle = start_angle + i * side_angle
                x = cos(angle) * radius
                y = sin(angle) * radius
                if flipped:
                    normals.append((-y, x))
                    coords.append((-y, x + offset))
                else:
                    normals.append((x, y))
                    coords.append((x + offset, y))
        coords = array(coords)
        tess = ([(0, sides, sides+1),(0, sides+1, 2*sides+1)] +         # connecting rectangle
                [(0, i, i+1) for i in range(1, sides)] +                # first piping
                [(sides+1, i, i+1) for i in range(sides+2, 2*sides+1)]) # second piping
        if param["faceted"]:
            return XSection(coords, faceted=True, tess=tess)
        else:
            if flipped:
                normals[0] = normals[-1] = (1, 0)
                normals[sides] = normals[sides + 1] = (-1, 0)
            else:
                normals[0] = normals[-1] = (0, -1)
                normals[sides] = normals[sides + 1] = (0, 1)
            normals = array(normals)
            return XSection(coords, normals=normals, faceted=False, tess=tess)

    def _xs_ribbon(self, r):
        if r is self.RIBBON_HELIX:
            return self.xs_helix
        elif r is self.RIBBON_SHEET:
            return self.xs_sheet
        elif r is self.RIBBON_COIL:
            return self.xs_coil
        elif r is self.RIBBON_NUCLEIC:
            return self.xs_nucleic
        elif r is self.RIBBON_HELIX_ARROW:
            if self.arrow_helix:
                return self.xs_helix_arrow
            else:
                return self.xs_helix
        elif r is self.RIBBON_SHEET_ARROW:
            if self.arrow_sheet:
                return self.xs_sheet_arrow
            else:
                return self.xs_sheet
        else:
            raise ValueError("unknown ribbon ref %d" % r)

    def _set_gc_ribbon(self):
        # Mark ribbon for rebuild
        s = self.structure()
        if s is not None:
            s._graphics_changed |= s._RIBBON_CHANGE


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


def constrained_normals(tangents, n_start, n_end, flip_mode, s_flipped, e_flipped, no_twist):
    from .molc import c_function
    import ctypes
    f = c_function("constrained_normals",
                   args=(ctypes.py_object, ctypes.py_object, ctypes.py_object,
                         ctypes.c_int, ctypes.c_bool, ctypes.c_bool, ctypes.c_bool),
                   ret = ctypes.py_object)
    return f(tangents, n_start, n_end, flip_mode, s_flipped, e_flipped, no_twist)


def curvature_to_normals(curvature, tangents, prev_normal):
    from numpy.linalg import norm
    from numpy import empty, cross, dot
    normals = empty(curvature.shape, dtype=float)
    for i in range(len(curvature)):
        c_len = norm(curvature[i])
        if c_len < EPSILON:
            # No curvature, must use previous normal
            # TODO: more here
            pass
        else:
            # Normal case
            n = cross(curvature[i], tangents[i])
            if prev_normal is not None and dot(prev_normal, n) < 0:
                n = -n
            prev_normal = normals[i] = n
    #normals = normalize_vector_array(cross(curvature, tangents))
    normals = normalize_vector_array(normals)
    return normals



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
