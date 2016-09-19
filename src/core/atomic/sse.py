# vim: set expandtab shiftwidth=4 softtabstop=4:

class OptLine:
    """Optimize for a straight line through a set of coordinates.
    
    Starting with an initial guess of a centroid and a direction,
    try to minimize the variance of distance of the coordinates
    from the center line."""

    DEFAULT_MAX_ITERATIONS = 5

    def __init__(self, coords, centroid, axis,
                 maxiter=DEFAULT_MAX_ITERATIONS, tol=0.1):
        from scipy.optimize import minimize
        from numpy.linalg import norm
        from numpy import mean
        self.coords = coords
        guess = self._encode(centroid, axis)
        options = {"disp":False, "maxiter":maxiter}
        res = minimize(self._residual, guess, tol=tol, options=options)
        # print("straight residual", self._residual(res.x),
        #       "iterations", res.nit)
        # Even on failure, we use the last results from the optimization
        # rather than the initial guess.
        self.centroid, self.axis = self._decode(res.x)

    def _encode(self, centroid, axis):
        from numpy import array
        return array([centroid[0], centroid[1], centroid[2],
                      axis[0], axis[1], axis[2]])

    def _decode(self, params):
        from numpy.linalg import norm
        centroid = params[:3]
        axis = params[3:6]
        axis = axis / norm(axis)
        return centroid, axis

    def _residual(self, params):
        from numpy import dot, sqrt, sum, fabs, mean, var
        centroid, axis = self._decode(params)
        # x = atom coordinates (vector Nx3)
        # xmp = coordinates relative to centroid (vector Nx3)
        xmp = self.coords - centroid
        # xa = coordinates . axis (vector N)
        xa = dot(self.coords, axis)
        # ca = centroid . axis (scalar)
        ca = dot(centroid, axis)
        # f = squared distance from torus center line (vector N)
        f = sum(xmp * xmp, axis=1) - xa * xa + 2 * xa * ca - ca * ca
        # residual = variance in squared distance
        res = var(f)
        return res


class OptArc:
    """Optimize for an arc through a set of coordinates.
    
    Starting with an initial guess of a centroid and a direction,
    try to minimize the variance of distance of the coordinates
    from the arc."""

    DEFAULT_MAX_ITERATIONS = 5

    def __init__(self, coords, centroid, axis, radius,
                 maxiter=DEFAULT_MAX_ITERATIONS, tol=0.1):
        from scipy.optimize import minimize
        from numpy.linalg import norm
        from numpy import mean
        self.coords = coords
        guess = self._encode(centroid, axis, radius)
        options = {"disp":False, "maxiter":maxiter}
        res = minimize(self._residual, guess, tol=tol, options=options)
        # print("straight residual", self._residual(res.x),
        #       "iterations", res.nit)
        # Even on failure, we use the last results from the optimization
        # rather than the initial guess.
        self.center, self.axis, self.radius = self._decode(res.x)

    def _encode(self, center, axis, radius):
        from numpy import array
        return array([center[0], center[1], center[2],
                      axis[0], axis[1], axis[2], radius])

    def _decode(self, params):
        from numpy.linalg import norm
        center = params[:3]
        axis = params[3:6]
        axis = axis / norm(axis)
        radius = params[6]
        return center, axis, radius

    def _residual(self, params):
        from numpy import dot, outer, stack, newaxis, var
        from numpy.linalg import norm
        center, axis, radius = self._decode(params)
        # Calculate the vector from atom to torus center (vector Nx3)
        rel_coords = self.coords - center
        # Get distance of atom from plane of torus (vector N)
        y = dot(rel_coords, axis)
        # Get projection of atom onto plane of torus (vector Nx3)
        in_plane = rel_coords - outer(y, axis)
        # Get distance from circle in plane of torus (vector N)
        # and then atom distance from circle (vector N)
        x = norm(in_plane, axis=1) - radius
        delta = norm(stack([x, y], axis=1), axis=1)
        # Alternate computation for verification
        # uv = in_plane / norm(in_plane, axis=1)[:, newaxis]
        # centers = center + uv * radius
        # radii = norm(self.coords - centers, axis=1)
        # Get difference from ideal (vector N)
        # delta = radii - minor_radius
        # res = norm(delta)
        res = var(delta)
        # print("residual", res)
        return res


class HelixCylinder:
    """Compute the best-fit cylinder when given atomic coordinates.

    The goal is to minimize the total squared distance of atoms
    from the surface of the cylinder.  The best-fit cylinder may
    be either straight or curved.

    A straight cylinder is described by three parameters:
    a point on the cylinder center line, orientation axis vector,
    and radius.  We do not define the ends of the cylinder.

    A curved cylinder is actually a section of a torus and is
    described by four parameters: center, orientation axis,
    major radius (radius of the torus center-line circle) and
    minor radius (radius of the circular cross section).
    Again, we do not define the ends of the cylinder.

    A straight cylinder is used for fewer than 13 residues
    because that is roughly 3 turns of an alpha helix.
    Using curved cylinders for shorter helices often result
    in cylinders that minimize the atom-to-cylinder-surface
    distances but look wrong.
    """

    MIN_CURVE_LENGTH = 13

    def __init__(self, coords, maxiter=None):
        self.coords = coords
        self.maxiter = maxiter
        self._centers = None
        self._normals = None
        self._surface = None
        if len(coords) < self.MIN_CURVE_LENGTH:
            self._straight_optimize()
        else:
            self._try_curved()

    def cylinder_radius(self):
        """Return radius of cylinder."""
        if self.curved:
            return self.minor_radius
        else:
            return self.radius

    def cylinder_centers(self):
        """Return array of points on center line of cylinder.
        
        The returned points are the nearest points on the cylinder
        center line nearest the given atomic coordinates."""
        from numpy import dot, outer, newaxis
        from numpy.linalg import norm
        if self._centers is not None:
            return self._centers
        if self.curved:
            # Calculate the vector from atom to torus center (vector Nx3)
            rel_coords = self.coords - self.center
            # Get distance of atom from plane of torus (vector N)
            y = dot(rel_coords, self.axis)
            # Get projection of atom onto plane of torus (vector Nx3)
            in_plane = rel_coords - outer(y, self.axis)
            # Get unit vector from torus center
            # to in_plane position (vector Nx3)
            uv = in_plane / norm(in_plane, axis=1)[:, newaxis]
            # Get centers by projecting along unit vectors
            self._centers = self.center + uv * self.major_radius
        else:
            # Get distance of each atomic coordinate
            # from centroid along the center line
            d = dot(self.coords - self.centroid, self.axis)
            # Get centers by adding offsets to centroid along axis
            self._centers = self.centroid + outer(d, self.axis)
        return self._centers

    def cylinder_normals(self):
        """Return tuple of two arrays for the normals and binormals.

        Normals and binormals are relative to the cylinder center."""
        if self._normals is not None:
            return self._normals
        from numpy import tile, vdot, newaxis, cross
        from numpy.linalg import norm
        tile_shape = [len(self.coords), 1]
        if self.curved:
            normals = tile(self.axis, tile_shape)
            in_plane = self.coords - self.center
            binormals = in_plane / norm(in_plane, axis=1)[:, newaxis]
            self._normals = (normals, binormals)
        else:
            centers = self.cylinder_centers()
            delta = self.coords[1] - self.centroid
            d = vdot(delta, self.axis)
            normal = delta - d * self.axis
            normal = normal / norm(normal)
            binormal = cross(self.axis, normal)
            self._normals = (tile(normal, tile_shape),
                             tile(binormal, tile_shape))
        return self._normals

    def cylinder_surface(self):
        """Return (array of points, array of normals) on cylinder surface.
        
        The returned points are the nearest points on the cylinder
        surface nearest the given atomic coordinates.
        The returned normals are perpendicular to both the cylinder
        center and the line of projection from the atomic coordinates
        to the cylinder center line."""
        if self._surface is not None:
            return self._surface
        from numpy import newaxis
        from numpy.linalg import norm
        centers = self.cylinder_centers()
        deltas = self.coords - centers
        uv = delta / norm(delta, axis=1)[:, newaxis]
        if self.curved:
            self._surface = centers + uv * self.minor_radius
        else:
            self._surface = centers + uv * self.radius
        return self._surface

    def cylinder_intermediates(self):
        """Return three arrays (points, normals, binormals) for intermediates.

        Intermediate points are points half way between points returned
        by ''cylinder_center''.  These values are useful when rendering
        the cylinder such that each segment can be independently displayed
        and colored with sharp boundaries."""
        from numpy import tile, newaxis
        from numpy.linalg import norm
        centers = self.cylinder_centers()
        if self.curved:
            t = (centers[:-1] + centers[1:]) - self.center
            normals = tile(self.axis, [len(centers) - 1, 1])
            binormals = t / norm(t, axis=1)[:, newaxis]
            ipoints = binormals * self.cylinder_radius() + self.center
        else:
            normals, binormals = self.cylinder_normals()
            normals = normals[:-1]
            binormals = binormals[:-1]
            ipoints = (centers[:-1] + centers[1:]) / 2
        return ipoints, normals, binormals

    def _try_curved(self):
        from numpy import mean, cross, sum, vdot
        from numpy.linalg import norm
        from math import sqrt
        # First we compute three centroids at the
        # front, middle and end of the helix.
        # We assume all three points are on (or at least
        # near) the center line of the torus.  We can then
        # estimate the torus center, orientation and major
        # radius.  The minor radius is estimated from the
        # distance of atoms to the torus center line.
        # We do not use the first or last coordinates
        # because they tend to deviate from the
        # cylinder surface more than the middle ones.
        p1 = mean(self.coords[1:4], axis=0)
        mid = len(self.coords) // 2
        p2 = mean(self.coords[mid - 1: mid + 2], axis=0)
        p3 = mean(self.coords[-4:-1], axis=0)
        t = p2 - p1
        u = p3 - p1
        v = p3 - p2
        w = cross(t, u)        # triangle normal
        wsl = sum(w * w)    # square of length of w
        if wsl < 1e-8:
            # Helix does not curve
            # print("helix straight")
            self._straight_optimize()
        else:
            iwsl2 = 1.0 / (2 * wsl)
            tt = vdot(t, t)
            uu = vdot(u, u)
            c_center = p1 + (u * tt * vdot(u, v) - t * uu * vdot(t, v)) * iwsl2
            c_radius = sqrt(tt * uu * vdot(v, v) * iwsl2 * 0.5)
            c_axis = w / sqrt(wsl)
            # print("helix curved: center", c_center, "radius", c_radius,
            #     "axis", c_axis)
            self._curved_optimize(c_center, c_axis, c_radius)

    def _straight_optimize(self):
        from numpy.linalg import norm
        from numpy import mean
        centroid, axis, radius = self._straight_initial()
        opt = OptLine(self.coords, centroid, axis)
        self.curved = False
        self.centroid = opt.centroid
        self.axis = opt.axis
        radii = norm(self.coords - self.cylinder_centers(), axis=1)
        self.radius = mean(radii)

    def _straight_initial(self):
        from numpy import mean, argmax, dot, newaxis
        from numpy.linalg import svd, norm
        centroid = mean(self.coords, axis=0)
        rel_coords = self.coords - centroid
        ignore, vals, vecs = svd(rel_coords)
        axis = vecs[argmax(vals)]
        axis_pos = dot(rel_coords, axis)[:, newaxis]
        radial_vecs = rel_coords - axis * axis_pos
        radius = mean(norm(radial_vecs, axis=1))
        return centroid, axis, radius

    def _curved_optimize(self, center, axis, major_radius):
        from numpy.linalg import norm
        from numpy import mean
        opt = OptArc(self.coords, center, axis, major_radius)
        self.curved = True
        self.center = opt.center
        self.axis = opt.axis
        self.major_radius = opt.radius
        radii = norm(self.coords - self.cylinder_centers(), axis=1)
        self.minor_radius = mean(radii)


class StrandPlank:
    """Compute the best-fit plank when given atomic coordinates.

    The goal is to minimize the total squared distance of atoms
    from the surface of the plank.  The best-fit plank may
    be either straight or curved.

    A straight plank is described by four parameters:
    a point on the plank center line, orientation axis vector,
    width parallel to the orientation axis, and thickness perpendicular
    to the orientation axis.  We do not define the ends of the plank.

    A curved cylinder is actually a section of a torus and is
    described by four parameters: center, orientation axis,
    major radius (radius of the torus center-line circle) and
    minor radius (radius of the circular cross section).
    Again, we do not define the ends of the cylinder.

    A straight cylinder is used for fewer than 13 residues
    because that is roughly 3 turns of an alpha helix.
    Using curved cylinders for shorter helices often result
    in cylinders that minimize the atom-to-cylinder-surface
    distances but look wrong.
    """

    MIN_CURVE_LENGTH = 9
    DEFAULT_MAX_ITERATIONS = 5

    def __init__(self, coords, maxiter=DEFAULT_MAX_ITERATIONS):
        self.coords = coords
        self.maxiter = maxiter
        self._centers = None
        self._normals = None
        self._surface = None
        if len(coords) < self.MIN_CURVE_LENGTH:
            self._straight_optimize()
        else:
            self._straight_optimize()
            # self._try_curved()

    def plank_centers(self):
        """Return array of points on center line of plank.
        
        The returned points are the nearest points on the plank
        center line nearest the given atomic coordinates."""
        from numpy import dot, outer, newaxis
        from numpy.linalg import norm
        if self._centers is not None:
            return self._centers
        if self.curved:
            # TODO: verify this works
            # Calculate the vector from atom to torus center (vector Nx3)
            rel_coords = self.coords - self.center
            # Get distance of atom from plane of torus (vector N)
            y = dot(rel_coords, self.axis)
            # Get projection of atom onto plane of torus (vector Nx3)
            in_plane = rel_coords - outer(y, self.axis)
            # Get unit vector from torus center
            # to in_plane position (vector Nx3)
            uv = in_plane / norm(in_plane, axis=1)[:, newaxis]
            # Get centers by projecting along unit vectors
            self._centers = self.center + uv * self.major_radius
        else:
            # Get distance of each atomic coordinate
            # from centroid along the center line
            d = dot(self.coords - self.centroid, self.axis)
            # Get centers by adding offsets to centroid along axis
            self._centers = self.centroid + outer(d, self.axis)
        return self._centers

    def plank_size(self):
        return self.width, self.thickness

    def plank_normals(self):
        if self._normals is not None:
            return self._normals
        if self.curved:
            raise ValueError("unimplemented")
        else:
            # TODO: more here
            from numpy import cross, tile
            bn = cross(self.width_vector, self.axis)
            shape = (len(self.coords), 1)
            self._normals = tile(self.width_vector, shape), tile(bn, shape)
        return self._normals

    def _straight_optimize(self):
        from numpy import vdot, cross, dot, stack, argsort, mean, fabs
        from numpy.linalg import svd, norm
        centroid, axis, width_vector = self._straight_initial()
        opt = OptLine(self.coords, centroid, axis)
        self.curved = False
        self.centroid = opt.centroid
        self.axis = opt.axis
        # Make sure axis is pointing from front to back
        p_dir = self.coords[-1] - self.coords[0]
        if vdot(self.axis, p_dir) < 0:
            self.axis = -self.axis
        # Get the coordinates relative to centroid
        centers = self.plank_centers()
        rel_coords = self.coords - centers
        # Get vectors that define reference coordinate system
        # for computing the orientation of the plank
        mid_pt = rel_coords[len(rel_coords) // 2]
        ref_u = cross(self.axis, mid_pt)
        ref_u /= norm(ref_u)
        ref_v = cross(ref_u, self.axis)
        # Get the coordinates of each point in the reference
        # coordinate system
        cu = dot(rel_coords, ref_u)
        cv = dot(rel_coords, ref_v)
        uv = stack((cu, cv), axis=1)
        # Find the best fit 2D line through the reference
        # coordinates
        ignore, vals, vecs = svd(uv)
        order = argsort(vals)
        width_uv = vecs[order[-1]]
        thickness_uv = vecs[order[0]]
        # Convert best line in reference coordinate system
        # back into vector
        width_vector = width_uv[0] * ref_u + width_uv[1] * ref_v
        thickness_vector = thickness_uv[0] * ref_u + thickness_uv[1] * ref_v
        # Calculate dimensions of plank
        width = mean(fabs(dot(uv, width_uv)))
        thickness = mean(fabs(dot(uv, thickness_uv)))
        # Save results
        self.width_vector = width_vector
        self.thickness_vector = thickness_vector
        self.width = width
        self.thickness = thickness

    def _straight_initial(self):
        from numpy import mean, argsort
        from numpy.linalg import svd, norm
        centroid = mean(self.coords, axis=0)
        rel_coords = self.coords - centroid
        ignore, vals, vecs = svd(rel_coords)
        order = argsort(vals)
        # The eigenvalues are sorted in increasing order,
        # so the last is the principal direction, the next
        # to last is the width direction, preceded by the
        # thickness direction.  All eigenvectors are unit
        # vectors.
        axis = vecs[order[-1]]
        width_vector = vecs[order[-2]]
        return centroid, axis, width_vector
