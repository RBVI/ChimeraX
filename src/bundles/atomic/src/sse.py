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

class OptLine:
    """Optimize for a straight line through a set of coordinates.
    
    Starting with an initial guess of a centroid and a direction,
    try to minimize the variance of distance of the coordinates
    from the center line."""

    DEFAULT_MAX_ITERATIONS = 5

    def __init__(self, coords, centroid, axis,
                 maxiter=DEFAULT_MAX_ITERATIONS, tol=0.1):
        from scipy.optimize import minimize
        self.coords = coords
        guess = self._encode(centroid, axis)
        options = {"disp":False, "maxiter":maxiter}
        res = minimize(self._residual, guess, tol=tol, options=options)
        # print("straight residual", self._residual(res.x),
        #       "iterations", res.nit)
        # Even on failure, we use the last results from the optimization
        # rather than the initial guess.
        from numpy import dot, sqrt, sum, fabs, mean, var
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
        centroid, axis = self._decode(params)
        if False:
            from numpy import dot, sqrt, sum, fabs, mean, var
            # x = atom coordinates (vector Nx3)
            # xmp = coordinates relative to centroid (vector Nx3)
            xmp = self.coords - centroid
            # xa = coordinates . axis (vector N)
            xa = dot(self.coords, axis)
            # ca = centroid . axis (scalar)
            ca = dot(centroid, axis)
            # f = squared distance from cylinder center line (vector N)
            f = sum(xmp * xmp, axis=1) - xa * xa + 2 * xa * ca - ca * ca
            # residual = variance in squared distance
            res = var(f)
        else:
            from numpy import dot, outer, var
            from numpy.linalg import norm
            d = dot(self.coords - centroid, axis)
            centers = centroid + outer(d, axis)
            radii = norm(self.coords - centers, axis=1)
            res = var(radii)
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
        from numpy import mean
        self.coords = coords
        guess = self._encode(centroid, axis, radius)
        options = {"disp":False, "maxiter":maxiter}
        # print("arc residual: initial", self._residual(guess))
        res = minimize(self._residual, guess, tol=tol, options=options)
        # print("arc residual: final", self._residual(res.x),
        #       "after", res.nit, "iterations")
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
        from numpy import dot, outer, stack
        from numpy.linalg import norm
        center, axis, radius = self._decode(params)
        # Calculate the vector from atom to arc center (vector Nx3)
        rel_coords = self.coords - center
        # Get distance of atom from plane of arc (vector N)
        y = dot(rel_coords, axis)
        # Get projection of atom onto plane of arc (vector Nx3)
        in_plane = rel_coords - outer(y, axis)
        # Get vector from arc center to projected atom
        uv = _normalize_vector_array(in_plane)
        # Get positions on circle in direction of atoms
        centers = center + uv * radius
        # Residual minimizes total distance from atom to ideal
        res = norm(centers - self.coords)
        #print("residual", res)
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
    # Ideal coordinates and parameters were generated using the Build
    # Structure and Axes/Planes/Centroids tools in Chimera.  The helix
    # sequences is ALA(18) and the coordinates are from some of the
    # middle residues.  The axis parameters are (center, direction).
    from numpy import array
    IDEAL_COORDS = array([          # CA coordinates
        (-4.543,-1.381,-5.088),
        (-4.871,-2.280,-1.408),
        (-1.332,-3.668,-1.376),
        (-0.007,-0.471,-2.952),
        (-1.782, 1.624,-0.322),
        (-0.267,-0.483, 2.456),
        ( 3.207,-0.031, 0.978),
        ( 2.712, 3.735, 0.827),
        ( 1.634, 3.794, 4.473),
    ])
    IDEAL_PARAMS = array([          # center, axis
        (-0.395,-0.049,-0.215),   
        ( 0.613, 0.501, 0.610),
    ])

    def __init__(self, coords, radius=None, maxiter=None):
        self.coords = coords
        self.maxiter = maxiter
        self._centers = None
        self._directions = None
        self._normals = None
        self._surface = None
        if len(coords) < self.MIN_CURVE_LENGTH:
            self._straight_optimize()
        else:
            self._try_curved()
        if radius is not None:
            if self.curved:
                self.minor_radius = radius
            else:
                self.radius = radius

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
        from numpy import dot, outer, cross, argsort
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
            uv = _normalize_vector_array(in_plane)
            # Get centers by projecting along unit vectors
            centers = self.center + uv * self.major_radius
            # For consecutive residues, the cross product
            # of the vectors from the arc center to the residues
            # should be in the same direction as the arc axis.
            # If not, we flip the coordinates for the two residues
            # to make sure that the arc does not double back on itself.
            num_pts = len(centers)
            order = list(range(num_pts))
            dv = centers - self.center
            i = 1
            while i < num_pts:
                v = cross(dv[order[i-1]], dv[order[i]])
                if dot(v, self.axis) >= 0:
                    i += 1
                else:
                    order[i-1], order[i] = order[i], order[i-1]
                    if i > 1:
                        # Since we flipped, we no longer know whether
                        # (i-2,i-1) is okay, so we go back and check
                        i -= 1
            self._centers = centers[order]
        else:
            # Get distance of each atomic coordinate
            # from centroid along the center line
            d = dot(self.coords - self.centroid, self.axis)
            d.sort()
            # Get centers by adding offsets to centroid along axis
            self._centers = self.centroid + outer(d, self.axis)
        return self._centers

    def cylinder_directions(self):
        """Return array for the direction vectors.

        The returned array are the direction of the cylinder
        corresponding to the given atomic coordinates."""
        if self._directions is not None:
            return self._directions
        from numpy import tile, cross
        if self.curved:
            centers = self.cylinder_centers()
            dv = cross(centers - self.center, self.axis)
            self._directions = _normalize_vector_array(dv)
        else:
            self._directions = tile(self.axis, (len(self.coords), 1))
        return self._directions

    def cylinder_normals(self):
        """Return tuple of two arrays for the normals and binormals.

        Normals and binormals are relative to the cylinder center."""
        if self._normals is not None:
            return self._normals
        if self.curved:
            tile_shape = [len(self.coords), 1]
            from numpy import tile
            normals = tile(self.axis, tile_shape)
            centers = self.cylinder_centers()
            in_plane = centers - self.center
            binormals = _normalize_vector_array(in_plane)
            self._normals = (normals, binormals)
        else:
            self._normals = self._straight_cylinder_normals(len(self.coords))
        return self._normals

    def _straight_cylinder_normals(self, n):
        d = self.coords[1] - self.centroid
        # We do not use:
        #   normal = self.coords[1] - centers[1]
        # because the order of the centers MAY not
        # match the orders of the coords if the coords
        # projection are out of order, i.e., they
        # double back on themselves, which would
        # result in bad rendering of cylinders.
        from numpy import tile, dot, cross
        normal = d - dot(d, self.axis) * self.axis
        from numpy.linalg import norm
        normal = normal / norm(normal)
        binormal = cross(self.axis, normal)
        return (tile(normal, (n,1)),
                tile(binormal, (n,1)))

    def cylinder_surface(self):
        """Return array of points on cylinder surface.
        
        The returned points are the nearest points on the cylinder
        surface nearest the given atomic coordinates."""
        if self._surface is not None:
            return self._surface
        centers = self.cylinder_centers()
        delta = self.coords - centers
        uv = _normalize_vector_array(delta)
        if self.curved:
            self._surface = centers + uv * self.minor_radius
        else:
            self._surface = centers + uv * self.radius
        return self._surface

    def cylinder_intermediates(self, extend = 0.3):
        """Return three arrays (points, normals, binormals) for intermediates.

        Intermediate points are points half way between points returned
        by ''cylinder_center''.  These values are useful when rendering
        the cylinder such that each segment can be independently displayed
        and colored with sharp boundaries."""
        from numpy import tile
        centers = self.cylinder_centers()
        centers = self._extend_ends(centers, frac = 2*extend)
        if self.curved:
            v = centers - self.center
            t = v[:-1] + v[1:]
            normals = tile(self.axis, [len(t), 1])
            binormals = _normalize_vector_array(t)
            ipoints = binormals * self.major_radius + self.center
        else:
            ipoints = (centers[:-1] + centers[1:]) / 2
            normals, binormals = self._straight_cylinder_normals(len(ipoints))
        return ipoints, normals, binormals

    def _extend_ends(self, centers, frac):
        n = len(centers)
        if self.curved:
            tc = self.center  # Torus center
            r0,r1 = centers[0] - tc, centers[-1] - tc # radial vectors from center of torus.
            from chimerax.geometry import angle, rotation
            a = frac * angle(r0,r1) / (n-1)
            c0 = tc + rotation(self.axis, -a) * r0
            c1 = tc + rotation(self.axis, a) * r1
        else:
            e = frac/(n-1) * (centers[-1] - centers[0])
            c0 = centers[0] - e
            c1 = centers[-1] + e
        n = len(centers)
        from numpy import concatenate
        ecenters = concatenate((c0.reshape(1,3), centers, c1.reshape(1,3)))
        return ecenters
    
    def _try_curved(self):
        from numpy import mean, cross, sum, vdot
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
        from numpy import mean, vdot
        centroid, axis, radius = self._straight_initial()
        opt = OptLine(self.coords, centroid, axis)
        self.curved = False
        self.centroid = opt.centroid
        self.axis = opt.axis
        if vdot(self.coords[-1] - self.coords[0], self.axis) < 0:
            self.axis = -self.axis
        radii = norm(self.coords - self.cylinder_centers(), axis=1)
        self.radius = mean(radii)

    def _straight_initial(self):
        from numpy import mean, dot, newaxis
        from numpy.linalg import norm
        if len(self.coords) > len(self.IDEAL_COORDS):
            # "Normal" helices can be approximated by using all
            # coordinates on the assumption that the helix length
            # is sufficiently larger than the helix radius that
            # the biggest eigenvector will be the helical axis
            from numpy.linalg import svd
            from numpy import argmax
            centroid = mean(self.coords, axis=0)
            rel_coords = self.coords - centroid
            ignore, vals, vecs = svd(rel_coords)
            axis = vecs[argmax(vals)]
        else:
            from chimerax.geometry import align_points
            num_pts = len(self.coords)
            tf, rmsd = align_points(self.IDEAL_COORDS[:num_pts], self.coords)
            centroid = tf * self.IDEAL_PARAMS[0]
            axis = tf.transform_vector(self.IDEAL_PARAMS[1])
            rel_coords = self.coords - centroid
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
        self.minor_radius = min(2.5, mean(radii))


class StrandPlank:
    """Compute the best-fit plank when given atomic coordinates.

    The goal is to minimize the total squared distance of atoms
    from the surface of the plank.  The best-fit plank may
    be either straight or curved.

    A straight plank is described by four parameters:
    a point on the plank center line, orientation axis vector,
    width parallel to the orientation axis, and thickness perpendicular
    to the orientation axis.  We do not define the ends of the plank.

    A curved plank described by six parameters: circle center,
    orientation axis, circle radius, angle of plank relative
    to plane of circle, width of plank along the angle, height
    of plank perpendicular to the angle.

    A straight cylinder is used for short strands
    because there is not enough data for good averaging.
    """

    MIN_CURVE_LENGTH = 7

    def __init__(self, coords, guides, maxiter=None):
        self.coords = coords
        self.guides = guides
        self.maxiter = maxiter
        self._centers = None
        self._directions = None
        self._normals = None
        self._surface = None
        if len(coords) < self.MIN_CURVE_LENGTH:
            self._straight_optimize()
        else:
            self._try_curved()

    def plank_centers(self):
        """Return array of points on center line of plank.
        
        The returned points are the nearest points on the plank
        center line nearest the given atomic coordinates."""
        from numpy import dot, outer
        if self._centers is not None:
            return self._centers
        if self.curved:
            # Calculate the vector from atom to circle center (vector Nx3)
            rel_coords = self.coords - self.center
            # Get distance of atom from plane of circle (vector N)
            y = dot(rel_coords, self.axis)
            # Get projection of atom onto plane of circle (vector Nx3)
            in_plane = rel_coords - outer(y, self.axis)
            # Get unit vector from circle center
            # to in_plane position (vector Nx3)
            uv = _normalize_vector_array(in_plane)
            # Get centers by projecting along unit vectors
            self._centers = self.center + uv * self.radius
        else:
            # Get distance of each atomic coordinate
            # from centroid along the center line
            d = dot(self.coords - self.centroid, self.axis)
            # Get centers by adding offsets to centroid along axis
            self._centers = self.centroid + outer(d, self.axis)
        return self._centers

    def plank_directions(self):
        """Return array for the direction vectors.

        The returned array are the direction of the plank
        corresponding to the given atomic coordinates."""
        if self._directions is not None:
            return self._directions
        from numpy import tile, cross
        if self.curved:
            centers = self.plank_centers()
            dv = cross(centers - self.center, self.axis)
            self._directions = _normalize_vector_array(dv)
        else:
            self._directions = tile(self.axis, (len(self.coords), 1))
        return self._directions

    def plank_normals(self):
        if self._normals is not None:
            return self._normals
        from numpy import cross, tile
        if self.curved:
            in_plane = self.plank_centers() - self.center
            normals = in_plane #  + self.tilt * self.axis
            normals = _normalize_vector_array(normals)
        else:
            bn = cross(self.width_vector, self.axis)
            shape = (len(self.coords), 1)
            normals = tile(self.width_vector, shape), tile(bn, shape)
        directions = self.plank_directions()
        binormals = cross(directions, normals)
        binormals = _normalize_vector_array(binormals)
        self._normals = normals, binormals
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
        self.width_vector, self.thickness_vector = self._straight_tilt()

        # TODO: Compute normals using guide positions



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
        return width_vector, thickness_vector
        # Save results
        self.width_vector = width_vector
        self.thickness_vector = thickness_vector

    def _straight_initial(self):
        from numpy import mean, argsort
        from numpy.linalg import svd
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

    def _try_curved(self):
        from numpy import mean, cross, sum, vdot
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
            # Strand does not curve
            # print("strand straight")
            self._straight_optimize()
        else:
            iwsl2 = 1.0 / (2 * wsl)
            tt = vdot(t, t)
            uu = vdot(u, u)
            c_center = p1 + (u * tt * vdot(u, v) - t * uu * vdot(t, v)) * iwsl2
            c_radius = sqrt(tt * uu * vdot(v, v) * iwsl2 * 0.5)
            c_axis = w / sqrt(wsl)
            # print("strand curved: center", c_center, "radius", c_radius,
            #     "axis", c_axis)
            self._curved_optimize(c_center, c_axis, c_radius)

    def _curved_optimize(self, center, axis, radius):
        opt = OptArc(self.coords, center, axis, radius)
        self.curved = True
        self.center = opt.center
        self.axis = opt.axis
        self.radius = opt.radius
        if False:
            from numpy import cross, dot
            v = cross(self.coords[0] - self.center, self.coords[-1] - self.center)
            print("axis check", v, self.axis, dot(v, self.axis))
        self.tilt = self._curved_tilt()

    def _curved_tilt(self):
        # With center, axis and radius set, we can compute the "x" and "y"
        # coordinates for each guide relative to the point on the circle
        # closest to each guide ("reference").  "x" is the signed distance
        # along the center-reference line; "y" is the signed distance
        # perpendicular to the center-reference line.  We then fit a line
        # through the (x, y) coordinates for all guides and the slope of
        # the line gives us the plank angle relative to the plane of the circle.
        from numpy import dot, outer, concatenate, polyfit, newaxis, stack
        from numpy.linalg import norm, lstsq
        from math import atan
        rel_coords = self.guides - self.center
        # y = tilt magnitudes in direction of arc normal
        y = dot(rel_coords, self.axis)
        # yv = tilt vectors in direction of arc normal
        yv = outer(y, self.axis)
        # xv = tilt vectors perpendicuar to arc normal
        xv = rel_coords - yv
        # x = tilt magnitudes perpendicular to arc normal
        x = norm(xv, axis=1) - self.radius
        # d = length of (x, y)
        d = norm(stack([x, y], axis=1), axis=1)
        x /= d
        y /= d
        # debug stuff
        from numpy import newaxis
        nxv = _normalize_vector_array(xv)
        cxv = nxv * self.radius
        dxv = xv - cxv
        self.tilt_centers = cxv + self.center
        self.tilt_xv = dxv
        self.tilt_yv = yv
        self.tilt_x = x
        self.tilt_y = y
        self.tilt_guides = self.guides[:]
        # intercept, slope = best-fit line through magnitudes
        # slope = ratio of magnitudes parallel/perpendicular to arc normal
        intercept, slope = polyfit(x, y, 1)
        #print("x", list(x))
        #print("y", list(y))
        print("slope", slope)
        m, _, _, _ = lstsq(x[2:-2,newaxis], y[2:-2])
        print("lstsq", m)
        slope = m[0]
        if True:
            global inited
            if not inited:
                import matplotlib
                matplotlib.use("qt5agg")
                inited = True
            import matplotlib.pyplot as plt
            fig = plt.figure()
            plt.axis('equal')
            plt.axis([-1.2, 1.2, -1.2, 1.2])
            plt.plot(x[2:-2], y[2:-2], 'o')
            plt.xlabel("slope: %.4f" % slope)
            plt.show()
        return slope

inited = False

def _normalize_vector_array(v):
    from numpy.linalg import norm
    from numpy import newaxis
    return v / norm(v, axis=1)[:, newaxis]
