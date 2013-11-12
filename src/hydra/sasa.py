# Analytic computation of solvent accessible surface area, ie. the surface area of a union of spheres.

from math import pi, cos, sin, sqrt, atan2, acos

def surface_area_of_spheres(centers, radii):

    areas = radii.copy()
    for i,r in enumerate(radii):
        ba = buried_sphere_area(i, centers, radii)
        areas[i] = 4*pi*r*r - ba
    return areas

def buried_sphere_area(i, centers, radii, draw = False):

    if draw:
        surf0 = sphere_model([i], centers, radii)
        jlist = list(range(i)) + list(range(i+1,len(centers)))
        surfn = sphere_model(jlist, centers, radii)
        for p in surfn.surface_pieces():
            p.color = (.7,.7,.9,.7)
        from .ui.gui import main_window
        main_window.view.add_models((surf0, surfn))

    r = radii[i]

    # Check if sphere is completely contained in another sphere
    if sphere_in_another_sphere(i, centers, radii):
        area = 4*pi*r*r
        return area

    # Compute sphere intersections
    circles = sphere_intersection_circles(i, centers, radii)

    # Compute analytical buried area on sphere.
    area = area_in_circles_on_unit_sphere(circles, draw)*r*r

    return area

def sphere_in_another_sphere(i, centers, radii):

    c, r = centers[i], radii[i]
    d = centers - c
    from numpy import sqrt
    inside = (sqrt((d*d).sum(axis=1)) + r <= radii)
    return inside[:i].any() or inside[i+1:].any()

def sphere_intersection_circles(i, centers, radii):

    circles = []
    c,r = centers[i], radii[i]
    for j,rj in enumerate(radii):
        if j != i:
            circle = sphere_intersection(c, r, centers[j], rj)
            if not circle is None:
                circles.append(circle)
    return circles

def sphere_intersection(c0, r0, c1, r1):

    from .geometry.vector import distance
    d = distance(c0,c1)
    if d > r0+r1 or r1+d < r0 or r0+d < r1:
        return None
    ca = (r0*r0 + d*d - r1*r1) / (2*r0*d)
    if ca < -1 or ca > 1:
        return None
    c = (c1-c0)/d
    return Circle(c, ca)

# Circle on a unit sphere specified by center point on sphere and angular radius.
class Circle:
    def __init__(self, center, cos_angle):
        self.center = center
        self.angle = acos(cos_angle)
        self.cos_angle = cos_angle

def area_in_circles_on_unit_sphere(circles, draw = False):

    # Check if sphere is outside all other spheres.
    if len(circles) == 0:
        return 0

    if draw:
        from .surface import Surface
        surfc = Surface('circles')
        s0 = unit_sphere()
        draw_circles(circles, s0, surfc, width = 0.01, offset = 0.01)

    cint, lc, nreg = circle_intersections(circles)

    if draw:
        surfi = Surface('boundary points')
        draw_sphere_points([ci.point for ci in cint], s0, surfi, (.8,.2,0,1), offset = 0.02)

    # Check if circles cover the sphere
    if len(cint) == 0 and len(lc) == 0:
        return 4*pi

    # Connect circle arcs to form boundary paths.
    paths = boundary_paths(cint)

#    print('boundary lengths', ','.join(str(nr) for nr in [1]*len(lc) + [len(p) for p in paths]),
#          (('for %d regions' % (nreg + len(lc))) if nreg < len(paths) else ''))

    if draw:
        surfb = Surface('boundary')
        for bp in paths:
            draw_boundary(bp, s0, surfb, color = (.8,.5,.5,1), width = 0.01, offset = 0.02)
        draw_circles(tuple(lc), s0, surfb, color = (.8,.5,.5,1), width = 0.01, offset = 0.02)

    la = lone_circles_area(lc)
    ba = bounded_area(paths, nreg)
    area = la + ba

    if draw:
        from .ui.gui import main_window
        main_window.view.add_models((surfc, surfi, surfbp, surfb))

    return area

def circle_intersections(circles):

    # Remove circles contained in other circles.
    circles2 = [c for i,c in enumerate(circles) if not circle_in_circles(i, circles)]

    # Compute intersection points of circles that are not contained in other circles.
    cint = []
    rc = Region_Count(len(circles2))
    for i, c0 in enumerate(circles2):
        for j, c1 in enumerate(circles2[i+1:], i+1):
            p0,p1 = circle_intercepts(c0, c1)
            if not p0 is None:
                rc.join(i,j)
                if not point_in_circles(p0, circles2, (i,j)):
                    cint.append(Circle_Intersection(c0,c1,p0))
                if not point_in_circles(p1, circles2, (i,j)):
                    cint.append(Circle_Intersection(c1,c0,p1))
    sz = rc.region_sizes()
    lc = [circles2[i] for i,s in enumerate(sz) if s == 1]       # Lone circles
    nreg = rc.number_of_regions() - len(lc)     # Number of multicircle regions

    return cint, lc, nreg

def circle_in_circles(i, circles):
    from .geometry.vector import inner_product
    p,a = circles[i].center, circles[i].angle
    for j,c in enumerate(circles):
        if c.angle >= a and inner_product(p, c.center) >= cos(c.angle-a) and j != i:
            return True
    return False

def circle_intercepts(c0, c1):

    from .geometry.vector import inner_product, cross_product
    ca01 = inner_product(c0.center,c1.center)
    x01 = cross_product(c0.center,c1.center)
    s2 = inner_product(x01,x01)
    if s2 == 0:
        return None,None
    
    ca0 = c0.cos_angle
    ca1 = c1.cos_angle
    a = (ca0 - ca01*ca1) / s2
    b = (ca1 - ca01*ca0) / s2
    d2 = (s2 - ca0*ca0 - ca1*ca1 + 2*ca01*ca0*ca1)
    if d2 < 0:
        return None,None
    d = sqrt(d2) / s2
    
    return (a*c0.center + b*c1.center - d*x01,
            a*c0.center + b*c1.center + d*x01)

def point_in_circles(p, circles, exclude):
    from .geometry.vector import inner_product
    for i,c in enumerate(circles):
        if inner_product(p,c.center) >= c.cos_angle and not i in exclude:
            return True
    return False

class Circle_Intersection:
    def __init__(self, circle1, circle2, point):
        self.circle1 = circle1
        self.circle2 = circle2
        self.point = point

class Region_Count:
    def __init__(self, n):
        from numpy import arange
        self.c = arange(n)
    def join(self, i, j):
        mi = self.min_connected(i)
        mj = self.min_connected(j)
        if mi < mj:
            self.c[mj] = mi
        elif mj < mi:
            self.c[mi] = mj
    def min_connected(self, i):
        c = self.c
        while c[i] != i:
            i = c[i]
        return i
    def number_of_regions(self):
        nr = 0
        for i,ci in enumerate(self.c):
            if ci == i:
                nr += 1
        return nr
    def region_sizes(self):
        n = len(self.c)
        from numpy import zeros
        s = zeros((n,), self.c.dtype)
        for i in range(n):
            s[self.min_connected(i)] += 1
        return s

def boundary_paths(cint):

    ciset = set(cint)
    bpaths = [boundary_path(ci, ciset) for ci in cint if ci in ciset]
    return bpaths

def boundary_path(bpoint, bset):

    bpath = [bpoint]
    bp = bpoint
    while True:
        c1points = [bpn for bpn in bset if bpn.circle1 is bp.circle2]
        if len(c1points) == 0:
            raise RuntimeError('Could not follow boundary. Probably due to 3 or more circles intersecting at one point.')
        bpn = min(c1points, key=lambda cp: polar_angle(bp.circle2.center, bp.point, cp.point))
        if bpn is bpoint:
            break
        bpath.append(bpn)
        bset.remove(bpn)
        bp = bpn
    bset.remove(bpoint)
    return bpath

def lone_circles_area(lone_circles):

    area = sum((2*pi*(1-c.cos_angle) for c in lone_circles), 0)
    return area

def bounded_area(paths, nreg):

    area = 0
    for path in paths:
        n = len(path)
        ba = 0
        for i in range(n):
            bp1 = path[i]
            p, c1, c2 = bp1.point, bp1.circle1, bp1.circle2
            ia = circle_intercept_angle(c1.center, p, c2.center)
            ba += ia - 2*pi
            bp2 = path[(i+1)%n]
            a = polar_angle(c2.center, p, bp2.point)  # Circular arc angle
#            print('seg', i, 'kink', (ia - 2*pi)*180/pi, 'arc', a*180/pi)
            ba += a * bp1.circle2.cos_angle  # circular segment bend angle
#        print('path length', n, 'area', 2*pi-ba)
        area += 2*pi - ba
    if len(paths) > nreg:
        area -= 4*pi*(len(paths)-nreg)
    return area

def circle_intercept_angle(center1, pintersect, center2):
    # Angle made by tangent vectors t1 = c1 x p and t2 = c2 x p is same as
    # polar angle of c1 and c2 about p.
    a = polar_angle(pintersect, center1, center2)
    return a

# Angle from plane defined by z and v1 rotated to plane defined by z and v2
# range 0 to 2*pi.
def polar_angle(zaxis, v1, v2):
    from .geometry.place import orthonormal_frame
    f = orthonormal_frame(zaxis, xdir = v1)
    x,y,z = f.transpose()*v2
    a = atan2(y,x)
    if a < 0:
        a += 2*pi
    return a

def estimate_surface_area_of_spheres(centers, radii, sphere_points, point_weights):
    areas = radii.copy()
    for i,r in enumerate(radii):
        ba = estimate_buried_sphere_area(i, centers, radii, sphere_points, point_weights)
        areas[i] = 4*pi*r*r - ba
    return areas

def estimate_buried_sphere_area(i, centers, radii, points, weights):
    c,r = centers[i], radii[i]
    points = points.copy()
    points *= r
    points += c
    from numpy import zeros, int32, logical_or
    inside = zeros((len(points),), int32)
    for j, radius in enumerate(radii):
        if j != i:
            d = points - centers[j]
            d2 = (d*d).sum(axis = 1)
            logical_or(inside, d2 <= radius*radius, inside)
    a = 4*pi*r*r*sum(inside*weights)/weights.sum()
    return a

def sphere_points_and_weights(npoints):
    from .molecule.molecule import sphere_geometry
    va, na, ta = sphere_geometry(2*npoints)
    # Weight vertices by area since distribution is not uniform.
    from ._image3d import vertex_areas
    weights = vertex_areas(va, ta)
    return va, weights

def draw_boundary(bp, sphere, surf, color, width, offset):

    n = len(bp)
    for i in range(n):
        bp1 = bp[i]
        bp2 = bp[(i+1)%n]
        draw_arc(bp1.circle2, bp1.point, bp2.point, sphere, surf, color, width, offset)

def draw_arc(circle, p1, p2, sphere, surf, color, width, offset):

    arc = polar_angle(circle.center, p1, p2)
    va, ta = sphere_band_arc(circle.angle, arc, width)
    from .geometry.place import orthonormal_frame
    f = orthonormal_frame(circle.center, xdir = p1)
    f.move(va)
    na = va.copy()
    c, r = sphere
    va *= r + offset
    va += c
    p = surf.newPiece()
    p.geometry = va, ta
    p.normals = na
    p.color = color

def draw_sphere_points(points, sphere, s, color, radius = 0.02, offset = 0.02):
    circles = tuple(Circle(p,radius) for p in points)
    draw_circles(circles, sphere, s, offset, 0.5*radius, color)

def draw_circles(circles, sphere, s, offset, width, color = (0,.2,.9,1)):
    cs, r = sphere
    from .geometry.place import orthonormal_frame
    for c in circles:
        f = orthonormal_frame(c.center)
        va, ta = sphere_band_geometry(c.angle, width = width)
        na = va.copy()
        va *= r + offset
        f.move(va)
        f.move(na)
        va += cs
        p = s.newPiece()
        p.geometry = va, ta
        p.normals = na
        p.color = color

def sphere_band_geometry(a, step = 0.01, width = 0.01):

    n = max(3,int(2*pi*sin(a)/step))
    from numpy import empty, float32, int32, arange
    va = empty((2*n,3), float32)
    cp = circle_points(n)
    va[:n,:] = cp
    va[:n,:] *= sin(a-0.5*width)
    va[:n,2] = cos(a-0.5*width)
    va[n:,:] = cp
    va[n:,:] *= sin(a+0.5*width)
    va[n:,2] = cos(a+0.5*width)
    ta = empty((2*n,3), int32)
    n0 = arange(n)
    n1 = (n0+1) % n
    ta[:n,0] = n0
    ta[:n,1] = n0 + n
    ta[:n,2] = n1 + n
    ta[n:,0] = n0
    ta[n:,1] = n1 + n
    ta[n:,2] = n1
    return va, ta

def sphere_band_arc(aradius, arc, width = 0.01, step = 0.01):

    a = aradius
    n = max(2,int(arc*sin(a)/step))
    from numpy import empty, float32, int32, arange
    va = empty((2*n,3), float32)
    ap = arc_points(arc, n)
    va[:n,:] = ap
    va[:n,:] *= sin(a-0.5*width)
    va[:n,2] = cos(a-0.5*width)
    va[n:,:] = ap
    va[n:,:] *= sin(a+0.5*width)
    va[n:,2] = cos(a+0.5*width)
    ta = empty((2*(n-1),3), int32)
    n0 = arange(n-1)
    n1 = n0+1
    ta[:n-1,0] = n0
    ta[:n-1,1] = n0 + n
    ta[:n-1,2] = n1 + n
    ta[n-1:,0] = n0
    ta[n-1:,1] = n1 + n
    ta[n-1:,2] = n1
    return va, ta

def circle_points(n):

    from numpy import arange, pi, cos, sin, zeros, transpose, float32
    a = arange(n) * (2*pi / n)
    p = transpose((cos(a), sin(a), zeros(n))).astype(float32)
    return p

def arc_points(arc, n):

    from numpy import arange, pi, cos, sin, zeros, transpose, float32
    a = arange(n) * (arc / (n-1))
    p = transpose((cos(a), sin(a), zeros(n))).astype(float32)
    return p

def sphere_model(indices, centers, radii, ntri = 2000):

    from .surface import Surface
    s = Surface('spheres')
    from .molecule.molecule import sphere_geometry
    for i in indices:
        va, na, ta = sphere_geometry(ntri)
        va = va*radii[i] + centers[i]
        p = s.newPiece()
        p.geometry = va, ta
        p.normals = na

    return s

def random_spheres_intersecting_unit_sphere(n):

    from numpy import random, sqrt
    radii = random.uniform(0.25, .5, n)
    centers = random.normal(size = (n,3))

    # Move spheres radially so the intersect unit sphere.
    d = sqrt((centers*centers).sum(axis=1))
    s = ((2*random.random(n)-1)*radii + 1)/d
    for a in (0,1,2):
        centers[:,a] *= s

    centers[0,:] = (0,0,0)
    radii[0] = 1
    return centers, radii

def unit_sphere():
    from numpy import array, float64
    center = array((0,0,0),float64)
    radius = 1
    return (center, radius)

def molecule_spheres(probe_radius = 1.4):
    from .ui.gui import main_window
    mlist = main_window.view.molecules()
    from .molecule import Atom_Set
    aset = Atom_Set()
    aset.add_molecules(mlist)
    aset = aset.exclude_water()
    centers = aset.coordinates()
    radii = aset.radii() + probe_radius
    return centers, radii

def test_sasa(n = 30):
#    centers, radii = random_spheres_intersecting_unit_sphere(n)
#    from numpy import array
#    centers, radii = array(((0.0,0,0), (1.0,0,0))), array((1.0, 1.0))     # area test, pi
#    centers, radii = array(((0.0,0,0), (1.0,0,0), (0,1.0,0))), array((1.0, 1.0, 1.0)) # area test
#    centers, radii = array(((0.0,0,0), (1.0,0,0), (1.0,0,0))), array((1.0, 0.5, 0.25))  # Nested circle test
#    r = sqrt(2)
#    centers, radii = array(((0.0,0,0), (1.0,0,0))), array((1.0, r))     # area test, 2*pi
#    centers, radii = array(((0.0,0,0), (1.0,0,0), (0,1.0,0))), array((1.0, r, r))     # area test, 3*pi

#    buried_sphere_area(0, centers, radii, draw = True)

    centers, radii = molecule_spheres()

#    import cProfile
#    cProfile.runctx('print("area =", spheres_surface_area(centers, radii).sum())', globals(), locals())
    from ._image3d import surface_area_of_spheres, estimate_surface_area_of_spheres
    from time import time
    t0 = time()
    areas = surface_area_of_spheres(centers, radii)
    t1 = time()
    points, weights = sphere_points_and_weights(npoints = 1000)
    eareas = estimate_surface_area_of_spheres(centers, radii, points, weights)
    t2 = time()
    print(len(centers), 'atoms, area', areas.sum(), 'estimate', eareas.sum(),
          '(%d points)' % len(points), 'times %.3f %.3f' % (t1-t0, t2-t1))
    from numpy import absolute
    aerr = absolute(areas - eareas) / (4*pi*radii*radii)
    print('est error max %.05f mean %.05f' % (aerr.max(), aerr.mean()))

# Example results.
# Testing on PDB 1a0m excluding waters, 242 atoms. 15 seconds for analytic area.  Most time culling circle intersections.
# Average of 36 circles per sphere, ~660 circle intersections per sphere, ~8 intersections on boundary.
# Error with 10000 sphere point estimate, max over each sphere 0.002, mean 0.0004 as fraction of full sphere area, time 5 seconds.
