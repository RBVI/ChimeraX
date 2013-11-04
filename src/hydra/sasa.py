# Analytic computation of solvent accessible surface area, ie. the surface area of a union of spheres.

def spheres_surface_area(spheres):

    surf = sphere_model(spheres)
    for p in surf.surface_pieces()[1:]:
        p.color = (.7,.7,.7,.5)
        p.display = False

    # Compute intersections with first sphere.
    s0 = spheres[0]
    circles = []
    for s1 in spheres[1:]:
        c = sphere_intersection(s0, s1)
        if not c is None:
            circles.append(c)
    draw_circles(circles, s0, surf, width = 0.01, offset = 0.01)

    # Compute intersections of circles on sphere.
    cint = []
    for i, c0 in enumerate(circles):
        for c1 in circles[i+1:]:
            pts = circle_intercepts(c0, c1)
            if pts:
                cint.append(Circle_Intersection(c0,c1,pts[0]))
                cint.append(Circle_Intersection(c1,c0,pts[1]))
    draw_sphere_points([ci.point for ci in cint], s0, surf, (.8,.2,0,1))

    # Count disjoint regions in union of circles
    reg = {}
#    for ci in cint:
#        ci.circle1

    # Find circles with no intercepts.
    lc = set(circles)
    for ci in cint:
        lc.discard(ci.circle1)
        lc.discard(ci.circle2)

    # Remove circle intersections that are inside another circle.
    bndry = [ci for ci in cint if not in_circles(ci, circles)]
    draw_sphere_points([ci.point for ci in bndry], s0, surf, (.2,.8,0,1), 0.03)

    # Connect arcs to form boundary paths.
    paths = boundary_paths(bndry)
    for bp in paths:
        draw_boundary(bp, s0, surf, color = (.8,.5,.5,1), width = 0.01, offset = 0.02)
    draw_circles(tuple(lc), s0, surf, color = (.8,.5,.5,1), width = 0.01, offset = 0.02)

def draw_boundary(bp, sphere, surf, color, width, offset):

    n = len(bp)
    for i in range(n):
        bp1 = bp[i]
        bp2 = bp[(i+1)%n]
        if not bp1.circle2 is bp2.circle1:
            raise RuntimeError('boundary error')
        draw_arc(bp1.circle2, bp1.point, bp2.point, sphere, surf, color, width, offset)

def draw_arc(circle, p1, p2, sphere, surf, color, width, offset):

    from .geometry.place import orthonormal_frame
    f = orthonormal_frame(circle.center, xdir = p1)
    arc = polar_angle(f.transpose()*p2)
    va, ta = sphere_band_arc(circle.angle, arc, width)
    f.move(va)
    na = va.copy()
    c, r = sphere
    va *= r + offset
    va += c
    p = surf.newPiece()
    p.geometry = va, ta
    p.normals = na
    p.color = color

# Angle about z measured from x-axis, range 0 to 2*pi
def polar_angle(p):
    (x,y,z) = p
    from math import atan2, pi
    a = atan2(y,x)
    if a < 0:
        a += 2*pi
    return a

def boundary_paths(bndry):

    bpaths = []
    bset = set(bndry)
    for bp in bndry:
        if bp in bset:
            bp = boundary_path(bp, bset)
            bpaths.append(bp)
#            for pp in bp:
#                bset.remove(pp)
    return bpaths

def boundary_path(bpoint, bset):

    bpath = [bpoint]
    bp = bpoint
    from .geometry.place import orthonormal_frame
    from math import atan2
    while True:
        f = orthonormal_frame(bp.circle2.center, xdir = bp.point).transpose()
        c1points = [bpn for bpn in bset if bpn.circle1 is bp.circle2]
        def pangle(cp,f=f):
            return polar_angle(f*cp.point)
        bpn = min(c1points, key=pangle)
        if bpn is bpoint:
            break
        bpath.append(bpn)
        bset.remove(bpn)
        bp = bpn
    bset.remove(bpoint)
    return bpath

def in_circles(ci, circles):
    from .geometry.vector import inner_product
    from math import cos
    for c in circles:
        if not c in (ci.circle1, ci.circle2):
            if inner_product(ci.point,c.center) >= cos(c.angle):
                return True
    return False

def draw_sphere_points(points, sphere, s, color, radius = 0.02, offset = 0.02):
    circles = tuple(Circle(p,radius) for p in points)
    draw_circles(circles, sphere, s, offset, 0.5*radius, color)

class Circle_Intersection:
    def __init__(self, circle1, circle2, point):
        self.circle1 = circle1
        self.circle2 = circle2
        self.point = point

def circle_intercepts(c0, c1):

    from math import cos, sqrt
    ca0 = cos(c0.angle)
    ca1 = cos(c1.angle)
    from .geometry.vector import inner_product, cross_product, norm
    ca01 = inner_product(c0.center,c1.center)
    x01 = cross_product(c0.center,c1.center)
    sa01 = norm(x01)
    s2 = sa01*sa01
    if s2 == 0:
        return []
    
    a = (ca0 - ca01*ca1) / s2
    b = (ca1 - ca01*ca0) / s2
    d2 = (sa01*sa01 - ca0*ca0 - ca1*ca1 + 2*ca01*ca0*ca1)
    if d2 < 0:
        return []
    d = sqrt(d2) / s2
    
    return (a*c0.center + b*c1.center - d*x01,
            a*c0.center + b*c1.center + d*x01)

def draw_circles(circles, sphere, s, offset, width, color = (0,.2,.9,1)):
    cs, r = sphere
    from .geometry.place import translation, orthonormal_frame
    t = translation(cs)
    for c in circles:
        f = orthonormal_frame(c.center)
        va, ta = sphere_band_geometry(c.angle, width = width)
        na = va.copy()
        va *= r + offset
        f.move(va)
        f.move(na)
        t.move(va)
        p = s.newPiece()
        p.geometry = va, ta
        p.normals = na
        p.color = color

def sphere_band_geometry(a, step = 0.01, width = 0.01):

    from math import pi, sin, cos
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

    from math import pi, sin, cos
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

def sphere_intersection(s0, s1):

    c0, r0 = s0
    c1, r1 = s1
    from .geometry.vector import distance
    d = distance(c0,c1)
    if d > r0+r1 or r1+d < r0 or r0+d < r1:
        return None
    ca = (r0*r0 + d*d - r1*r1) / (2*r0*d)
    if ca < -1 or ca > 1:
        return None
    import math
    a = math.acos(ca)
    c = (c1-c0)/d
    return Circle(c, a)

# Circle on a unit sphere specified by center point on sphere and angular radius.
class Circle:
    def __init__(self, center, angle):
        self.center = center
        self.angle = angle

def sphere_model(spheres, ntri = 2000):

    from .surface import Surface
    s = Surface('spheres')
    from .molecule.molecule import sphere_geometry
    for center, r in spheres:
        va, na, ta = sphere_geometry(ntri)
        va = va*r + center
        p = s.newPiece()
        p.geometry = va, ta
        p.normals = na
    from .ui.gui import main_window
    main_window.view.add_model(s)

    return s

def random_spheres(n):

    from numpy import array, float64
    c0 = array((0,0,0),float64)
    spheres = [(c0, 1)]
    from random import uniform, gauss
    from .geometry.vector import distance
    while len(spheres) < n:
        r = uniform(0.5,1)
        center = array((gauss(0,1), gauss(0,1), gauss(0,1)), float64)
        d = distance(center, c0)
        if d > r+1 or d+r < 1 or r-d > 1:
            continue    # No intercept
        spheres.append((center, r))
    return spheres

def test_sasa(n = 5):
    spheres = random_spheres(n)
    spheres_surface_area(spheres)
