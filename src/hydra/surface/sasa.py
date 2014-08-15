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
        from ..geometry.vector import norm
        jlist = [j for j in range(len(centers)) if j != i and norm(centers[j]-centers[i]) < radii[j]+radii[i]]
#        jlist = list(range(i)) + list(range(i+1,len(centers)))
        print(len(jlist), 'spheres intersect sphere', i)
        surfn = sphere_model(jlist, centers, radii)
        for p in surfn.child_drawings():
            p.color = (.7,.7,.9,.7)
        draw.add_models((surf0, surfn))

    r = radii[i]

    # Check if sphere is completely contained in another sphere
    if sphere_in_another_sphere(i, centers, radii):
        area = 4*pi*r*r
        return area

    # Compute sphere intersections
    circles = sphere_intersection_circles(i, centers, radii)

    # Compute analytical buried area on sphere.
    area = area_in_circles_on_unit_sphere(circles, draw, centers[i], r)*r*r

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

    from ..geometry.vector import distance
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

def area_in_circles_on_unit_sphere(circles, draw = False, draw_center = (0,0,0), draw_radius = 1):

    # Check if sphere is outside all other spheres.
    if len(circles) == 0:
        return 0

    if draw:
        from ..graphics import Drawing
        surfc = Drawing('circles')
        s0 = (draw_center, draw_radius)
        draw_circles(circles, s0, surfc, width = 0.01, offset = 0.01)

    cint, lc, nreg = circle_intersections(circles, draw)

    if draw:
        surfi = Drawing('boundary points')
        draw_sphere_points([ci.point for ci in cint], s0, surfi, (.8,.2,0,1), offset = 0.02)

    # Check if circles cover the sphere
    if len(cint) == 0 and len(lc) == 0:
        return 4*pi

    # Connect circle arcs to form boundary paths.
    paths = boundary_paths(cint)

#    print('boundary lengths', ','.join(str(nr) for nr in [1]*len(lc) + [len(p) for p in paths]),
#          (('for %d regions' % (nreg + len(lc))) if nreg < len(paths) else ''))

    if draw:
        surfb = Drawing('boundary')
        for bp in paths:
            draw_boundary(bp, s0, surfb, color = (.8,.5,.5,1), width = 0.01, offset = 0.02)
        draw_circles(tuple(lc), s0, surfb, color = (.8,.5,.5,1), width = 0.01, offset = 0.02)

    la = lone_circles_area(lc)
    ba = bounded_area(paths, nreg)
    area = la + ba

    if draw:
        draw.add_models((surfc, surfi, surfb))

    return area

def circle_intersections(circles, draw = False):

    # Remove circles contained in other circles.
    circles2 = [c for i,c in enumerate(circles) if not circle_in_circles(i, circles)]
    if draw:
        print(len(circles), 'circles', len(circles)-len(circles2), 'inside other circles')

    # Compute intersection points of circles that are not contained in other circles.
    cint = []
    rc = Region_Count(len(circles2))
    ni = 0
    for i, c0 in enumerate(circles2):
        for j, c1 in enumerate(circles2[i+1:], i+1):
            p0,p1 = circle_intercepts(c0, c1)
            if not p0 is None:
                rc.join(i,j)
                if not point_in_circles(p0, circles2, (i,j)):
                    cint.append(Circle_Intersection(c0,c1,p0))
                if not point_in_circles(p1, circles2, (i,j)):
                    cint.append(Circle_Intersection(c1,c0,p1))
                ni += 2
    sz = rc.region_sizes()
    lc = [circles2[i] for i,s in enumerate(sz) if s == 1]       # Lone circles
    nreg = rc.number_of_regions() - len(lc)     # Number of multicircle regions
    if draw:
        print(ni, 'intersection points', len(cint), 'on boundary', len(lc), 'lone circles', nreg, 'multicircle regions')

    return cint, lc, nreg

def circle_in_circles(i, circles):
    from ..geometry.vector import inner_product
    p,a = circles[i].center, circles[i].angle
    for j,c in enumerate(circles):
        if c.angle >= a and inner_product(p, c.center) >= cos(c.angle-a) and j != i:
            return True
    return False

def circle_intercepts(c0, c1):

    from ..geometry.vector import inner_product, cross_product
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
    from ..geometry.vector import inner_product
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
    from ..geometry.place import orthonormal_frame
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
    from .shapes import sphere_geometry
    va, na, ta = sphere_geometry(2*npoints)
    # Weight vertices by area since distribution is not uniform.
    from ..map_cpp import vertex_areas
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
    from ..geometry.place import orthonormal_frame
    f = orthonormal_frame(circle.center, xdir = p1)
    f.move(va)
    na = va.copy()
    c, r = sphere
    va *= r + offset
    va += c
    p = surf.new_drawing()
    p.geometry = va, ta
    p.normals = na
    p.color = color

def draw_sphere_points(points, sphere, s, color, radius = 0.02, offset = 0.02):
    circles = tuple(Circle(p,cos(radius)) for p in points)
    draw_circles(circles, sphere, s, offset, 0.5*radius, color)

def draw_circles(circles, sphere, s, offset, width, color = (0,.2,.9,1)):
    cs, r = sphere
    from ..geometry.place import orthonormal_frame
    for c in circles:
        f = orthonormal_frame(c.center)
        va, ta = sphere_band_geometry(c.angle, width = width)
        na = va.copy()
        va *= r + offset
        f.move(va)
        f.move(na)
        va += cs
        p = s.new_drawing()
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

    from ..graphics import Drawing
    s = Drawing('spheres')
    from .shapes import sphere_geometry
    for i in indices:
        va, na, ta = sphere_geometry(ntri)
        va = va*radii[i] + centers[i]
        p = s.new_drawing()
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

def molecule_spheres(mlist = None, probe_radius = 1.4):
    atoms = molecule_atoms(mlist)
    centers = atoms.coordinates()
    radii = atoms.radii() + probe_radius
    return centers, radii

def molecule_atoms(mlist):
    from ..molecule import Atoms
    aset = Atoms()
    aset.add_molecules(mlist)
    aset = aset.exclude_water()
    return aset

def accessible_surface_area(molecule):
    centers, radii = molecule_spheres([molecule])
    return spheres_surface_area(centers, radii)

def spheres_surface_area(centers, radii, npoints = 1000):
    '''
    Return the exposed surface area of a set of possibly intersecting set of spheres.
    An array giving the exposed area for each input sphere is returned.
    The area is computed by an exact method except for round-off errors.
    The calculation can fail in rare cases where 4 spheres intersect at a point
    and area values of -1 are returned for spheres where the calculation fails.
    TODO: The code also computes the areas using numerical approximation, and the
    results compared with the maximum and average discrepancy printed for debugging.
    This code is not needed except for debugging.
    '''
    from .. import molecule_cpp
    areas = molecule_cpp.surface_area_of_spheres(centers, radii)
    return areas

def report_sphere_area_errors(areas, centers, radii, npoints = 1000, max_err = 0.02):
    points, weights = sphere_points_and_weights(npoints)
    from ..molecule_cpp import estimate_surface_area_of_spheres
    eareas = estimate_surface_area_of_spheres(centers, radii, points, weights)
    nf = (areas == -1).sum()
    if nf > 0:
        print('%d atoms, area calc failed for %d atoms, estimate %.1f (%d points)\n' %
              (len(centers), nf, eareas.sum(), len(points)))
        print('Failed calc for', str((areas == -1).nonzero()[0]))
    else:
        from numpy import absolute
        aerr = absolute(areas - eareas) / (4*pi*radii*radii)
        print('%d atoms, area %.1f, estimate %.1f (%d points)\nest error max %.05f mean %.05f' %
              (len(centers), areas.sum(), eareas.sum(), len(points), aerr.max(), aerr.mean()))
        if aerr.max() >= max_err:
            import numpy
            ei = numpy.argsort(aerr)
            for i in ei[::-1]:
                if aerr[i] >= max_err:
                    print (i, areas[i], eareas[i])

def test_sasa(n = 30, npoints = 1000):
#    test_pdb_models(pdbs, npoints)
#    return
#    test_all_pdb_models(['/Users/goddard/Downloads/Chimera/PDB'])
#    test_all_pdb_models(pdb_subdirs(), '.ent', npoints = npoints)
#    return

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

#    i = 10
#    buried_sphere_area(i, centers, radii, draw = True)
    spheres_surface_area(centers, radii)

# Example results.
# Testing on PDB 1a0m excluding waters, 242 atoms. 15 seconds for analytic area.  Most time culling circle intersections.
# Average of 36 circles per sphere, ~660 circle intersections per sphere, ~8 intersections on boundary.
# Error with 10000 sphere point estimate, max over each sphere 0.002, mean 0.0004 as fraction of full sphere area, time 5 seconds.

def pdb_subdirs(pdb_dir = '/usr/local/pdb'):
    from os import listdir
    from os.path import join
    subdirs = [join(pdb_dir,sd) for sd in listdir(pdb_dir) if len(sd) == 2]      # two letter subdirectories
    return subdirs

def test_all_pdb_models(pdb_dirs, pdb_suffix = '.pdb',
                        results = '/Users/goddard/ucsf/chimera2/src/hydra/sasa_results.txt',
                        npoints = 1000):

    from os import listdir
    from os.path import join
    from ..file_io import opensave
    from ..molecule_cpp import surface_area_of_spheres, estimate_surface_area_of_spheres

    points, weights = sphere_points_and_weights(npoints)

    rf = open(results, 'a')
    rf.write('Area numerical estimate using %d sample points per sphere\n' % len(points))
    rf.write('%6s %6s %9s %9s %5s %5s %8s %8s\n' %
             ('PDB', 'atoms', 'area', 'earea', 'atime', 'etime', 'max err', 'mean err'))
    for pdb_dir in pdb_dirs:
        pdb_names = [p for p in listdir(pdb_dir) if p.endswith(pdb_suffix)]
        for p in pdb_names:
            opensave.open_files([join(pdb_dir,p)])
            centers, radii = molecule_spheres()
            from time import time
            t0 = time()
            areas = surface_area_of_spheres(centers, radii)
            t1 = time()
            eareas = estimate_surface_area_of_spheres(centers, radii, points, weights)
            t2 = time()
            nf = (areas == -1).sum()
            pdb_id = p[3:-4]
            if nf > 0:
                rf.write('%6s %6d %4d-fail %9.1f %5.2f %5.2f\n' %
                         (pdb_id, len(centers), nf, eareas.sum(), t1-t0, t2-t1))
            else:
                from numpy import absolute
                aerr = absolute(areas - eareas) / (4*pi*radii*radii)
                rf.write('%6s %6d %9.1f %9.1f %5.2f %5.2f %8.5f %8.5f\n' %
                         (pdb_id, len(centers), areas.sum(), eareas.sum(), t1-t0, t2-t1, aerr.max(), aerr.mean()))
            rf.flush()
            opensave.close_models()
    rf.close()

pdbs = ('3znu','2hq3','3zqy','2vb1','2yab','3ze1','3ze2','4baj','3ztp','1jxw','1jxu','3ziy','4bs0','4bpu','1jxt','3zcc','4bza','1jxy','1cod','2c04','1jxx','1utn','2xfg','4a2s','1cbn','2ynw','2wur','1alz','4bc5','1f5e','2i2h','2i2j','1olr','2xhn','2yoi','2izq','1hhu','2c03','4b5o','2x5n','2jc5','2ww7','2xjp','4b5v','145d','1hll','4hp2','3bxq','1bv8','1krw','4bf7','2fyl','2xy8','4i9y','1f3c','1j4o','1gbn','1hj8','2ypc','4alf','2yj8','2ynx','4ba7','2yiv','2j45','3ze6','2xu3','2v9l','3zuc','1s1h','2x7k','3zdj','2ynv','4av5','4bag','1jfp','1e3d','4ajx','4c4p','1mli','3fsp','3zbz','2glw','1iyw','2r6p','1tge','1mz0','1myz','3dll','2dfk','2wse','4axi','4l1p','1y1y','4e7u','2d86','2tci','2jfb','3tnq','3lrt',)


def test_pdb_models(id_codes, npoints, session):

    from ..file_io import opensave
    from ..molecule_cpp import surface_area_of_spheres, estimate_surface_area_of_spheres

    points, weights = sphere_points_and_weights(npoints)

    print('Area numerical estimate using %d sample points per sphere\n' % len(points))
    print('%6s %6s %9s %9s %5s %5s %8s %8s\n' %
             ('PDB', 'atoms', 'area', 'earea', 'atime', 'etime', 'max err', 'mean err'))
    for id in id_codes:
            from ..molecule.fetch_pdb import fetch_pdb
            mlist = fetch_pdb(id)
            session.add_models(mlist)
            
            centers, radii = molecule_spheres()
            from time import time
            t0 = time()
            areas = surface_area_of_spheres(centers, radii)
            t1 = time()
            eareas = estimate_surface_area_of_spheres(centers, radii, points, weights)
            t2 = time()
            nf = (areas == -1).sum()
            if nf > 0:
                print('%6s %6d %4d-fail %9.1f %5.2f %5.2f\n' %
                         (id, len(centers), nf, eareas.sum(), t1-t0, t2-t1))
            else:
                from numpy import absolute
                aerr = absolute(areas - eareas) / (4*pi*radii*radii)
                print('%6s %6d %9.1f %9.1f %5.2f %5.2f %8.5f %8.5f\n' %
                         (id, len(centers), areas.sum(), eareas.sum(), t1-t0, t2-t1, aerr.max(), aerr.mean()))
            opensave.close_models()

def area_command(cmdname, args, session):

    from ..commands.parse import atoms_arg, float_arg, parse_arguments
    req_args = (('atoms', atoms_arg),)
    opt_args = ()
    kw_args = (('probeRadius', float_arg),)

    kw = parse_arguments(cmdname, args, session, req_args, opt_args, kw_args)
    kw['session'] = session
    sasa(**kw)

def sasa(atoms, session, probeRadius = 1.4):
    centers = atoms.coordinates()
    radii = atoms.radii() + probeRadius
    areas = spheres_surface_area(centers, radii)
    area = areas.sum()
    failed = (areas == -1).sum()
    if failed > 0:
        area = areas[areas >= 0].sum()
    msg = 'Solvent accessible area %.5g, for %d atoms' % (area, atoms.count())
    mols = list(atoms.molecules())
    mols.sort(key = lambda m: m.id)
    mids = '#' + ','.join('%d' % m.id for m in mols)
    msg += ', model%s %s' % (('s' if len(mols) > 1 else ''), mids)
    if failed > 0:
        msg += ', calculation failed for %d atoms' % failed
    session.show_status(msg)
    session.show_info(msg)
