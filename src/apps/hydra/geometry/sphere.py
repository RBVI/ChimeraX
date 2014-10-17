def sphere_points(n):
    '''
    Return N uniformly distributed points on a sphere of radius 1.  Uses spiral points algorithm from
    Robert Bauer.
    "Distribution of Points on a Sphere with Application to Star Catalogs"
    Journal of Guidance, Control, and Dynamics,
    Vol. 23, No. 1 (2000), pp. 130-137.
    '''
    from numpy import empty, float32
    p = empty((n,3), float32)
    from math import acos, sqrt, cos, sin, pi
    for i in range(n):
        phi = acos(-1.0 + float(2*i+1)/n)
        theta = sqrt(n*pi) * phi
        s = sin(phi)
        p[i,:] = (s*cos(theta), s*sin(theta), cos(phi))
    return p

def test(n = 1000, color = (.7,.7,.7,1), radius = 1):
    from VolumePath import Marker_Set, Marker
    m = Marker_Set('sphere points')
    p = sphere_points(n)
    for xyz in p:
        m.place_marker(20*xyz, color, radius)
