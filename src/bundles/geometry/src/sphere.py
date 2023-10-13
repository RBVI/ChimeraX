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

def sphere_points(n):
    '''
    Return N uniformly distributed points on a sphere of radius 1.
    Uses spiral points algorithm from Robert Bauer.  "Distribution of
    Points on a Sphere with Application to Star Catalogs" Journal of
    Guidance, Control, and Dynamics, Vol. 23, No. 1 (2000), pp. 130-137.
    '''
    from numpy import empty, float32
    p = empty((n, 3), float32)
    from math import acos, sqrt, cos, sin, pi
    snp = sqrt(n * pi)
    for i in range(n):
        phi = acos(-1.0 + float(2 * i + 1) / n)
        theta = snp * phi
        s = sin(phi)
        p[i, :] = (s * cos(theta), s * sin(theta), cos(phi))
    return p

def sphere_triangulation(ntri):
    '''
    Create a sphere triangulation having exactly the specified number
    of triangles which must be even and at least 4.  The vertices will
    be uniformly spaced on the sphere.
    '''
    ntri = max(4,ntri)	# Require at least 4 triangles
    if ntri % 2:
        ntri += 1	# Require even number of triangles.
    nv = 2 + ntri//2
    va = sphere_points(nv)
    from numpy import empty, int32
    ta = empty((ntri,3), int32)

    # Stitch spiral to form triangles.
    ta[0,:] = (0,1,2)
    ta[1,:] = (0,2,3)
    t = 2
    v1,v2 = 0,3
    while t < ntri:
        adv1 = (v2+1 == nv)
        if not adv1:
            d1 = va[v1+1] - va[v2]
            d2 = va[v2+1] - va[v1]
            adv1 = ((d1*d1).sum() < (d2*d2).sum())
        if adv1:
            ta[t,:] = (v1,v2,v1+1)
            v1 += 1
        else:
            ta[t,:] = (v1,v2,v2+1)
            v2 += 1
        t += 1

    return va, ta
    
def test(n=1000, color=(.7, .7, .7, 1), radius=1):
    from VolumePath import Marker_Set
    m = Marker_Set('sphere points')
    p = sphere_points(n)
    for xyz in p:
        m.place_marker(20 * xyz, color, radius)
