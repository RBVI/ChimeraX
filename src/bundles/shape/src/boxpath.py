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

#
# Create a square cross-section piecewise linear tube for protein backbone
# sculptures of the type made by Daniel Gurnon.
#
def box_path(points, width, twist = 0):

    from numpy import dot, concatenate, array

    # Find normal to bisecting plane through each point.
    n = len(points)
    p = array(points)
    from chimerax.geometry import normalize_vector as nv
    tangents = array([nv(array(nv(p[min(i+1,n-1)]-p[i]))
                         + array(nv(p[i]-p[max(i-1,0)]))) for i in range(n)])

    # Trace edge of square cross-section from start to finish.
    edges = []
    y, z = p[2]-p[0], tangents[0]
    from chimerax.geometry import rotation, orthonormal_frame
    if twist != 0:
        y = rotation(z, twist) * y
    f = orthonormal_frame(z, y)
    xa, ya = f.axes()[:2]
    corners = ((1,1), (-1,1), (-1,-1), (1,-1))
    for x, y in corners:
        ep = [points[0] + (x*0.5*width)*xa + (y*0.5*width)*ya]
        for i in range(n-1):
            e0, p0, p1, t = ep[i], p[i], p[i+1], tangents[i+1]
            ep.append(e0 + (dot(p1-e0, t)/dot(p1-p0, t))*(p1-p0))
        edges.append(ep)

    # Calculate triangles for each face of a surface model.
    # Make sharp edges.
    va = concatenate(edges + edges + edges + edges)
    ta = []
    nc = len(corners)
    for s in range(n-1):
        for c in range(nc):
            c1 = (c+1)%nc + nc
            t = s + (s % 2)*2*nc*n
            ta.append((c*n+t,c1*n+1+t,c*n+1+t))
            ta.append((c*n+t,c1*n+t,c1*n+1+t))

    # Add end caps.
    ta.extend([(nc*n+0,nc*n+(2+c)*n,nc*n+(1+c)*n) for c in range(nc-2)])
    ta.extend([(n-1,(1+c)*n+n-1,(2+c)*n+n-1) for c in range(nc-2)])
    ta = array(ta)

    return va, ta

#
# Cut distances along box edges.
#
def cut_distances(va, nc = 4):

    from chimerax.geometry import norm
    cuts = []
    cut = [0]*nc
    n = len(va)//(4*nc)
    for s in range(n-1):
        for c in range(nc):
            e = va[c*n:] if s%2 == 0 else va[((c + nc//2) % nc)*n:]
            cut[c] += norm(e[s+1] - e[s])
        cuts.append(tuple(cut))
    return cuts
        

