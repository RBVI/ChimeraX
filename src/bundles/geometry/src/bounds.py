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

'''
bounds: Bounding boxes
======================

Compute bounding boxes for objects in scene.
The bounds of no object is represented as None rather than a Bounds object.
'''

# Bounding box computations
class Bounds:
    '''Bounding box specified by minimum and maximum x,y,z coordinates.'''
    def __init__(self, xyz_min, xyz_max):
        '''Supported API. Define a bounding box using two corner points.'''
        # Make sure bounds are numpy arrays
        from numpy import ndarray, array, float32
        xmin = xyz_min if isinstance(xyz_min, ndarray) else array(xyz_min, float32)
        xmax = xyz_max if isinstance(xyz_max, ndarray) else array(xyz_max, float32)

        self.xyz_min = xmin
        "Supported API. Minimum x,y,z bounds as numpy float32 array."
        self.xyz_max = xmax
        "Supported API. Maximum x,y,z bounds as numpy float32 array."

    def center(self):
        "Supported API. Center of bounding box."
        return 0.5 * (self.xyz_min + self.xyz_max)

    def size(self):
        "Supported API. Size of box along x,y,z axes. Returns 3 values."
        return self.xyz_max - self.xyz_min

    def width(self):
        "Supported API. Maximum of size of box x,y,z axes."
        return (self.xyz_max - self.xyz_min).max()

    def radius(self):
        "Supported API. Radius of sphere containing bounding box."
        size = self.xyz_max - self.xyz_min
        from math import sqrt
        r = 0.5*sqrt((size*size).sum())
        return r

    def box_corners(self):
        '''Supported API. Return a float32 numpy array of the 8 corners of the bounding box.'''
        (x0, y0, z0), (x1, y1, z1) = self.xyz_min, self.xyz_max
        corners = ((x0, y0, z0), (x1, y0, z0), (x0, y1, z0), (x1, y1, z0),
                   (x0, y0, z1), (x1, y0, z1), (x0, y1, z1), (x1, y1, z1))
        from numpy import array, float32
        c = array(corners, float32)
        return c

    def contains_point(self, p):
        for a in (0,1,2):
            if p[a] < self.xyz_min[a] or p[a] > self.xyz_max[a]:
                return False
        return True
                

def point_bounds(xyz, placements=None):
    '''
    Supported API.
    Return :py:class:`.Bounds` for a set of points, optionally
    multiple positions given by a :py:class:`.Places` instance.
    '''
    if len(xyz) == 0:
        return None

    if placements:
        from ._geometry import point_copies_bounds
        xyz_min, xyz_max = point_copies_bounds(xyz, placements.array())
    else:
        from ._geometry import point_bounds
        xyz_min, xyz_max = point_bounds(xyz)

    b = Bounds(xyz_min, xyz_max)
    return b

def union_bounds(blist):
    '''
    Supported API.
    Return :py:class:`.Bounds` that is the union of a list
    of :py:class:`.Bounds`. The list can contain None elements
    which are ignored.  If the list contains no :py:class:`.Bounds`
    then None is returned.
    '''
    xyz_min, xyz_max = None, None
    for b in blist:
        if b is None:
            continue
        pmin, pmax = b.xyz_min, b.xyz_max
        if xyz_min is None:
            xyz_min, xyz_max = pmin, pmax
        else:
            xyz_min = tuple(min(x, px) for x, px in zip(xyz_min, pmin))
            xyz_max = tuple(max(x, px) for x, px in zip(xyz_max, pmax))
    b = None if xyz_min is None else Bounds(xyz_min, xyz_max)
    return b


def copies_bounding_box(bounds, positions):
    '''
    Supported API.
    Return :py:class:`.Bounds` that covers a specified bounding
    box replicated at :py:class:`.Places`.
    '''
    if bounds is None or len(positions) == 0:
        return None
    sas = positions.shift_and_scale_array()
    if sas is not None and len(sas) > 0:
        # Optimize shift and scale positions.
        xyz, s = sas[:, :3], sas[:,3]
        # TODO: Optimize this with a C++ routine to avoid array copies.
        from numpy import outer
        b = Bounds((xyz + outer(s, bounds.xyz_min)).min(axis=0),
                   (xyz + outer(s, bounds.xyz_max)).max(axis=0))
    else:
        b = point_bounds(bounds.box_corners(), positions)
    return b

def copy_tree_bounds(bounds, positions_list):
    '''
    Return :py:class:`.Bounds` that covers a specified bounding
    box replicated at a hierarchy of :py:class:`.Places`.
    '''
    for p in reversed(positions_list):
        bounds = copies_bounding_box(bounds, p)
    return bounds

def sphere_bounds(centers, radii):
    '''
    Supported API.
    Return :py:class:`.Bounds` containing a set of spheres.
    '''
    if len(centers) == 0:
        return None
    from . import _geometry
    b = _geometry.sphere_bounds(centers, radii)
    return Bounds(b[0], b[1])

def clip_bounds(b, planes):
    '''
    Supported API.
    Clip a bounding box using planes each given as a plane point and normal.
    '''
    if len(planes) == 0:
        return b
    from . import inner_product
    for p,n in planes:
        points = []
        corners = b.box_corners()
        for c in corners:
            if inner_product(c-p,n) >= 0:
                points.append(c)
        for i1,i2 in ((0,1),(1,3),(3,2),(2,0),
                      (4,5),(5,7),(7,6),(6,4),
                      (0,4),(1,5),(2,6),(3,7)):
            c1, c2 = corners[i1], corners[i2]
            f1, f2 = inner_product(c1-p,n), inner_product(c2-p,n)
            if (f1 > 0 and f2 < 0) or (f1 < 0 and f2 > 0):
                t = f1/(f1-f2)
                points.append((1-t)*c1 + t*c2)
        b = point_bounds(points)
    return b
