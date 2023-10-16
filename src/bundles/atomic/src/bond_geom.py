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

"""
bond_geom: bond geometry utilities and info relevant to bond geometry
===============================

TODO
"""

import tinyarray, numpy
from chimerax.geometry import normalize_vector as normalize
from chimerax.geometry import norm, cross_product, Plane, look_at, rotation, z_align, angle
sqlength = lambda v: numpy.sum(v*v)

from math import pi, sin, cos

geometry_name = ['ion', 'single', 'linear', 'trigonal', 'tetrahedral']
ion, single, linear, planar, tetrahedral = range(len(geometry_name))
# above corresponds to C++ IdatmGeometry enum

sin5475 = sin(pi * 54.75 / 180.0)
cos5475 = cos(pi * 54.75 / 180.0)
cos705 = cos(pi * 70.5 / 180.0)

def bond_positions(bondee, geom, bond_len, bonded, coplanar=None,
            toward=None, away=None, toward2=None, away2=None):
    """Returns a list of possible bond partner positions for 'bondee' that
       satisfy geometry 'geom' and are of length 'bond_len'.  'bonded' are
       positions of already-bonded substituents to 'bondee'.

       'bondee' is a "point" (array of 3 64-bit floats) and 'bonded'
       is a list of points.  The return value is also a list points.

       For planar geometries, 'coplanar' can be a list of one or two
       points that the bond positions should be coplanar with.

       For rotamers, if 'toward' is specified then one bond position of
       the rotamer will be as close as possible to the 'toward' point.
       Conversely, if 'away' is specified, then all bond positions will
       be as far as possible from the 'away' point.  'toward' has
       precedence over 'away'.  If this doesn't completely determine
       the bond positions (i.e. no initial bonded substituents), then
       'toward2' and 'away2' will be considered in a similar manner to
       determine the remaining positions.
    """

    if (toward is not None and away is not None) or (toward2 is not None and away2 is not None):
        raise ValueError("Cannot specify both toward and away, or both toward2 and away2")
    if geom == single:
        if len(bonded) > 0:
            return []
        return [single_pos(bondee, bond_len, toward, away)]

    if geom == linear:
        if len(bonded) > 1:
            return []
        return linear_pos(bondee, bonded, bond_len, toward, away)

    if geom == planar:
        if len(bonded) > 2:
            return []
        return planar_pos(bondee, bonded, bond_len, coplanar, toward, away, toward2, away2)

    if geom == tetrahedral:
        if len(bonded) > 3:
            return []
        return tetra_pos(bondee, bonded, bond_len, toward, away, toward2, away2)

    raise ValueError("Unknown geometry type '%s'" % geom)


def single_pos(bondee, bond_len, toward=None, away=None):
    if toward is not None:
        v = toward - bondee
        return bondee + normalize(v) * bond_len
    elif away is not None:
        v = bondee - away
        return bondee + normalize(v) * bond_len
    return bondee + tinyarray.array([bond_len, 0.0, 0.0])

def linear_pos(bondee, bonded, bond_len, toward=None, away=None):
    new_bonded = []
    cur_bonded = bonded[:]
    if len(bonded) == 0:
        if away is not None:
            # need 90 angle, rather than directly away
            # (since otherwise second added position will then be
            # directly towards)
            ninety = right_angle(away - bondee)
            away = bondee + ninety
        pos = single_pos(bondee, bond_len, toward, away)
        new_bonded.append(pos)
        cur_bonded.append(pos)

    if len(cur_bonded) == 1:
        bondVec = normalize(bondee - cur_bonded[0]) * bond_len
        new_bonded.append(bondee + bondVec)
    return new_bonded

def planar_pos(bondee, bonded, bond_len, coplanar=None, toward=None, away=None,
                        toward2=None, away2=None):
    new_bonded = []
    cur_bonded = bonded[:]
    if len(cur_bonded) == 0:
        pos = single_pos(bondee, bond_len, toward, away)
        toward = away = None
        new_bonded.append(pos)
        cur_bonded.append(pos)

    if len(cur_bonded) == 1:
        # add at 120 degree angle, co-planar if required
        if coplanar is None:
            if toward is not None or toward2 is not None:
                coplanar = [toward if toward is not None else toward2]
            elif away is not None or away2 is not None:
                ninety = right_angle((away if away is not None else away2) - bondee)
                coplanar = [bondee + ninety]
        pos = angle_pos(bondee, cur_bonded[0], bond_len, 120.0, coplanar)
        new_bonded.append(pos)
        cur_bonded.append(pos)

    if len(cur_bonded) == 2:
        # position along anti-bisector of current bonds
        v1 = normalize(cur_bonded[0] - bondee)
        v2 = normalize(cur_bonded[1] - bondee)
        anti_bi = normalize(tinyarray.negative(v1 + v2)) * bond_len
        new_bonded.append(bondee + anti_bi)
    return new_bonded

def tetra_pos(bondee, bonded, bond_len, toward=None, away=None, toward2=None, away2=None):
    new_bonded = []
    cur_bonded = bonded[:]
    if len(cur_bonded) == 0:
        pos = single_pos(bondee, bond_len, toward, away)
        toward = toward2
        away = away2
        new_bonded.append(pos)
        cur_bonded.append(pos)

    if len(cur_bonded) == 1:
        # add at 109.5 degree angle
        coplanar = toward or away
        if coplanar and angle(bondee, cur_bonded[0], coplanar) not in (0, 180):
            coplanar = [coplanar]
        else:
            coplanar = None
        pos = angle_pos(bondee, cur_bonded[0], bond_len, 109.5, coplanar=coplanar)
        if toward or away:
            # find the other 109.5 position in the toward/away
            # plane and the closer/farther position as appropriate
            old = normalize(bondee - cur_bonded[0])
            new = pos - bondee
            midpoint = tinyarray.array(bondee + old * norm(new) * cos705)
            other_pos = pos + (midpoint - pos) * 2
            d1 = sqlength(pos - (toward or away))
            d2 = sqlength(other_pos - (toward or away))
            if toward is not None:
                if d2 < d1:
                    pos = other_pos
            elif away is not None and d2 > d1:
                pos = other_pos

        new_bonded.append(pos)
        cur_bonded.append(pos)
    
    if len(cur_bonded) == 2:
        # add along anti-bisector of current bonds and raised up
        # 54.75 degrees from plane of those bonds (half of 109.5)
        v1 = normalize(cur_bonded[0] - bondee)
        v2 = normalize(cur_bonded[1] - bondee)
        anti_bi = normalize(tinyarray.negative(v1 + v2))
        # in order to stabilize the third and fourth tetrahedral
        # positions, cross the longer vector by the shorter
        if sqlength(v1) > sqlength(v2):
            cross_v = normalize(cross_product(v1, v2))
        else:
            cross_v = normalize(cross_product(v2, v1))

        anti_bi = anti_bi * cos5475 * bond_len
        cross_v = cross_v * sin5475 * bond_len

        pos = bondee + anti_bi + cross_v
        if toward or away:
            other_pos = bondee + anti_bi - cross_v
            d1 = sqlength(pos - (toward or away))
            d2 = sqlength(other_pos - (toward or away))
            if toward is not None:
                if d2 < d1:
                    pos = other_pos
            elif away is not None and d2 > d1:
                pos = other_pos
        new_bonded.append(pos)
        cur_bonded.append(pos)
    
    if len(cur_bonded) == 3:
        unitized = []
        for cb in cur_bonded:
            v = normalize(cb - bondee)
            unitized.append(bondee + v)
        pl = Plane(unitized)
        normal = pl.normal
        # if normal on other side of plane from bondee, we need to
        # invert the normal;  the (signed) distance from bondee
        # to the plane indicates if it is on the same side
        # (positive == same side)
        d = pl.distance(bondee)
        if d < 0.0:
            normal = tinyarray.negative(normal)
        new_bonded.append(bondee + normal * bond_len)
    return new_bonded

        
def angle_pos(atom_pos, bond_pos, bond_length, degrees, coplanar=None):
    if coplanar is not None and len(coplanar) > 0:
        # may have one or two coplanar positions specified,
        # if two, compute both resultant positions and average
        # (the up vector has to be negated for the second one)
        xforms = []
        if len(coplanar) > 2:
            raise ValueError("More than 2 coplanar positions specified!")
        for cpos in coplanar:
            up = cpos - atom_pos
            if xforms:
                up = tinyarray.negative(up)
            # lookAt puts ref point opposite that of zAlign, so 
            # also rotate 180 degrees around y axis
            xform = rotation((0.0,1.0,0.0), 180.0) * look_at(atom_pos, bond_pos, up)
            xforms.append(xform)

    else:
        xforms = [z_align(atom_pos, bond_pos)]
    points = []
    for xform in xforms:
        radians = pi * degrees / 180.0
        angle = tinyarray.array([0.0, bond_length * sin(radians), bond_length * cos(radians)])
        points.append(xform.inverse() * angle)
    
    if len(points) > 1:
        midpoint = points[0] + (points[1] - points[0]) / 2.0
        v = midpoint - atom_pos
        v = v * bond_length / norm(v)
        return atom_pos + v

    return tinyarray.array(points[0])

def right_angle(orig):
    if orig[0] == 0.0:
        return tinyarray.array([0.0, 0 - orig[2], orig[1]])
    return tinyarray.array([0.0 - orig[1], orig[0], 0.0])
