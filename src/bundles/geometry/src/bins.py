# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2022 Regents of the University of California. All rights reserved.
# The ChimeraX application is provided pursuant to the ChimeraX license
# agreement, which covers academic and commercial uses. For more details, see
# <http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
#
# This particular file is part of the ChimeraX library. You can also
# redistribute and/or modify it under the terms of the GNU Lesser General
# Public License version 2.1 as published by the Free Software Foundation.
# For more details, see
# <https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html>
#
# THIS SOFTWARE IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND, EITHER
# EXPRESSED OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
# OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. ADDITIONAL LIABILITY
# LIMITATIONS ARE DESCRIBED IN THE GNU LESSER GENERAL PUBLIC LICENSE
# VERSION 2.1
#
# This notice must be embedded in or attached to all copies, including partial
# copies, of the software or any revisions or derivations thereof.
# === UCSF ChimeraX Copyright ===

# -----------------------------------------------------------------------------
# Eliminate copies of close transformations from a set of transformations.
#
class Binned_Transforms:

    def __init__(self, angle, translation, center=(0, 0, 0), bfactor=10):

        self.angle = angle              # In radians.
        self.translation = translation
        self.center = center            # Used for defining translation.
        spacing = (angle, translation, translation, translation)
        self.spacing = spacing
        bin_size = [s * bfactor for s in spacing]
        self.bins = Bins(bin_size)

    # -------------------------------------------------------------------------
    #
    def add_transform(self, tf):

        self.bins.add_object(self.bin_point(tf), tf)

    # -------------------------------------------------------------------------
    #
    def bin_point(self, tf):

        a = tf.rotation_angle()  # In range 0 to pi
        x, y, z = tf * self.center
        return (a, x, y, z)

    # -------------------------------------------------------------------------
    #
    def close_transforms(self, tf):

        a, x, y, z = c = self.bin_point(tf)
        clist = self.bins.close_objects(c, self.spacing)
        if len(clist) == 0:
            return []

        close = []
        itf = tf.inverse()
        d2max = self.translation * self.translation
        for ctf in clist:
            cx, cy, cz = ctf * self.center
            dx, dy, dz = x - cx, y - cy, z - cz
            d2 = dx * dx + dy * dy + dz * dz
            if d2 <= d2max:
                dtf = ctf * itf
                a = dtf.rotation_angle()
                if a < self.angle:
                    close.append(ctf)

        return close


# -----------------------------------------------------------------------------
# Bin objects in a grid for fast lookup of objects close to a given object.
#
class Bins:

    def __init__(self, bin_size):

        self.bin_size = tuple([float(s) for s in bin_size])
        self.bins = {}

    # -------------------------------------------------------------------------
    #
    def add_object(self, coords, object):

        bc = [c / bs for c, bs in zip(coords, self.bin_size)]
        from math import floor
        b = tuple([int(floor(c)) for c in bc])
        if b in self.bins:
            self.bins[b].append((coords, object))
        else:
            self.bins[b] = [(coords, object)]

    # -------------------------------------------------------------------------
    #
    def close_objects(self, coords, range):

        bc = [c / bs for c, bs in zip(coords, self.bin_size)]
        br = [r / bs for r, bs in zip(range, self.bin_size)]
        cobjects = {}
        cbins = self.close_bins(bc, br)
        for b in cbins:
            if b in self.bins:
                for c, o in self.bins[b]:
                    if self.are_coordinates_close(c, coords, range):
                        cobjects[id(o)] = o
        clist = list(cobjects.values())
        return clist

    # -------------------------------------------------------------------------
    #
    def close_bins(self, bc, br):

        from math import floor
        rs = [range(int(floor(c - r)), int(floor(c + r)) + 1)
              for c, r in zip(bc, br)]
        cbins = outer_product(rs)
        return cbins

    # -------------------------------------------------------------------------
    #
    def are_coordinates_close(self, c, coords, dist):

        for k in range(len(c)):
            if abs(c[k] - coords[k]) > dist[k]:
                return False
        return True


# -----------------------------------------------------------------------------
#
def outer_product(sets):

    if len(sets) == 0:
        op = []
    elif len(sets) == 1:
        op = [(c,) for c in sets[0]]
    else:
        op = []
        op1 = outer_product(sets[1:])
        for c in sets[0]:
            for cr in op1:
                op.append((c,) + cr)
    return op
