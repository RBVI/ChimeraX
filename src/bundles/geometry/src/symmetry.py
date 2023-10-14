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

# -----------------------------------------------------------------------------
# Rotation about z axis.
#
def cyclic_symmetry_matrices(n, center = (0,0,0)):

    from . import Place, Places
    tflist = []
    from math import sin, cos, pi
    for k in range(n):
        a = 2*pi * float(k) / n
        c = cos(a)
        s = sin(a)
        tf = Place(((c, -s, 0, 0),
                    (s, c, 0, 0),
                    (0,0,1,0)))
        tflist.append(tf)
    ops = recenter_symmetries(Places(tflist), center)
    return ops

# -----------------------------------------------------------------------------
# Rotation about z axis, reflection about x axis.
#
def dihedral_symmetry_matrices(n, center = (0,0,0)):

    clist = cyclic_symmetry_matrices(n)
    from . import Place, Places
    reflect = Place(((1,0,0,0),(0,-1,0,0),(0,0,-1,0)))
    tflist = Places([Place(), reflect]) * clist
    tflist = recenter_symmetries(tflist, center)
    return tflist

# -----------------------------------------------------------------------------
#
tetrahedral_orientations = ('222', 'z3')
def tetrahedral_symmetry_matrices(orientation = '222', center = (0,0,0)):

    aa = (((0,0,1),0), ((1,0,0),180), ((0,1,0),180), ((0,0,1),180),
          ((1,1,1),120), ((1,1,1),240), ((-1,-1,1),120), ((-1,-1,1),240),
          ((-1,1,-1),120), ((-1,1,-1),240), ((1,-1,-1),120), ((1,-1,-1),240))
    from . import rotation, Places
    syms = Places([rotation(axis, angle) for axis, angle in aa])

    if orientation == 'z3':
        # EMAN convention, 3-fold on z, 3-fold in yz plane along neg y.
        from math import acos, sqrt, pi
        tf = rotation((0,0,1), -45.0) * rotation((1,0,0), -acos(1/sqrt(3))*180/pi)
        syms = syms.transform_coordinates(tf)

    syms = recenter_symmetries(syms, center)
    return syms

# -----------------------------------------------------------------------------
# 4-folds along x, y, z axes.
#
def octahedral_symmetry_matrices(center = (0,0,0)):

    c4 = (((0,0,1),0), ((0,0,1),90), ((0,0,1),180), ((0,0,1),270))
    cube = (((1,0,0),0), ((1,0,0),90), ((1,0,0),180), ((1,0,0),270),
            ((0,1,0),90), ((0,1,0),270))
    from . import rotation, Places
    c4syms = Places([rotation(axis, angle) for axis, angle in c4])
    cubesyms = Places([rotation(axis, angle) for axis, angle in cube])
    syms = cubesyms * c4syms
    syms = recenter_symmetries(syms, center)
    return syms

# -----------------------------------------------------------------------------
# Rise and angle per-subunit.  Angle in degrees.
#
def helical_symmetry_matrices(rise, angle, axis = (0,0,1), center = (0,0,0),
                              n = 1):
    
    zlist = [(i if i <= n/2 else n/2 - i) for i in range(n)]
    from . import Places
    syms = Places([helical_symmetry_matrix(rise, angle, axis, center, z)
                   for z in zlist])
    return syms

# -----------------------------------------------------------------------------
# Angle in degrees.
#
def helical_symmetry_matrix(rise, angle, axis = (0,0,1), center = (0,0,0), n = 1):

    from . import Place, rotation, translation
    if n == 0:
        return Place()
    rtf = rotation(axis, n*angle, center)
    shift = translation([x*n*rise for x in axis])
    tf = shift * rtf
    return tf

# -----------------------------------------------------------------------------
#
def translation_symmetry_matrices(n, delta):

    dx, dy, dz = delta
    from . import Place, Places
    tflist = Places([Place(((1, 0, 0, k*dx),
                            (0, 1, 0, k*dy),
                            (0, 0, 1, k*dz)))
                     for k in range(n)])
    return tflist

# -----------------------------------------------------------------------------
#
def recenter_symmetries(tflist, center):

    if center == (0,0,0):
      return tflist
    from . import translation
    ctf = translation([-x for x in center])
    return tflist.transform_coordinates(ctf)
