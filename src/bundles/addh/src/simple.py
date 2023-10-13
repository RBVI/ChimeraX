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

from math import sin, cos, pi, sqrt

sin5475 = sin(pi * 54.75 / 180.0)
cos5475 = cos(pi * 54.75 / 180.0)

def add_hydrogens(atom, *args, **kw):
    # determine what alt_loc(s) to add hydrogens to...
    alt_loc_atom = None
    if len(atom.alt_locs) > 1:
        alt_loc_atom = atom
    else:
        for nb in atom.neighbors:
            if len(nb.alt_locs) > 1:
                alt_loc_atom = nb
                break
    _alt_loc_add_hydrogens(atom, alt_loc_atom, *args, **kw)

def _alt_loc_add_hydrogens(atom, alt_loc_atom, bonding_info, naming_schema, total_hydrogens,
        idatm_type, invert, coordinations):
    from .cmd import find_nearest, roomiest, find_rotamer_nearest, add_altloc_hyds
    from .util import bond_with_H_length
    away = away2 = planar = None
    geom = bonding_info.geometry
    substs = bonding_info.substituents
    needed = substs - atom.num_explicit_bonds
    if needed <= 0:
        return
    added = None
    if alt_loc_atom is None:
        alt_locs = [atom.alt_loc]
    else:
        alt_locs = alt_loc_atom.alt_locs
        # move current alt_loc to end of list to minimize
        # the number of times we have to change alt locs
        cur_alt_loc = alt_loc_atom.alt_loc
        alt_locs.remove(cur_alt_loc)
        alt_locs.append(cur_alt_loc)
    alt_loc_info = []
    for alt_loc in alt_locs:
        if alt_loc_atom:
            alt_loc_atom.alt_loc = alt_loc
            occupancy = alt_loc_atom.occupancy
        else:
            occupancy = 1.0
        at_pos = atom._addh_coord
        exclude = coordinations + list(atom.neighbors)
        if geom == 3:
            if atom.num_bonds == 1:
                bonded = atom.neighbors[0]
                grand_bonded = list(bonded.neighbors)
                grand_bonded.remove(atom)
                if len(grand_bonded) < 3:
                    planar = [a._addh_coord for a in grand_bonded]
        if geom == 4 and atom.num_bonds == 0:
            away, d, natom = find_nearest(at_pos, atom, exclude, 3.5)
            if away is not None:
                away2, d2, natom2 = find_rotamer_nearest(at_pos, idatm_type[atom],
                    atom, natom, 3.5)
        elif geom == 4 and len(coordinations) + atom.num_bonds == 1:
            away, d, natom = find_rotamer_nearest(at_pos,
                    idatm_type[atom], atom, (list(atom.neighbors)+coordinations)[0], 3.5)
        else:
            away, d, natom = find_nearest(at_pos, atom, exclude, 3.5)

        bonded_pos = []
        for bonded in atom.neighbors:
            bonded_pos.append(bonded._addh_coord)

        if coordinations:
            toward = coordinations[0]._addh_coord
            away2 = away
            away = None
        else:
            toward = None
        from chimerax.atomic.bond_geom import bond_positions
        from chimerax.geometry import distance_squared
        positions = bond_positions(at_pos, geom, bond_with_H_length(atom, geom),
            bonded_pos, toward=toward, coplanar=planar, away=away, away2=away2)
        if coordinations:
            coord_pos = None
            for pos in positions:
                d = distance_squared(pos, toward)
                if coord_pos is None or d < lowest:
                    coord_pos = pos
                    lowest = d
            positions.remove(coord_pos)
        if len(positions) > needed:
            positions = roomiest(positions, atom, 3.5, bonding_info)[:needed]
        alt_loc_info.append((alt_loc, occupancy, positions))
    # delay adding Hs until all positions computed so that neighbors, etc. correct
    # for later alt locs
    add_altloc_hyds(atom, alt_loc_info, invert, bonding_info, total_hydrogens, naming_schema)
