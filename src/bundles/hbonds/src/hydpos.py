# vim: set expandtab ts=4 sw=4:

from chimerax.atomic.idatm import type_info, tetrahedral, planar
from chimerax.atomic import Element
from chimerax.atomic.bond_geom import bond_positions

def hyd_positions(heavy, include_lone_pairs=False):
    """Return list of positions for hydrogens attached to this atom.
       If a hydrogen could be in one of several positions, don't return any of those.
    """

    # first, find known attached atoms
    bonded_heavys = []
    hyds = []
    for atom in heavy.neighbors:
        if atom.element.number > 1:
            bonded_heavys.append(atom)
        else:
            hyds.append(atom)

    # convert to Points
    hyd_locs = []
    for hyd in hyds:
        hyd_locs.append(hyd._hb_coord)

    if hyd_locs and not include_lone_pairs:
        # explicit hydrogens "win" over atom types
        return hyd_locs

    if heavy.idatm_type in type_info:
        info = type_info[heavy.idatm_type]
        geom = info.geometry
        if include_lone_pairs:
            subs = geom
        else:
            subs = info.substituents
        bonded_locs = hyd_locs[:]
        for b_heavy in bonded_heavys:
            bonded_locs.append(b_heavy._hb_coord)
    else:
        return hyd_locs

    known_subs = len(bonded_locs)
    if known_subs >= subs or known_subs == 0:
        return hyd_locs
    # above eliminates 'single' geometry

    if known_subs == 1 and geom == tetrahedral:
        # rotamer
        return hyd_locs

    max_subs = geom
    if max_subs - subs > 0:
        # the "empty" bond could be anywhere
        return hyd_locs

    heavy_loc = heavy._hb_coord
    bond_len = Element.bond_length(heavy.element, "H")

    if geom == planar:
        coplanar = []
        for b_heavy in bonded_heavys:
            try:
                bh_geom = type_info[b_heavy.idatm_type].geometry
            except KeyError:
                bh_geom = None
            if bh_geom != planar:
                continue
            for atom in b_heavy.neighbors:
                if atom != heavy:
                    coplanar.append(atom._hb_coord)

    else:
        coplanar = None

    hyd_locs = hyd_locs + list(
        bond_positions(heavy_loc, geom, bond_len, bonded_locs, coplanar=coplanar))
    return hyd_locs
