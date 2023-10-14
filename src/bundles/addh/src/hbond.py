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

from chimerax.atomic.bond_geom import tetrahedral, planar, linear, single, bond_positions
from chimerax.atomic import Atom, idatm, Ring
from .cmd import find_nearest, roomiest, _tree_dist, vdw_radius, \
                find_rotamer_nearest, h_rad, add_altloc_hyds
from .util import bond_with_H_length, N_H
from chimerax.geometry import distance_squared, angle, dihedral, distance, normalize_vector

c_rad = 1.7
_near_dist = _room_dist = _tree_dist + h_rad + c_rad

_test_angles = [None, None, 60.0, 40.0, 36.5]

debug = False

def add_hydrogens(session, atom_list, *args):
    """Add hydrogens to maximize h-bonding and minimize steric clashes.

       First, add hydrogens that are fixed and always present.  Then,
       find H-bonds only to possibly-hydrogen-needing donors.  For
       aromatic nitogens protonate at least one on a six-membered ring.
       For remaining rotamers with H-bonds, place hydrogens
       to maximize H-bond interactions.  Then, for remaining rotamers,
       minimize steric clashes via torsion-driving.

       May not work well for atoms that are already partially protonated.
    """

    logger = session.logger
    if debug:
        print("here (test)")
    if atom_list and atom_list[0].structure.num_coordsets > 1:
        # First off, trajectories should already _have_ hydrogens.
        # Second, it would be a nightmare to compute all the positions
        # before adding any of the hydrogens (so that neighbor lists
        # are correct during the computation)
        from chimerax.core.errors import LimitationError
        raise LimitationError("Adding H-bond-preserving hydrogens to trajectories not supported.")
    if debug:
        print("here 2")
    global type_info_for_atom, naming_schemas, hydrogen_totals, \
                    idatm_type, inversion_cache, coordinations
    type_info_for_atom, naming_schemas, hydrogen_totals, idatm_type, his_Ns, \
                            coordinations, in_isolation = args
    inversion_cache = {}
    logger.status("Categorizing heavy atoms", blank_after=0, secondary=True)
    problem_atoms = []
    fixed = []
    flat = []
    aro_N_rings = {}
    global aro_amines
    aro_amines = {}
    unsaturated = []
    saturated = []
    finished = {}
    # the early donor/acceptor version of hbond_info is not keyed by alt loc (for personal
    # sanity, treat hbonds as applicable for all altlocs), but later H-position version
    # is a list with current alt loc last
    hbond_info = {}
    for a, crds in coordinations.items():
        hbond_info[a] = [(True, crd) for crd in crds]
        for crd in crds:
            hbond_info.setdefault(crd, []).append((False, a))
    for atom in atom_list:
        bonding_info = type_info_for_atom[atom]
        num_bonds = atom.num_bonds
        substs = bonding_info.substituents
        geom = bonding_info.geometry
        if atom.num_explicit_bonds >= substs:
            if atom.element.number == 7 and num_bonds == 3 and geom == planar:
                for ring in atom.rings():
                    if ring.aromatic:
                        if ring in aro_N_rings:
                            aro_N_rings[ring].append(atom)
                        else:
                            aro_N_rings[ring] = [atom]
            continue
        
        if num_bonds == 0:
            unsaturated.append(atom)
        elif num_bonds == 1:
            if geom == linear:
                fixed.append(atom)
            elif geom == planar:
                if substs == 2:
                    unsaturated.append(atom)
                else:
                    if idatm_type[atom] == 'Npl' \
                    and idatm_type[atom.neighbors[0]] == 'Car':
                        # primary aromatic amine
                        aro_amines[atom] = True
                    else:
                        flat.append(atom)
            elif geom == tetrahedral:
                if substs < 4:
                    unsaturated.append(atom)
                else:
                    saturated.append(atom)
            else:
                raise AssertionError("bad geometry for atom"
                    " (%s) with one bond partner" % atom)
        elif num_bonds == 2:
            if geom == tetrahedral:
                if substs == 3:
                    unsaturated.append(atom)
                else:
                    fixed.append(atom)
            elif geom == planar:
                aro_N_ring = None
                if atom.element.number == 7:
                    for ring in atom.rings():
                        if ring.aromatic:
                            aro_N_ring = ring
                            break
                if aro_N_ring:
                    if aro_N_ring in aro_N_rings:
                        aro_N_rings[aro_N_ring].append(atom)
                    else:
                        aro_N_rings[aro_N_ring] = [atom]
                else:
                    fixed.append(atom)
            else:
                raise AssertionError("bad geometry for atom"
                    " (%s) with two bond partners" % atom)
        elif num_bonds == 3:
            if geom != tetrahedral:
                raise AssertionError("bad geometry for atom"
                    " (%s) with three bond partners" % atom)
            fixed.append(atom)

    # protonate aromatic nitrogens if only one nitrogen in ring:
    multi_N_rings = {}
    aro_Ns = {}
    for ring, ns in aro_N_rings.items():
        if len(ns) == 1:
            fixed.append(ns[0])
        elif ns[0] in his_Ns:
            for n in ns:
                if his_Ns[n]:
                    fixed.append(n)
        else:
            multi_N_rings[ring] = ns
            for n in ns:
                aro_Ns[n] = True
    if debug:
        print(len(fixed), "fixed")
        print(len(flat), "flat")
        print(len(aro_amines), "primary aromatic amines")
        print(len(multi_N_rings), "aromatic multi-nitrogen rings")
        print(len(unsaturated), "unsaturated")
        print(len(saturated), "saturated rotamers")

    for need_coplanar in [False, True]:
        if need_coplanar:
            atoms = flat
            logger.status("Adding co-planar hydrogens", blank_after=0, secondary=True)
        else:
            atoms = fixed
            logger.status("Adding simple fixed hydrogens", blank_after=0, secondary=True)

        for atom in atoms:
            bonding_info = _type_info(atom)
            geom = bonding_info.geometry
            bond_length = bond_with_H_length(atom, geom)
            altloc_atom = None
            if atom.num_alt_locs > 1:
                altloc_atom = atom
            else:
                for nb in atom.neighbors:
                    if nb.num_alt_locs > 1:
                        altloc_atom = nb
                        break
                else:
                    if need_coplanar:
                        for nb in atom.neighbors:
                            for gnb in nb.neighbors:
                                if gnb.num_alt_locs > 1:
                                    altloc_atom = gnb
                                    break
                            else:
                                continue
                            break
            if altloc_atom is None:
                altloc_atom = atom
            coplanar_atoms = []
            if need_coplanar:
                for na in atom.neighbors:
                    for nna in na.neighbors:
                        if nna != atom:
                            coplanar_atoms.append(nna)
            if len(coplanar_atoms) > 2:
                problem_atoms.append(atom)
                continue
            altloc_hpos_info = []
            for alt_loc in _alt_locs(altloc_atom):
                altloc_atom.alt_loc = alt_loc
                h_positions = bond_positions(atom._addh_coord, geom, bond_length,
                        [nb._addh_coord for nb in atom.neighbors],
                        coplanar=[cpa._addh_coord for cpa in coplanar_atoms])
                altloc_hpos_info.append((alt_loc, altloc_atom.occupancy, h_positions))
            try:
                _attach_hydrogens(atom, altloc_hpos_info, bonding_info)
            except:
                print("altloc_atom:", altloc_atom)
                raise
            finished[atom] = True
            hbond_info[atom] = [(alt_loc, h_positions, [])
                for alt_loc, _, h_positions in altloc_hpos_info]

    from chimerax.hbonds import find_hbonds, rec_dist_slop, rec_angle_slop

    logger.status("Finding hydrogen bonds", blank_after=0, secondary=True)
    donors = {}
    acceptors = {}
    for atom in unsaturated:
        donors[atom] = unsaturated
        acceptors[atom] = unsaturated
    for atom in saturated:
        donors[atom] = saturated
    for ring, ns in multi_N_rings.items():
        for n in ns:
            donors[n] = ring
            acceptors[n] = ring
    for atom in aro_amines:
        donors[atom] = aro_amines
        acceptors[atom] = aro_amines

    if in_isolation:
        if atom_list:
            s_list = [atom_list[0].structure]
        else:
            return
    else:
        from chimerax.atomic import AtomicStructure
        s_list = [s for s in session.models if isinstance(s, AtomicStructure)]
    hbonds = find_hbonds(session, s_list, dist_slop=rec_dist_slop, angle_slop=rec_angle_slop)
    logger.info("%d hydrogen bonds" % len(hbonds))
    # want to assign hydrogens to strongest (shortest) hydrogen
    # bonds first, so sort by distance
    logger.status("Sorting hydrogen bonds by distance", blank_after=0, secondary=True)
    sortable_hbonds = []
    for d, a in hbonds:
        # ignore hbonds for atoms completely occupied with
        # coordinating metals...
        if a in coordinations:
            if len(coordinations[a]) + a.num_bonds >= _type_info(a).geometry:
                if debug:
                    print(a, "completely coordinated by metal")
                continue
        if d in donors:
            sortable_hbonds.append((distance_squared(d._addh_coord, a._addh_coord), False, (d, a)))
        if a in acceptors:
            sortable_hbonds.append((distance_squared(d._addh_coord, a._addh_coord), True, (d, a)))
    sortable_hbonds.sort()
    logger.status("Organizing h-bond info", blank_after=0, secondary=True)
    hbonds = {}
    ambiguous = {}
    rel_bond = {}
    for dsq, isAcc, hbond in sortable_hbonds:
        hbonds[hbond] = isAcc
        for da in hbond:
            rel_bond.setdefault(da, []).append(hbond)
    processed = set()
    break_ambiguous = False
    break_N_ring = False
    candidates = {}
    candidates.update(donors)
    candidates.update(acceptors)
    pruned = set()
    pruned_by = {}
    reexamine = {}
    while len(processed) < len(hbonds):
        logger.status("Adding hydrogens by h-bond strength (%d/%d)"
            % (len(processed), len(hbonds)), blank_after=0, secondary=True)
        processed_one = False
        seen_ambiguous = {}
        for dsq, is_acc, hbond in sortable_hbonds:
            if hbond in processed:
                continue
            d, a = hbond
            if hbond in pruned:
                if hbond in reexamine:
                    del reexamine[hbond]
                    if debug:
                        print("re-examining", hbond[0], "->", hbond[1])
                elif not break_ambiguous:
                    seen_ambiguous[d] = True
                    seen_ambiguous[a] = True
                    continue
            if (d not in candidates or d in finished) \
            and (a not in candidates or a in finished):
                # not relevant
                processed.add(hbond)
                continue

            if (a, d) in hbonds \
            and a not in finished \
            and d not in finished \
            and not _resolve_anilene(d, a, aro_amines, hbond_info) \
            and _angle_check(d, a, hbond_info) == (True, True):
                # possibly still ambiguous

                # if one or both ends are aromatic nitrogen, then ambiguity depends on
                # whether other nitrogens in ring are finished and are acceptors...
                nring = False
                try:
                    ns = multi_N_rings[acceptors[a]]
                except (KeyError, TypeError):
                    ns = []
                if len(ns) > 1:
                    nring = True
                    for n in ns:
                        if n not in finished:
                            break
                        alt_loc, protons, lps = hbond_info[n][-1]
                        if protons:
                            break
                    else:
                        # unambiguous, but in the reverse direction!
                        processed.add(hbond)
                        if debug:
                            print("reverse ambiguity resolved")
                        continue
                unambiguous = False
                try:
                    ns = multi_N_rings[donors[d]]
                except (KeyError, TypeError):
                    ns = []
                if len(ns) > 1:
                    nring = True
                    for n in ns:
                        if n not in finished:
                            break
                        alt_loc, protons, lps = hbond_info[n][-1]
                        if protons:
                            break
                    else:
                        if debug:
                            print("ambiguity resolved due to ring acceptors")
                        unambiguous = True

                if not unambiguous:
                    if not break_ambiguous or nring and not break_N_ring:
                        if debug:
                            print("postponing", [str(a) for a in hbond])
                        seen_ambiguous[d] = True
                        seen_ambiguous[a] = True
                        if hbond not in pruned:
                            _do_prune(hbond, pruned, rel_bond, processed, pruned_by)
                        continue
                    if nring:
                        if debug:
                            print("breaking ambiguous N-ring hbond")
                    else:
                        if debug:
                            print("breaking ambiguous hbond")
            if (a, d) in hbonds:
                if a in finished:
                    if debug:
                        print("breaking ambiguity because acceptor finished")
                elif d in finished:
                    if debug:
                        print("breaking ambiguity because donor finished")
                elif _resolve_anilene(d, a, aro_amines, hbond_info):
                    if debug:
                        print("breaking ambiguity by resolving anilene")
                elif _angle_check(d, a, hbond_info) != (True, True):
                    if debug:
                        print("breaking ambiguity due to angle check")
            processed.add(hbond)
            if debug:
                from math import sqrt
                print("processed", d, "->", a, sqrt(dsq))
            # if donor or acceptor is tet, see if geometry can work out
            if d not in finished and _type_info(d).geometry == 4:
                if not _tet_check(d, a, hbond_info, False):
                    continue
            if a not in finished and _type_info(a).geometry == 4:
                if d in finished:
                    checks = []
                    for b in d.neighbors:
                        if b.element.number == 1:
                            checks.append(b)
                else:
                    checks = [d]
                tet_okay = True
                for c in checks:
                    if not _tet_check(a, c, hbond_info, True):
                        tet_okay = False
                        break
                if not tet_okay:
                    continue
            if a in finished:
                # can d still donate to a?
                if not _can_accept(d, a, *hbond_info[a][-1][1:]):
                    # can't
                    continue
                hbond_info.setdefault(d, []).append((False, a))
            elif d in finished:
                # can d still donate to a?
                no_protons_ok = d in aro_Ns and d.num_bonds == 2
                if not _can_donate(d, a, no_protons_ok, *hbond_info[d][-1][1:]):
                    # nope
                    continue
                hbond_info.setdefault(a, []).append((True, d))
            else:
                add_a_to_d, add_d_to_a = _angle_check(d, a, hbond_info)
                if not add_a_to_d and not add_d_to_a:
                    continue
                if add_a_to_d:
                    hbond_info.setdefault(d, []).append((False, a))
                if add_d_to_a:
                    hbond_info.setdefault(a, []).append((True, d))
            if (a, d) in hbonds:
                processed.add((a, d))
            processed_one = True
            break_ambigous = False
            break_N_ring = False

            if hbond not in pruned:
                _do_prune(hbond, pruned, rel_bond, processed, pruned_by)
            for end in hbond:
                if end in finished or end not in candidates:
                    continue
                if end not in hbond_info:
                    # steric clash from other end
                    if debug:
                        print("no hbonds left for", end)
                    continue
                if debug:
                    print("try to finish", end, end="")
                    for is_acc, da in hbond_info[end]:
                        if is_acc:
                            print(" accepting from", da, end="")
                        else:
                            print(" donating to", da, end="")
                    print()
                did_finish = _try_finish(end, hbond_info, finished, aro_amines, pruned_by, processed)
                if not did_finish:
                    continue
                if debug:
                    print("finished", end)

                # if this atom is in candidates then actually add any hydrogens
                if end in candidates and hbond_info[end][-1][1]:
                    altloc_hpos_info = []
                    for alt_loc, h_positions, lone_pairs in hbond_info[end]:
                        altloc_hpos_info.append((alt_loc, end.occupancy, h_positions))
                    _attach_hydrogens(end, altloc_hpos_info, _type_info(end))
                    if debug:
                        print("protonated", end)
                    # if ring nitrogen, eliminate any hbonds where it is an acceptor
                    if isinstance(donors[end], Ring):
                        for rb in rel_bond[end]:
                            if rb[1] != end:
                                continue
                            processed.add(rb)

            if a in seen_ambiguous or d in seen_ambiguous:
                # revisit previously ambiguous
                if debug:
                    print("revisiting previous ambiguous")
                for da in hbond:
                    for rel in rel_bond[da]:
                        if rel in processed:
                            continue
                        if rel not in pruned:
                            continue
                        reexamine[rel] = True
                break
        if break_ambiguous and not processed_one:
            break_N_ring = True
        break_ambiguous = not processed_one
    num_finished = 0
    for a in candidates.keys():
        if a in finished:
            num_finished += 1
    if debug:
        print("finished", num_finished, "of", len(candidates), "atoms")

    logger.status("Adding hydrogens to primary aromatic amines", blank_after=0, secondary=True)
    if debug:
        print("primary aromatic amines")
    for a in aro_amines:
        if a in finished:
            continue
        if a not in candidates:
            continue
        if debug:
            print("amine", a)
        finished[a] = True
        num_finished += 1

        # point protons right toward acceptors;
        # if also accepting, finish as tet, otherwise as planar
        accept_from = None
        targets = []
        at_pos = a._addh_coord
        for is_acc, other in hbond_info.get(a, []):
            if is_acc:
                if not accept_from:
                    if debug:
                        print("accept from", other)
                    accept_from = other._addh_coord
                continue
            if debug:
                print("possible donate to", other)
            # find nearest lone pair position on acceptor
            target = _find_target(a, at_pos, other, not is_acc, hbond_info, finished)
            # make sure this proton doesn't form # an angle < 90 degrees
            # with any other bonds and, if trigonal planar, no angle > 150
            bad_angle = False
            chack_planar = type_info_for_atom[a].geometry == planar
            for bonded in a.neighbors:
                ang = angle(bonded._addh_coord, at_pos, target)
                if ang < 90.0:
                    bad_angle = True
                    break
                if chack_planar and ang > 150.0:
                    bad_angle = True
                    break
            if bad_angle:
                if debug:
                    print("bad angle")
                continue
            if targets and angle(targets[0], at_pos, target) < 90.0:
                if debug:
                    print("bad angle")
                continue
            targets.append(target)
            if len(targets) > 1:
                break

        hbond_info[a] = altloc_hbond_info = []
        altloc_hpos_info = []
        bonding_info = _type_info(a)
        for alt_loc in _alt_locs(a):
            a.alt_loc = alt_loc
            alt_pos = a._addh_coord
            positions = []
            for target in targets:
                vec = normalize_vector(target - alt_pos)
                positions.append(alt_pos + vec * bond_with_H_length(a, bonding_info.geometry))

            if len(positions) < 2:
                if accept_from:
                    geom = 4
                    knowns = positions + [accept_from]
                    coplanar = None
                else:
                    geom = 3
                    knowns = positions
                    coplanar = []
                    for bonded in a.neighbors:
                        for b2 in bonded.neighbors:
                            if a == b2:
                                continue
                            coplanar.append(b2._addh_coord)

                new_pos = bond_positions(alt_pos, geom, bond_with_H_length(a, geom),
                    [nb._addh_coord for nb in a.neighbors] + knowns,
                    coplanar=coplanar)
                positions.extend(new_pos)
            if accept_from:
                acc_vec = normalize_vector(accept_from - at_pos) * vdw_radius(a)
                accs = [at_pos + acc_vec]
            else:
                accs = []
            altloc_hbond_info.append((alt_loc, positions, accs))
            altloc_hpos_info.append((alt_loc, a.occupancy, positions))
        _attach_hydrogens(a, altloc_hpos_info, bonding_info)

    if debug:
        print("finished", num_finished, "of", len(candidates), "atoms")

    # shouldn't need to keep hbond_info updated after this point...
    logger.status("Using steric criteria to resolve partial h-bonders",
                                blank_after=0, secondary=True)
    for a in candidates.keys():
        if a in finished:
            continue
        if a not in hbond_info:
            continue
        finished[a] = True
        num_finished += 1

        bonding_info = _type_info(a)
        geom = bonding_info.geometry

        num_bonds = a.num_bonds
        hyds_to_position = bonding_info.substituents - num_bonds
        openings = geom - num_bonds

        altloc_hpos_info = []
        hb_info = hbond_info[a]
        toward_atom = hb_info[0][1]
        for alt_loc in _alt_locs(a):
            a.alt_loc = alt_loc
            if debug:
                print(a, "toward", toward_atom, end=" ")
            toward_pos = toward_atom._addh_coord
            toward2 = away2 = None
            at_pos = a._addh_coord
            if len(hb_info) > 1:
                toward2 = hb_info[1][1]._addh_coord
                if debug:
                    print("and toward", hb_info[1][1])
            elif openings > 3:
                # okay, we need an "away from" atom just to position the rotamer
                away2, dist, near_a = find_nearest(at_pos, a, [toward_atom], _near_dist)
                if debug:
                    print("and away from nearest other", end=" ")
                    if away2:
                        print(near_a, "[%.2f]" % dist)
                    else:
                        print("(none)")
            else:
                if debug:
                    print("with no other positioning determinant")
            bonded_pos = []
            for bonded in a.neighbors:
                bonded_pos.append(bonded._addh_coord)
            positions = bond_positions(at_pos, geom,
                bond_with_H_length(a, geom), bonded_pos,
                toward=toward_pos, toward2=toward2, away2=away2)
            altloc_hpos_info.append((alt_loc, a.occupancy, positions))
        if len(positions) == hyds_to_position:
            # easy, do them all...
            _attach_hydrogens(a, altloc_hpos_info, bonding_info)
            continue

        altloc_hpos_info = []
        for alt_loc in _alt_locs(a):
            a.alt_loc = alt_loc
            used = {}
            for is_acc, other in hb_info:
                nearest = None
                other_pos = other._addh_coord
                for pos in positions:
                    dsq = distance_squared(pos, other_pos)
                    if not nearest or dsq < nsq:
                        nearest = pos
                        nsq = dsq
                if nearest in used:
                    continue
                used[nearest] = is_acc

            remaining = []
            for pos in positions:
                if pos in used:
                    continue
                remaining.append(pos)
            # definitely protonate the positions where we donate...
            protonate = []
            for pos, is_acc in used.items():
                if not is_acc:
                    protonate.append(pos)
            # ... and the "roomiest" remaining positions.
            needed = hyds_to_position - len(protonate)
            if needed and remaining:
                rooms = roomiest(remaining, a, _room_dist, bonding_info)
                protonate.extend(rooms[:needed])
            altloc_hpos_info.append((alt_loc, a.occupancy, protonate))
        # then the least sterically challenged...
        _attach_hydrogens(a, altloc_hpos_info, bonding_info)

    if debug:
        print("finished", num_finished, "of", len(candidates), "atoms")

    logger.status("Adding hydrogens to non-h-bonding atoms", blank_after=0, secondary=True)
    for a in candidates.keys():
        if a in finished:
            continue
        if a in aro_Ns:
            continue
        finished[a] = True
        num_finished += 1

        bonding_info = _type_info(a)
        geom = bonding_info.geometry

        neighbors = a.neighbors
        num_bonds = len(neighbors)
        hyds_to_position = bonding_info.substituents - num_bonds
        openings = geom - num_bonds

        altloc_hpos_info = []
        for alt_loc in _alt_locs(a):
            a.alt_loc = alt_loc
            at_pos = a._addh_coord
            away = away2 = toward = None
            if debug:
                print("position", a, end=" ")
            if openings > 2:
                # okay, we need an "away from" atom for positioning
                #
                # if atom is tet with one bond (i.e. possible positions
                # describe a circle away from the bond), then use the
                # center of the circle as the test position rather 
                # than the atom position itself
                if geom == 4 and openings == 3:
                    away, dist, away_atom = find_rotamer_nearest(at_pos,
                            idatm_type[a], a, neighbors[0], 3.5)
                else:
                    away, dist, away_atom = find_nearest(at_pos, a, [], _near_dist)

                # actually, if the nearest atom is a metal and we have # a free lone pair,
                # we want to position (the lone pair) towards the metal
                if away_atom and away_atom.element.is_metal \
                  and geom - bonding_info.substituents > 0:
                    if debug:
                        print("towards metal", away_atom, "[%.2f]" % dist, end=" ")
                    toward = away
                    away = None
                else:
                    if debug:
                        print("away from", end=" ")
                        if away_atom:
                            print(away_atom, "[%.2f]" % dist, end=" ")
                        else:
                            print("(none)", end=" ")
            if openings > 3 and away is not None:
                # need another away from
                away2, dist, near_a = find_nearest(at_pos, a, [away_atom], _near_dist)
                if debug:
                    print("and away from", end=" ")
                    if near_a:
                        print(near_a, "[%.2f]" % dist, end=" ")
                    else:
                        print("(none)", end=" ")
            if debug:
                print()
            bonded_pos = []
            for bonded in neighbors:
                bonded_pos.append(bonded._addh_coord)
            positions = bond_positions(at_pos, geom, bond_with_H_length(a, geom),
                bonded_pos, toward=toward, away=away, away2=away2)
            altloc_hpos_info.append((alt_loc, a.occupancy, positions))
        if len(positions) == hyds_to_position:
            # easy, do them all...
            _attach_hydrogens(a, altloc_hpos_info, bonding_info)
            continue

        # protonate "roomiest" positions
        for alt_loc, occupancy, positions in altloc_hpos_info:
            a.alt_loc = alt_loc
            positions[:] = roomiest(positions, a, _room_dist, bonding_info)[:hyds_to_position]
        _attach_hydrogens(a, altloc_hpos_info, bonding_info)

    if debug:
        print("finished", num_finished, "of", len(candidates), "atoms")

    logger.status("Deciding aromatic nitrogen protonation", blank_after=0, secondary=True)
    # protonate one N of an aromatic multiple-N ring if none are yet protonated
    for ns in multi_N_rings.values():
        any_bonded = True
        Npls = []
        for n in ns:
            if n.num_bonds > 2:
                break
            if n not in coordinations and _type_info(n).substituents == 3:
                Npls.append(n)
        else:
            any_bonded = False
        if any_bonded:
            if debug:
                print(ns[0], "ring already protonated")
            continue
        if not Npls:
            Npls = ns

        # decide which N to protonate
        if len(Npls) == 1:
            prot_n = Npls[0]
            if debug:
                print("protonate precise ring", prot_n)
        else:
            positions = []
            for n in Npls:
                bonded_pos = []
                for bonded in n.neighbors:
                    bonded_pos.append(bonded._addh_coord)
                positions.append(bond_positions(n._addh_coord, planar,
                    bond_with_H_length(n, planar), bonded_pos)[0])
            prot_n = roomiest(positions, Npls, _room_dist, _type_info)[0][0]
            if debug:
                print("protonate roomiest ring", prot_n)
        altloc_hpos_info = []
        for alt_loc in _alt_locs(prot_n):
            prot_n.alt_loc = alt_loc
            bonded_pos = []
            for bonded in prot_n.neighbors:
                bonded_pos.append(bonded._addh_coord)
            positions = bond_positions(prot_n._addh_coord, planar,
                bond_with_H_length(prot_n, planar), bonded_pos)
            altloc_hpos_info.append((alt_loc, prot_n.occupancy, positions))
        _attach_hydrogens(prot_n, altloc_hpos_info, _type_info(prot_n))
    # now correct IDATM types of these rings...
    for ns in multi_N_rings.values():
        for n in ns:
            if len(n.bonds) == 3:
                n.idatm_type = "Npl"
            else:
                n.idatm_type = "N2"

    logger.status("", secondary=True)
    if problem_atoms:
        logger.error("Problems adding hydrogens to %d atom(s); see Log for details"
            % len(problem_atoms))
        logger.info("Did not protonate the following atoms:\n%s"
            % ", ".join([str(pa) for pa in problem_atoms]))
    type_info_for_atom = naming_schemas = hydrogen_totals = idatm_type \
            = aro_amines = inversion_cache = coordinations = None

def _alt_locs(atom):
    # get an alt_loc list with the current one last (so after looping
    # you are at the right alt loc)
    cur_loc = atom.alt_loc
    locs = atom.alt_locs
    if not locs:
        return [cur_loc]
    locs.remove(cur_loc)
    locs.append(cur_loc)
    return locs

def _find_target(from_atom, at_pos, to_atom, as_donor, hbond_info, finished):
    if to_atom in coordinations.get(from_atom, []):
        return to_atom._addh_coord
    if to_atom in finished:
        # known positioning already
        altloc, protons, lone_pairs = hbond_info[to_atom][-1]
        if as_donor:
            positions = lone_pairs
        else:
            positions = protons
        nearest = None
        for pos in positions:
            dsq = distance_squared(pos, at_pos)
            if not nearest or dsq < nsq:
                nearest = pos
                nsq = dsq
        return nearest
    to_h_pos = []
    to_bonded = []
    to_coplanar = []
    to_geom = _type_info(to_atom).geometry
    for tb in to_atom.neighbors:
        to_bonded.append(tb._addh_coord)
        if tb.element.number == 1:
            to_h_pos.append(tb._addh_coord)
        if to_geom == planar:
            for btb in tb.neighbors:
                if btb == to_atom:
                    continue
                to_coplanar.append(btb._addh_coord)
    if as_donor:
        # point towards nearest lone pair
        targets = bond_positions(to_atom._addh_coord, to_geom, vdw_radius(to_atom), to_bonded,
                    coplanar=to_coplanar, toward=at_pos)
    else:
        # point to nearest hydrogen
        if _type_info(to_atom).substituents <= to_atom.num_bonds:
            targets = to_h_pos
        else:
            targets = to_h_pos + bond_positions(to_atom._addh_coord, to_geom,
                bond_with_H_length(to_atom, to_geom),
                to_bonded, coplanar=to_coplanar, toward=at_pos)
    nearest = None
    for target in targets:
        dsq = distance_squared(target, at_pos)
        if not nearest or dsq < nsq:
            nearest = target
            nsq = dsq
    return nearest

def _attach_hydrogens(atom, altloc_hpos_info, bonding_info):
    total_hydrogens = hydrogen_totals[atom]
    naming_schema = (naming_schemas[atom.residue], naming_schemas[atom.structure])
    if Atom._addh_coord == Atom.scene_coord:
        try:
            invert = inversion_cache[atom.structure]
        except KeyError:
            inversion_cache[atom.structure] = invert = atom.structure.scene_position.inverse()
    else:
        invert = None
    add_altloc_hyds(atom, altloc_hpos_info, invert, bonding_info, total_hydrogens, naming_schema)

def _can_accept(donor, acceptor, protons, lone_pairs):
    # acceptor is fixed; can it accept from donor?
    if not protons:
        return True
    if not lone_pairs:
        raise ValueError("No lone pairs on %s for %s to donate to" % (acceptor, donor))
    don_pos = donor._addh_coord
    h_dist = min([distance_squared(p, don_pos) for p in protons])
    lp_dist, lp = min([(distance_squared(lp, don_pos), lp) for lp in lone_pairs])
    # besides a lone pair being closest, it must be sufficiently pointed towards the donor
    if lp_dist >= h_dist:
        if debug:
            from math import sqrt
            print("can't still accept; lp dist (%g) >= h dist (%g)" % (sqrt(lp_dist), sqrt(h_dist)))
    elif angle(lp, acceptor._addh_coord, don_pos) >= _test_angles[
            type_info_for_atom[acceptor].geometry]:
        if debug:
            print("can't still accept; angle (%g) >= test angle (%g)"
                % (angle(lp, acceptor._addh_coord, don_pos),
                _test_angles[type_info_for_atom[acceptor].geometry]))
    return lp_dist < h_dist and angle(lp, acceptor._addh_coord, don_pos) < _test_angles[type_info_for_atom[acceptor].geometry]

def _can_donate(donor, acceptor, no_protons_ok, protons, lone_pairs):
    # donor is fixed; can it donate to acceptor?
    if not lone_pairs:
        return True
    if not protons:
        if no_protons_ok:
            if debug:
                print("can't still donate; no protons")
            return False
        raise ValueError("No protons for %s to accept from" % acceptor)
    acc_pos = acceptor._addh_coord
    h_dist, h = min([(distance_squared(xyz, acc_pos), xyz) for xyz in protons])
    lp_dist = min([distance_squared(xyz, acc_pos) for xyz in lone_pairs])
    # besides a proton being closest, it must be sufficiently pointed towards the acceptor
    if h_dist >= lp_dist:
        if debug:
            from math import sqrt
            print("can't still donate; h dist (%g) >= lp dist (%g)" % (sqrt(h_dist), sqrt(lp_dist)))
    elif angle(h, donor._addh_coord, acc_pos) >= _test_angles[type_info_for_atom[donor].geometry]:
        if debug:
            print("can't still donate; angle (%g) >= test angle (%g)" % (angle(h, donor._addh_coord, acc_pos), _test_angles[ type_info_for_atom[donor].geometry]))
    return h_dist < lp_dist and angle(h, donor._addh_coord, acc_pos) < _test_angles[
            type_info_for_atom[donor].geometry]

def _try_finish(atom, hbond_info, finished, aro_amines, pruned_by, processed):
    # do we have enough info to establish all H/LP positions for atom?

    bonding_info = _type_info(atom)
    geom = bonding_info.geometry

    # from number of donors/acceptors, determine
    # if we can position Hs/lone pairs
    num_bonds = atom.num_bonds
    hyds_to_position = bonding_info.substituents - num_bonds
    openings = geom - num_bonds

    donors = []
    acceptors = []
    all = []
    for is_acc, other in hbond_info[atom]:
        all.append(other)
        if is_acc:
            donors.append(other)
        else:
            acceptors.append(other)
    if len(all) < openings \
    and len(donors) < openings - hyds_to_position \
    and len(acceptors) < hyds_to_position:
        if debug:
            print("not enough info (all/donors/acceptors):", len(all), len(donors), len(acceptors))
        return False

    # if so, find their positions and record in hbond_info; mark as finished
    at_pos = atom._addh_coord
    targets = []
    for is_acc, other in hbond_info[atom][:2]:
        targets.append(_find_target(atom, at_pos, other, not is_acc, hbond_info, finished))

    # for purposes of this intermediate measurement, use hydrogen distances instead
    # of lone pair distances; determine true lone pair positions once hydrogens are found
    bonded_pos = []
    test_positions = []
    coplanar = []
    for bonded in atom.neighbors:
        bonded_pos.append(bonded._addh_coord)
        if bonded.element.number == 1:
            test_positions.append((True, bonded._addh_coord))
        if geom == planar:
            for btb in bonded.neighbors:
                if btb == atom:
                    continue
                coplanar.append(btb._addh_coord)
    toward = targets[0]
    if len(targets) > 1:
        toward2 = targets[1]
    else:
        toward2 = None
    h_len = bond_with_H_length(atom, geom)
    lp_len = vdw_radius(atom)
    if debug:
        print(atom, "co-planar:", coplanar)
        print(atom, "toward:", toward)
        print(atom, "toward2:", toward2)
    normals = bond_positions(at_pos, geom, 1.0, bonded_pos,
            coplanar=coplanar, toward=toward, toward2=toward2)
    if debug:
        print(atom, "bond_positions:", [str(x) for x in normals])
    # use vectors so we can switch between lone-pair length and H-length
    for normal in normals:
        test_positions.append((False, normal - at_pos))

    # try to hook up positions with acceptors/donors
    if atom in aro_amines:
        if debug:
            print("delay finishing aromatic amine")
        return False
    all = {}
    protons = {}
    lone_pairs = {}
    conflicting = []
    for is_acc, other in hbond_info[atom]:
        if debug:
            print("other:", other)
        if is_acc:
            key = (other, atom)
        else:
            key = (atom, other)
        conflict_allowable = key not in processed
        nearest = None
        if other in finished:
            oprotons, olps = hbond_info[other][-1][1:]
            if is_acc:
                # proton may have been stripped if donor near metal...
                if not oprotons:
                    continue
                opositions = oprotons
                mul = lp_len
            else:
                opositions = olps
                mul = h_len
            for opos in opositions:
                for is_h, check in test_positions:
                    if is_h:
                        pos = check
                    else:
                        pos = at_pos + check * mul
                    dsq = distance_squared(opos, pos)
                    if nearest is None or dsq < nsq:
                        nearest = check
                        nearest_is_h = is_h
                        nsq = dsq
        else:
            other_pos = other._addh_coord
            if is_acc:
                mul = lp_len
            else:
                mul = h_len
            for is_h, check in test_positions:
                if is_h:
                    pos = check
                else:
                    pos = at_pos + check * mul
                dsq = distance_squared(pos, other_pos)
                if debug:
                    print("dist from other to", end=" ")
                    if is_h:
                        if check in all:
                            if check in protons:
                                print("new proton:", end=" ")
                            else:
                                print("new lone pair:", end=" ")
                        else:
                            print("unfilled position:", end=" ")
                    else:
                        print("pre-existing proton:", end=" ")
                    import math
                    print(math.sqrt(dsq))
                if nearest is None or dsq < nsq:
                    nearest = check
                    nearest_is_h = is_h
                    nsq = dsq
        if nearest and nearest_is_h:
            # closest to known hydrogen; no help in positioning...
            if is_acc and conflict_allowable:
                # other is trying to donate and is nearest to one of our hydrogens
                conflicting.append((is_acc, other))
            continue
        if nearest in all:
            if is_acc:
                if nearest in protons and conflict_allowable:
                    conflicting.append((is_acc, other))
            elif nearest in lone_pairs and conflict_allowable:
                conflicting.append((is_acc, other))
            continue
        # check for steric conflict (frequent with metal coordination)
        if is_acc:
            pos = at_pos + nearest * lp_len
            at_bump = 0.0
        else:
            pos = at_pos + nearest * h_len
            at_bump = h_rad
        check_dist = 2.19 + at_bump
        # since searchTree is a module variable that changes need to access via the module...
        from .cmd import search_tree
        nearby = search_tree.search(pos, check_dist)
        steric_clash = False
        okay = set([atom, other])
        okay.update(atom.neighbors)
        for nb in nearby:
            if nb in okay:
                continue
            if nb.structure != atom.structure and (
                    atom.structure.id is None or (
                        len(nb.structure.id) > 1
                        and (nb.structure.id[:-1] == atom.structure.id[:-1]))):
                # ignore clashes with sibling submodels or if our model isn't open
                continue
            d_chk = vdw_radius(nb) + at_bump - 0.4
            if d_chk * d_chk >= distance_squared(nb._addh_coord, pos):
                steric_clash = True
                if debug:
                    print("steric clash with", nb, "(%.3f < %.3f)"
                        % (distance(pos, nb._addh_coord), d_chk))
                break
        if steric_clash and conflict_allowable:
            conflicting.append((is_acc, other))
            continue

        all[nearest] = 1
        if is_acc:
            if debug:
                print("determined lone pair")
            lone_pairs[nearest] = 1
        else:
            if debug:
                print("determined proton")
            protons[nearest] = 1
        if len(protons) == hyds_to_position:
            # it is possible that an earlier call to this routine had protons in
            # the same position and therefore decided that the full set of proton
            # positions hadn't been determined, but between that call and a later
            # call one of the H-bond partners became fully determined and caused
            # the "nearest" proton to switch to a different position, so that by
            # the time this routine is called again, you don't need the extra
            # H-bond, so this check is to prevent adding extra protons!  Example
            # is /H:2081 (water) in 2c9t
            break

    for is_acc, other in conflicting:
        if debug:
            print("Removing hbond to %s due to conflict" % other)
        hbond_info[atom].remove((is_acc, other))
        if not hbond_info[atom]:
            del hbond_info[atom]
        if other in finished:
            continue
        try:
            hbond_info[other].remove((not is_acc, atom))
            if not hbond_info[other]:
                del hbond_info[other]
        except ValueError:
            pass
    # since any conflicting hbonds may have been used to determine positions,
    # determine the positions again with the remaining hbonds
    if conflicting:
        # restore hbonds pruned by the conflicting hbonds
        for is_acc, other in conflicting:
            if is_acc:
                key = (other, atom)
            else:
                key = (atom, other)
            if key in pruned_by:
                for phb in pruned_by[key]:
                    if phb[0] in finished or phb[1] in finished:
                        continue
                    if debug:
                        print("restoring %s/%s hbond pruned by hbond to %s"
                            % (phb[0], phb[1], other))
                    processed.remove(phb)
                    hbond_info.setdefault(phb[0], []).append((False, phb[1]))
                    hbond_info.setdefault(phb[1], []).append((True, phb[0]))
                del pruned_by[key]
        if atom not in hbond_info:
            if debug:
                print("No non-conflicting hbonds left!")
            return False
        if debug:
            print("calling _try_finish with non-conflicting hbonds")
        return _try_finish(atom, hbond_info, finished, aro_amines, pruned_by, processed)
    # did we determine enough positions?
    if len(all) < openings and len(protons) < hyds_to_position \
    and len(lone_pairs) < openings - hyds_to_position:
        if debug:
            print("not enough hookups (all/protons/lps):", len(all), len(protons), len(lone_pairs))
        return False

    if len(protons) < hyds_to_position:
        for is_h, pos in test_positions:
            if is_h:
                continue
            if pos not in all:
                protons[pos] = 1
    hbond_info[atom] = altloc_info = []
    for alt_loc in _alt_locs(atom):
        atom.alt_loc = alt_loc
        at_pos = atom._addh_coord

        h_locs = []
        for h_vec in protons.keys():
            h_locs.append(at_pos + h_vec * h_len)

        lp_locs = []
        for is_h, vec in test_positions:
            if is_h:
                continue
            if vec not in protons:
                lp_locs.append(at_pos + vec * lp_len)

        altloc_info.append((alt_loc, h_locs, lp_locs))
    finished[atom] = True
    return True

def _angle_check(d, a, hbond_info):
    add_A_to_D = add_D_to_A = True
    # are the protons/lps already added to the donor pointed toward the acceptor?
    if d in hbond_info:
        geom = _type_info(d).geometry
        for is_acc, da in hbond_info[d]:
            ang = angle(da._addh_coord, d._addh_coord, a._addh_coord)
            if ang > _test_angles[geom]:
                continue
            if is_acc:
                # lone pair pointing toward acceptor;
                # won't work
                add_A_to_D = add_D_to_A = False
                if debug:
                    print("can't donate; lone pair (to %s) pointing toward acceptor (angle %g)"
                        % (da, ang))
                break
            add_A_to_D = False
            if debug:
                print("donor already pointing (angle %g) towards acceptor (due to %s)" % (ang, da))
    if not add_A_to_D and not add_D_to_A:
        return add_A_to_D, add_D_to_A
    if a in hbond_info:
        geom = _type_info(a).geometry
        for is_acc, da in hbond_info[a]:
            ang = angle(da._addh_coord, a._addh_coord, d._addh_coord)
            if ang > _test_angles[geom]:
                continue
            if not is_acc:
                # proton pointing toward donor; won't work
                if debug:
                    print("can't accept; proton (to %s) pointing too much toward donor (angle %g)"
                        % (da, ang))
                add_A_to_D = add_D_to_A = False
                break
            add_D_to_A = False
            if debug:
                print("acceptor already pointing too much (angle %g) towards donor (due to %s)"
                    % (ang, da))
    return add_A_to_D, add_D_to_A

def _prune_check(pivot_atom, gold_hbond, test_hbond):
    geom = _type_info(pivot_atom).geometry
    if geom < 2:
        return False
    if geom < 4 and pivot_atom.num_bonds > 0:
        return False
    if gold_hbond[0] == pivot_atom:
        ga = gold_hbond[1]
    else:
        ga = gold_hbond[0]
    if test_hbond[0] == pivot_atom:
        ta = test_hbond[1]
    else:
        ta = test_hbond[0]
    ang = angle(ga._addh_coord, pivot_atom._addh_coord, ta._addh_coord)
    full_angle = _test_angles[geom] * 3
    while ang > full_angle / 2.0:
        ang -= full_angle
    ang = abs(ang)
    return ang > full_angle / 4.0

def _resolve_anilene(donor, acceptor, aro_amines, hbond_info):
    # donor/acceptor are currently ambiguous;  if donor and/or acceptor are
    # anilenes, see if they can be determined to prefer to donate/accept
    # respectively (if a proton and/or lone pair has been added, see if
    # vector to other atom is more planar or more tetrahedral)

    if donor in aro_amines and donor in hbond_info:
        toward = None
        for is_acc, da in hbond_info[donor]:
            if is_acc:
                return True
            if toward:
                break
            toward = da._addh_coord
        else:
            # one proton attached
            donor_pos = donor._addh_coord
            acceptor_pos = acceptor._addh_coord
            attached = [donor.neighbors[0]._addh_coord]
            planars = bond_positions(donor_pos, 3, N_H, attached, toward=toward)
            planar_dist = None
            for planar in planars:
                dist = distance_squared(acceptor_pos, planar)
                if planar_dist is None or dist < planar_dist:
                    planar_dist = dist

            for tet_pos in bond_positions(donor_pos, 4, N_H, attached):
                if distance_squared(tet_pos, acceptor_pos) < planar_dist:
                    # closer to tet position, prefer acceptor-like behavior
                    return False
            if debug:
                print("resolving", donor, "->", acceptor, "because of donor")
            return True

    if acceptor in aro_amines and acceptor in hbond_info:
        toward = None
        for is_acc, da in hbond_info[acceptor]:
            if is_acc:
                return False
            if toward:
                break
            toward = da._addh_coord
        else:
            # one proton attached
            donor_pos = donor._addh_coord
            acceptor_pos = acceptor._addh_coord
            attached = [acceptor.neighbors[0]._addh_coord]
            planars = bond_positions(acceptor_pos, 3, N_H, attached, toward=toward)
            planar_dist = None
            for planar in planars:
                dist = distance_squared(acceptor_pos, planar)
                if planar_dist is None or dist < planar_dist:
                    planar_dist = dist

            for tet_pos in bond_positions(acceptor_pos, 4, N_H, attached):
                if distance_squared(tet_pos, donor_pos) < planar_dist:
                    # closer to tet position, prefer acceptor-like behavior
                    if debug:
                        print("resolving", donor, "->", acceptor, "because of acceptor")
                    return True
            return False
    return False

def _type_info(atom):
    if atom in aro_amines:
        return idatm.type_info['N3']
    return type_info_for_atom[atom]

def _tet_check(tet, partner, hbond_info, tet_acc):
    """Check if tet can still work"""
    tet_pos = tet._addh_coord
    partner_pos = partner._addh_coord
    bonded = tet.neighbors
    tet_info = hbond_info.get(tet, [])
    towards = []
    # if there is a real bond to the tet, we want to check the dihedral to the new position
    # (using a 120 angle) rather than the angle (109.5) since the vector from the tet to the
    # not-yet-added positions is probably not directly along the future bond
    if bonded:
        if debug:
            print("checking dihedral")
        chk_func = lambda op, pp=partner_pos, tp=tet_pos, \
            bp=bonded[0]._addh_coord: dihedral(pp, tp, bp, op) / 30.0
    else:
        if debug:
            print("checking angle")
        chk_func = lambda op, pp=partner_pos, tp=tet_pos: angle(pp, tp, op) / 27.375
    for is_acc, other in tet_info:
        if is_acc == tet_acc:
            # same "polarity"; pointing towards is good
            result = True
        else:
            # opposite "polarity"; pointing towards is bad
            result = False
        angle_group = int(chk_func(other._addh_coord))
        if angle_group in [1,2,5,6]:
            # in the tetrahedral "dead zones"
            if debug:
                print("tetCheck for %s; dead zone for %s" % (tet, other))
            return False
        if angle_group == 0:
            if debug:
                print("tetCheck for %s; pointing towards %s returning %s" % (tet, other, result))
            return result
        towards.append(other._addh_coord)
    # further tests only for 2 of 4 filled...
    if bonded or len(towards) != 2:
        return True

    return _tet2_check(tet_pos, towards[0], towards[1], partner_pos)

def _tet2_check(tet_pos, toward, toward2, partner_pos):
    if debug:
        print("2 position tetCheck", end="")
    for pos in bond_positions(tet_pos, 4, 1.0, [], toward=toward, toward2=toward2):
        if debug:
            print(angle(partner_pos, tet_pos, pos), end="")
        if angle(partner_pos, tet_pos, pos) < _test_angles[4]:
            if debug:
                print("true")
            return True
    if debug:
        print("false")
    return False

def _do_prune(hbond, pruned, rel_bond, processed, pruned_by):
    # prune later hbonds that conflict
    pruned.add(hbond)
    for da in hbond:
        skipping = True
        prev = []
        for rel in rel_bond[da]:
            if rel in pruned:
                other = da == rel[0] and rel[1] or rel[0]
                if other not in prev:
                    prev.append(other)
            if skipping:
                if rel == hbond:
                    skipping = False
                continue
            if rel in processed:
                continue
            if _prune_check(da, hbond, rel):
                if debug:
                    print("pruned hbond ", [str(a)for a in rel])
                processed.add(rel)
                pruned_by.setdefault(hbond, []).append(rel)
                continue
        if len(prev) != 2 or _type_info(da).geometry != 4:
            continue
        # prune based on 2-position tet check
        skipping = True
        for rel in rel_bond[da]:
            if skipping:
                if rel == hbond:
                    skipping = False
                continue
            if rel in processed:
                continue
            other = da == rel[0] and rel[1] or rel[0]
            if not _tet2_check(da._addh_coord, prev[0]._addh_coord, prev[1]._addh_coord,
                            other._addh_coord):
                if debug:
                    print("pruned hbond (tet check)", [str(a) for a in rel])
                processed.add(rel)
                pruned_by.setdefault(hbond, []).append(rel)
