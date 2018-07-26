# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2016 Regents of the University of California.
# All rights reserved.  This software provided pursuant to a
# license agreement containing restrictions on its disclosure,
# duplication and use.  For details see:
# http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
# This notice must be embedded in or attached to all copies,
# including partial copies, of the software or any revisions
# or derivations thereof.
# === UCSF ChimeraX Copyright ===

from chimerax.atomic.bond_geom import tetrahedral, planar, linear, single, bond_positions
from chimerax.atomic import Atom, idatm
from .cmd import new_hydrogen, find_nearest, roomiest, _tree_dist, vdw_radius, \
                bond_with_H_length, N_H, find_rotamer_nearest, h_rad, add_altloc_hyds
from chimerax.core.geometry import distance_squared, angle, dihedral

c_rad = 1.7
_near_dist = _room_dist = _tree_dist + h_rad + c_rad

_test_angles = [None, None, 60.0, 40.0, 36.5]

debug = True

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
        logger.error("Adding H-bond-preserving hydrogens to trajectories not supported.")
        return
    if debug:
        print("here 2")
    global add_atoms, type_info_for_atom, naming_schemas, hydrogen_totals, \
                    idatm_type, inversion_cache, coordinations
    type_info_for_atom, naming_schemas, hydrogen_totals, idatm_type, his_Ns, \
                            coordinations, in_isolation = args
    inversion_cache = {}
    add_atoms = {}
    for atom in atom_list:
        add_atoms[atom] = 1
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
    for atom in atom_list:
        bonding_info = type_info_for_atom[atom]
        num_bonds = atom.num_bonds
        substs = bonding_info.substituents
        geom = bonding_info.geometry
        if num_bonds >= substs:
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

    logger.status("Building search tree of atom positions", blank_after=0, secondary=True)
    # since adaptive search tree is static, it will not include hydrogens added after
    # this; they will have to be found by looking off their heavy atoms

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
            _attach_hydrogens(atom, altloc_hpos_info, bonding_info)
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
        from chimerax.core.atomic import AtomicStructure
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
            if in_isolation and a not in acceptors:
                continue
            sortable_hbonds.append((distance_squared(d._hb_coord, a._hb_coord), False, (d, a)))
        if a in acceptors:
            if in_isolation and d not in donors:
                continue
            sortable_hbonds.append((distance_squared(d._hb_coord, a._hb_coord), True, (d, a)))
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
            #TODO
            """
            if a in finished:
                # can d still donate to a?
                if not _can_accept(d, a, *hbond_info[a]):
                    # can't
                    continue
                hbond_info.setdefault(d, []).append((False, a))
            elif d in finished:
                # can d still donate to a?
                noProtonsOK = d in aro_Ns and d.num_bonds == 2
                if not _canDonate(d, a, noProtonsOK,
                                *hbond_info[d]):
                    # nope
                    continue
                hbond_info.setdefault(a, []).append((True, d))
            else:
                addAtoD, addDtoA = _angle_check(d, a,
                                hbond_info)
                if not addAtoD and not addDtoA:
                    continue
                if addAtoD:
                    hbond_info.setdefault(d, []).append(
                                (False, a))
                if addDtoA:
                    hbond_info.setdefault(a, []).append(
                                (True, d))
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
                        print "no hbonds left for", end.oslIdent()
                    continue
                if debug:
                    print "try to finish", end.oslIdent(),
                    for is_acc, da in hbond_info[end]:
                        if is_acc:
                            print " accepting from", da.oslIdent(),
                        else:
                            print " donating to", da.oslIdent(),
                    print
                didFinish = _tryFinish(end, hbond_info, finished,
                        aro_amines, pruned_by, processed)
                if not didFinish:
                    continue
                if debug:
                    print "finished", end.oslIdent()

                # if this atom is in candidates 
                # then actually add any hydrogens
                if end in candidates and hbond_info[end][0]:
                    _attach_hydrogens(end, hbond_info[end][0])
                    if debug:
                        print "protonated", end.oslIdent()
                    # if ring nitrogen, eliminate
                    # any hbonds where it is an acceptor
                    if isinstance(donors[end],chimera.Ring):
                        for rb in rel_bond[end]:
                            if rb[1] != end:
                                continue
                            processed.add(rb)
                                
            if a in seen_ambiguous or d in seen_ambiguous:
                # revisit previously ambiguous
                if debug:
                    print "revisiting previous ambiguous"
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
    numFinished = 0
    for a in candidates.keys():
        if a in finished:
            numFinished += 1
    if debug:
        print "finished", numFinished, "of", len(candidates), "atoms"

    logger.status("Adding hydrogens to primary aromatic amines",
                                blank_after=0, secondary=True)
    if debug:
        print "primary aromatic amines"
    for a in aro_amines:
        if a in finished:
            continue
        if a not in candidates:
            continue
        if debug:
            print "amine", a.oslIdent()
        finished[a] = True
        numFinished += 1
        
        # point protons right toward acceptors;
        # if also accepting, finish as tet, otherwise as planar
        acceptFrom = None
        targets = []
        atPos = a._hb_coord
        for is_acc, other in hbond_info.get(a, []):
            if is_acc:
                if not acceptFrom:
                    if debug:
                        print "accept from", other.oslIdent()
                    acceptFrom = other._hb_coord
                continue
            if debug:
                print "possible donate to", other.oslIdent()
            # find nearest lone pair position on acceptor
            target = _findTarget(a, atPos, other, not is_acc,
                            hbond_info, finished)
            # make sure this proton doesn't form
            # an angle < 90 degrees with any other bonds
            # and, if trigonal planar, no angle > 150
            badAngle = False
            checkPlanar = type_info_for_atom[a].geometry == planar
            for bonded in a.primaryNeighbors():
                ang = chimera.angle(bonded._hb_coord, atPos, target)
                if ang < 90.0:
                    badAngle = True
                    break
                if checkPlanar and ang > 150.0:
                    badAngle = True
                    break
            if badAngle:
                if debug:
                    print "bad angle"
                continue
            if targets and chimera.angle(targets[0], atPos,
                                target) < 90.0:
                if debug:
                    print "bad angle"
                continue
            targets.append(target)
            if len(targets) > 1:
                break

        positions = []
        for target in targets:
            vec = target - atPos
            vec.normalize()
            positions.append(atPos +
                    vec * bond_with_H_length(a, _type_info(a).geometry))

        if len(positions) < 2:
            if acceptFrom:
                geom = 4
                knowns = positions + [acceptFrom]
                coPlanar = None
            else:
                geom = 3
                knowns = positions
                coPlanar = []
                for bonded in a.primaryNeighbors():
                    for b2 in bonded.primaryNeighbors():
                        if a == b2:
                            continue
                        coPlanar.append(b2._hb_coord)
            sumVec = chimera.Vector()
            for x in a.primaryNeighbors():
                vec = x._hb_coord - atPos
                vec.normalize()
                sumVec += vec
            for k in knowns:
                vec = k - atPos
                vec.normalize()
                sumVec += vec
            sumVec.negate()
            sumVec.normalize()

            newPos = bond_positions(atPos, geom, bond_with_H_length(a, geom),
                [nb._hb_coord for nb in a.primaryNeighbors()] + knowns,
                coPlanar=coPlanar)
            positions.extend(newPos)
        _attach_hydrogens(a, positions)
        if acceptFrom:
            accVec = acceptFrom - atPos
            accVec.length = vdw_radius(a)
            accs = [atPos + accVec]
        else:
            accs = []
        hbond_info[a] = (positions, accs)

    if debug:
        print "finished", numFinished, "of", len(candidates), "atoms"

    logger.status("Using steric criteria to resolve partial h-bonders",
                                blank_after=0, secondary=True)
    for a in candidates.keys():
        if a in finished:
            continue
        if a not in hbond_info:
            continue
        finished[a] = True
        numFinished += 1

        bonding_info = _type_info(a)
        geom = bonding_info.geometry

        num_bonds = a.num_bonds
        hydsToPosition = bonding_info.substituents - num_bonds
        openings = geom - num_bonds

        hbInfo = hbond_info[a]
        towardAtom = hbInfo[0][1]
        if debug:
            print a.oslIdent(), "toward", towardAtom.oslIdent(),
        towardPos = towardAtom._hb_coord
        toward2 = away2 = None
        atPos = a._hb_coord
        if len(hbInfo) > 1:
            toward2 = hbInfo[1][1]._hb_coord
            if debug:
                print "and toward", hbInfo[1][1].oslIdent()
        elif openings > 3:
            # okay, we need an "away from" atom just to position
            # the rotamer
            away2, dist, nearA = find_nearest(atPos, a, [towardAtom], _near_dist)
            if debug:
                print "and away from nearest other",
                if away2:
                    print nearA.oslIdent(), "[%.2f]" % dist
                else:
                    print "(none)"
        else:
            if debug:
                print "with no other positioning determinant"
        bondedPos = []
        for bonded in a.primaryNeighbors():
            bondedPos.append(bonded._hb_coord)
        positions = bond_positions(atPos, geom,
            bond_with_H_length(a, geom), bondedPos,
            toward=towardPos, toward2=toward2, away2=away2)
        if len(positions) == hydsToPosition:
            # easy, do them all...
            _attach_hydrogens(a, positions)
            continue

        used = {}
        for is_acc, other in hbInfo:
            nearest = None
            otherPos = other._hb_coord
            for pos in positions:
                dsq = (pos - otherPos).sqlength()
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
        rooms = roomiest(remaining, a, _room_dist, _type_info(a))
        needed = hydsToPosition - len(protonate)
        protonate.extend(rooms[:needed])
        # then the least sterically challenged...
        _attach_hydrogens(a, protonate)
        hbond_info[a] = (protonate, rooms[needed:])

    if debug:
        print "finished", numFinished, "of", len(candidates), "atoms"

    logger.status("Adding hydrogens to non-h-bonding atoms", blank_after=0, secondary=True)
    for a in candidates.keys():
        if a in finished:
            continue
        if a in aro_Ns:
            continue
        finished[a] = True
        numFinished += 1

        bonding_info = _type_info(a)
        geom = bonding_info.geometry

        primaryNeighbors = a.primaryNeighbors()
        num_bonds = len(primaryNeighbors)
        hydsToPosition = bonding_info.substituents - num_bonds
        openings = geom - num_bonds

        away = None
        away2 = None
        toward = None
        atPos = a._hb_coord
        if debug:
            print "position", a.oslIdent(),
        if openings > 2:
            # okay, we need an "away from" atom for positioning
            #
            # if atom is tet with one bond (i.e. possible positions
            # describe a circle away from the bond), then use the
            # center of the circle as the test position rather 
            # than the atom position itself
            if geom == 4 and openings == 3:
                away, dist, awayAtom = find_rotamer_nearest(atPos,
                        idatm_type[a], a,
                        primaryNeighbors[0], 3.5)
            else:
                away, dist, awayAtom = find_nearest(atPos, a, [], _near_dist)

            # actually, if the nearest atom is a metal and we have
            # a free lone pair, we want to position (the lone
            # pair) towards the metal
            if awayAtom and awayAtom.element.is_metal \
              and geom - bonding_info.substituents > 0:
                if debug:
                    print "towards metal", awayAtom.oslIdent(), "[%.2f]" % dist,
                toward = away
                away = None
            else:
                if debug:
                    print "away from",
                    if awayAtom:
                        print awayAtom.oslIdent(), "[%.2f]" % dist,
                    else:
                        print "(none)",
        if openings > 3 and away is not None:
            # need another away from
            away2, dist, nearA = find_nearest(atPos, a, [awayAtom], _near_dist)
            if debug:
                print "and away from",
                if nearA:
                    print nearA.oslIdent(), "[%.2f]" % dist,
                else:
                    print "(none)",
        if debug:
            print
        bondedPos = []
        for bonded in primaryNeighbors:
            bondedPos.append(bonded._hb_coord)
        positions = bond_positions(atPos, geom,
            bond_with_H_length(a, geom),
            bondedPos, toward=toward, away=away, away2=away2)
        if len(positions) == hydsToPosition:
            # easy, do them all...
            _attach_hydrogens(a, positions)
            continue

        # protonate "roomiest" positions
        _attach_hydrogens(a, roomiest(positions, a, _room_dist, _type_info(a))[
                    :hydsToPosition])

    if debug:
        print "finished", numFinished, "of", len(candidates), "atoms"

    logger.status("Deciding aromatic nitrogen protonation",
                                blank_after=0, secondary=True)
    # protonate one N of an aromatic multiple-N ring if none are yet
    # protonated
    for ns in multi_N_rings.values():
        anyBonded = True
        Npls = []
        for n in ns:
            if n.num_bonds > 2:
                break
            if n not in coordinations \
                    and _type_info(n).substituents == 3:
                Npls.append(n)
        else:
            anyBonded = False
        if anyBonded:
            if debug:
                print ns[0].oslIdent(), "ring already protonated"
            continue
        if not Npls:
            Npls = ns

        positions = []
        for n in Npls:
            bondedPos = []
            for bonded in n.primaryNeighbors():
                bondedPos.append(bonded._hb_coord)
            positions.append(bond_positions(n._hb_coord, planar,
                bond_with_H_length(n, planar), bondedPos)[0])
        if len(positions) == 1:
            if debug:
                print "protonate precise ring", Npls[0].oslIdent()
            _attach_hydrogens(Npls[0], positions)
        else:
            if debug:
                print "protonate roomiest ring", roomiest(positions, Npls, _room_dist, _type_info)[0][0].oslIdent()
            _attach_hydrogens(*roomiest(positions, Npls, _room_dist, _type_info)[0])
    # now correct IDATM types of these rings...
    for ns in multi_N_rings.values():
        for n in ns:
            if len(n.bonds) == 3:
                n.idatm_type = "Npl"
            else:
                n.idatm_type = "N2"

    logger.status("", secondary=True)
    if problem_atoms:
        logger.error("Problems adding hydrogens to %d atom(s); see Reply Log for details" % len(problem_atoms))
        from chimera.misc import chimeraLabel
        logger.info("Did not protonate following atoms:\n%s"
            % "\t".join(map(chimeraLabel, problem_atoms)))
    add_atoms = None
    type_info_for_atom = naming_schemas = hydrogen_totals = idatm_type \
            = aro_amines = inversion_cache = coordinations = None
    """

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

'''
def _findTarget(fromAtom, atPos, toAtom, asDonor, hbond_info, finished):
    if toAtom in coordinations.get(fromAtom, []):
        return toAtom._hb_coord
    if toAtom in finished:
        toInfo = hbond_info[toAtom]
        # known positioning already
        protons, lonePairs = toInfo
        if asDonor:
            positions = lonePairs
        else:
            positions = protons
        nearest = None
        for pos in positions:
            dsq = (pos - atPos).sqlength()
            if not nearest or dsq < nsq:
                nearest = pos
                nsq = dsq
        return nearest
    toHPos = []
    toBonded = []
    toCoplanar = []
    toGeom = _type_info(toAtom).geometry
    for tb in toAtom.primaryNeighbors():
        toBonded.append(tb._hb_coord)
        if tb.element.number == 1:
            toHPos.append(tb._hb_coord)
        if toGeom == planar:
            toAtomLocs = toAtom.allLocations()
            for btb in tb.primaryNeighbors():
                if btb in toAtomLocs:
                    continue
                toCoplanar.append(btb._hb_coord)
    if asDonor:
        # point towards nearest lone pair
        targets = bond_positions(toAtom._hb_coord, toGeom,
                    vdw_radius(toAtom), toBonded,
                    coPlanar=toCoplanar, toward=atPos)
    else:
        # point to nearest hydrogen
        if _type_info(toAtom).substituents <= toAtom.num_bonds:
            targets = toHPos
        else:
            targets = toHPos + bond_positions(
                toAtom._hb_coord, toGeom,
                bond_with_H_length(toAtom, toGeom),
                toBonded, coPlanar=toCoplanar, toward=atPos)
    nearest = None
    for target in targets:
        dsq = (target - atPos).sqlength()
        if not nearest or dsq < nsq:
            nearest = target
            nsq = dsq
    return nearest
'''

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
    don_pos = donor._hb_coord
    h_dist = min([distance_squared(p, don_pos) for p in protons])
    lp_dist, lp = min([distance_squared(lp, don_pos) for lp in lone_pairs])
    # besides a lone pair being closest, it must be sufficiently pointed towards the donor
    if lp_dist >= h_dist:
        if debug:
            from math import sqrt
            print("can't still accept; lp dist (%g) >= h dist (%g)" % (sqrt(lp_dist), sqrt(h_dist)))
    elif chimera.angle(lp, acceptor._hb_coord, don_pos) >= _test_angles[
            type_info_for_atom[acceptor].geometry]:
        if debug:
            print("can't still accept; angle (%g) >= test angle (%g)"
                % (angle(lp, acceptor._hb_coord, don_pos),
                _test_angles[type_info_for_atom[acceptor].geometry]))
    return lp_dist < h_dist and angle(lp, acceptor._hb_coord, don_pos) < _test_angles[type_info_for_atom[acceptor].geometry]

'''
def _canDonate(donor, acceptor, noProtonsOK, protons, lonePairs):
    # donor is fixed; can it donate to acceptor?
    if not lonePairs:
        return True
    if not protons:
        if noProtonsOK:
            if debug:
                print "can't still donate; no protons"
            return False
        raise ValueError, "No protons for %s to accept from" % (
                            acceptor.oslIdent())
    accPos = acceptor._hb_coord
    hDist, h = min(map(lambda xyz: ((xyz - accPos).sqlength(), xyz),
                                protons))
    lpDist = min(map(lambda xyz: (xyz - accPos).sqlength(), lonePairs))
    # besides a proton being closest, it must be sufficiently pointed
    # towards the acceptor
    if hDist >= lpDist:
        from math import sqrt
        if debug:
            print "can't still donate; h dist (%g) >= lp dist (%g)" % ( sqrt(hDist), sqrt(lpDist))
    elif chimera.angle(h, donor._hb_coord, accPos) >= _test_angles[
            type_info_for_atom[donor].geometry]:
        if debug:
            print "can't still donate; angle (%g) >= test angle (%g)" % ( chimera.angle(h, donor._hb_coord, accPos), _test_angles[ type_info_for_atom[donor].geometry])
    return hDist < lpDist and chimera.angle(h,
            donor._hb_coord, accPos) < _test_angles[
            type_info_for_atom[donor].geometry]

def _tryFinish(atom, hbond_info, finished, aro_amines, pruned_by, processed):
    # do we have enough info to establish all H/LP positions for atom?

    bonding_info = _type_info(atom)
    geom = bonding_info.geometry

    # from number of donors/acceptors, determine
    # if we can position Hs/lone pairs
    num_bonds = atom.num_bonds
    hydsToPosition = bonding_info.substituents - num_bonds
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
    and len(donors) < openings - hydsToPosition \
    and len(acceptors) < hydsToPosition:
        if debug:
            print "not enough info (all/donors/acceptors):", len(all), len(donors), len(acceptors)
        return False

    # if so, find their positions and
    # record in hbond_info; mark as finished
    atPos = atom._hb_coord
    targets = []
    for is_acc, other in hbond_info[atom][:2]:
        targets.append(_findTarget(atom, atPos, other, not is_acc,
                            hbond_info, finished))

    # for purposes of this intermediate measurement, use hydrogen
    # distances instead of lone pair distances; determine true
    # lone pair positions once hydrogens are found
    bondedPos = []
    testPositions = []
    coplanar = []
    for bonded in atom.primaryNeighbors():
        bondedPos.append(bonded._hb_coord)
        if bonded.element.number == 1:
            testPositions.append(bonded._hb_coord)
        if geom == planar:
            for btb in bonded.primaryNeighbors():
                if btb == atom:
                    continue
                coplanar.append(btb._hb_coord)
    toward = targets[0]
    if len(targets) > 1:
        toward2 = targets[1]
    else:
        toward2 = None
    Hlen = bond_with_H_length(atom, geom)
    LPlen = vdw_radius(atom)
    if debug:
        print atom.oslIdent(), "co-planar:", coplanar
        print atom.oslIdent(), "toward:", toward
        print atom.oslIdent(), "toward2:", toward2
    normals = bond_positions(atPos, geom, 1.0, bondedPos,
            coPlanar=coplanar, toward=toward, toward2=toward2)
    if debug:
        print atom.oslIdent(), "bond_positions:", [str(x) for x in normals]
    # use vectors so we can switch between lone-pair length and H-length
    for normal in normals:
        testPositions.append(normal - atPos)

    # try to hook up positions with acceptors/donors
    if atom in aro_amines:
        if debug:
            print "delay finishing aromatic amine"
        return False
    all = {}
    protons = {}
    lonePairs = {}
    conflicting = []
    for is_acc, other in hbond_info[atom]:
        if debug:
            print "other:", other.oslIdent()
        if is_acc:
            key = (other, atom)
        else:
            key = (atom, other)
        conflictAllowable = key not in processed
        nearest = None
        if other in finished:
            oprotons, olps = hbond_info[other]
            if is_acc:
                # proton may have been stripped if donor near metal...
                if not oprotons:
                    continue
                opositions = oprotons
                mul = LPlen
            else:
                opositions = olps
                mul = Hlen
            for opos in opositions:
                for check in testPositions:
                    if isinstance(check, chimera.Vector):
                        pos = atPos + check * mul
                    else:
                        pos = check
                    dsq = (opos - pos).sqlength()
                    if nearest is None or dsq < nsq:
                        nearest = check
                        nsq = dsq
        else:
            otherPos = other._hb_coord
            if is_acc:
                mul = LPlen
            else:
                mul = Hlen
            for check in testPositions:
                if isinstance(check, chimera.Vector):
                    pos = atPos + check * mul
                else:
                    pos = check
                dsq = (pos - otherPos).sqlength()
                if debug:
                    print "dist from other to",
                    if isinstance(check, chimera.Point):
                        print "pre-existing proton:",
                    elif check in all:
                        if check in protons:
                            print "new proton:",
                        else:
                            print "new lone pair:",
                    else:
                        print "unfilled position:",
                    import math
                    print math.sqrt(dsq)
                if nearest is None or dsq < nsq:
                    nearest = check
                    nsq = dsq
        if isinstance(nearest, chimera.Point):
            # closest to known hydrogen; no help in positioning...
            if is_acc and conflictAllowable:
                # other is trying to donate and is nearest
                # to one of our hydrogens
                conflicting.append((is_acc, other))
            continue
        if nearest in all:
            if is_acc:
                if nearest in protons and conflictAllowable:
                    conflicting.append((is_acc, other))
            elif nearest in lonePairs and conflictAllowable:
                conflicting.append((is_acc, other))
            continue
        # check for steric conflict (frequent with metal coordination)
        if is_acc:
            pos = atPos + nearest * LPlen
            atBump = 0.0
        else:
            pos = atPos + nearest * Hlen
            atBump = h_rad
        checkDist = 2.19 + atBump
        # since searchTree is a module variable that changes,
        # need to access via the module...
        nearby = AddH.searchTree.searchTree(pos.data(), checkDist)
        stericClash = False
        okay = set([atom, other])
        okay.update(atom.primaryNeighbors())
        for nb in nearby:
            if nb in okay:
                continue
            if nb.structure != atom.structure \
            and nb.structure.id == atom.structure.id:
                # ignore clashes with sibling submodels
                continue
            dChk = vdw_radius(nb) + atBump - 0.4
            if dChk*dChk >= distance_squared(nb._hb_coord, pos):
                stericClash = True
                if debug:
                    print "steric clash with", nb.oslIdent(), "(%.3f < %.3f)" % (pos.distance(nb._hb_coord), dChk)
                break
        if stericClash and conflictAllowable:
            conflicting.append((is_acc, other))
            continue

        all[nearest] = 1
        if is_acc:
            if debug:
                print "determined lone pair"
            lonePairs[nearest] = 1
        else:
            if debug:
                print "determined proton"
            protons[nearest] = 1

    for is_acc, other in conflicting:
        if debug:
            print "Removing hbond to %s due to conflict" % other.oslIdent()
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
    # since any conflicting hbonds may have been used to determine
    # positions, determine the positions again with the remaining
    # hbonds
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
                        print "restoring %s/%s hbond pruned by hbond to %s" % (phb[0].oslIdent(), phb[1].oslIdent(), other.oslIdent())
                    processed.remove(phb)
                    hbond_info.setdefault(phb[0], []).append((False, phb[1]))
                    hbond_info.setdefault(phb[1], []).append((True, phb[0]))
                del pruned_by[key]
        if atom not in hbond_info:
            if debug:
                print "No non-conflicting hbonds left!"
            return False
        if debug:
            print "calling _tryFinish with non-conflicting hbonds"
        return _tryFinish(atom, hbond_info, finished, aro_amines,
                            pruned_by, processed)
    # did we determine enough positions?
    if len(all) < openings \
    and len(protons) < hydsToPosition \
    and len(lonePairs) < openings - hydsToPosition:
        if debug:
            print "not enough hookups (all/protons/lps):", len(all), len(protons), len(lonePairs)
        return False

    if len(protons) < hydsToPosition:
        for pos in testPositions:
            if isinstance(pos, chimera.Point):
                continue
            if pos not in all:
                protons[pos] = 1
    Hlocs = []
    for Hvec in protons.keys():
        Hlocs.append(atPos + Hvec * Hlen)

    LPlocs = []
    for vec in testPositions:
        if isinstance(vec, chimera.Point):
            continue
        if vec not in protons:
            LPlocs.append(atPos + vec * LPlen)

    hbond_info[atom] = (Hlocs, LPlocs)
    finished[atom] = True
    return True

def addAltlocHyds(atom, geom, bond_length, need_coplanar, problem_atoms):
    altLocs = set()
    for n in atom.neighbors:
        if n.altLoc:
            altLocs.add(n.altLoc)
    if need_coplanar:
        for na in atom.neighbors:
            for nna in na.neighbors:
                if nna != atom:
                    if nna.altLoc:
                        altLocs.add(nna.altLoc)
    for altLoc in altLocs:
        neighbors = []
        neighborAtoms = []
        numOcc = 0
        totOcc = 0.0
        for n in atom.neighbors:
            if n.altLoc:
                if n.altLoc != altLoc:
                    continue
                numOcc += 1
                totOcc += getattr(n, 'occupancy', 0.5)
            neighborAtoms.append(n)
            neighbors.append(n._hb_coord)

        coplanar = []
        if need_coplanar:
            for na in neighborAtoms:
                for nna in na.neighbors:
                    if nna != atom:
                        if nna.altLoc:
                            if nna.altLoc != altLoc:
                                continue
                            numOcc += 1
                            totOcc += getattr(nna, 'occupancy', 0.5)
                        coplanar.append(nna._hb_coord)
        if len(coplanar) > 2:
            problem_atoms.append(atom)
            continue
        h_positions = bond_positions(atom._hb_coord, geom,
                bond_length, neighbors, coPlanar=coplanar)
        if numOcc:
            occupancy = totOcc / float(numOcc)
        else:
            occupancy = 0.5
        _attach_hydrogens(atom, h_positions, altLoc=altLoc, occupancy=occupancy)
    return h_positions
'''

def _angle_check(d, a, hbond_info):
    add_A_to_D = add_D_to_A = True
    # are the protons/lps already added to the donor pointed toward the acceptor?
    if d in hbond_info:
        geom = _type_info(d).geometry
        for is_acc, da in hbond_info[d]:
            ang = chimera.angle(da._hb_coord,
                        d._hb_coord, a._hb_coord)
            if ang > _test_angles[geom]:
                continue
            if is_acc:
                # lone pair pointing toward acceptor;
                # won't work
                add_A_to_D = add_D_to_A = False
                if debug:
                    print "can't donate; lone pair (to %s) pointing toward acceptor (angle %g)" % (da.oslIdent(), ang)
                break
            add_A_to_D = False
            if debug:
                print "donor already pointing (angle %g) towards acceptor (due to %s)" % (ang, da.oslIdent())
    if not add_A_to_D and not add_D_to_A:
        return add_A_to_D, add_D_to_A
    if a in hbond_info:
        geom = _type_info(a).geometry
        for is_acc, da in hbond_info[a]:
            ang = angle(da._hb_coord, a._hb_coord, d._hb_coord)
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
    ang = angle(ga._hb_coord, pivot_atom._hb_coord, ta._hb_coord)
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
            toward = da._hb_coord
        else:
            # one proton attached
            donor_pos = donor._hb_coord
            acceptor_pos = acceptor._hb_coord
            attached = [donor.neighbors[0]._hb_coord]
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
            toward = da._hb_coord
        else:
            # one proton attached
            donor_pos = donor._hb_coord
            acceptor_pos = acceptor._hb_coord
            attached = [acceptor.neighbors[0]._hb_coord]
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
    tet_pos = tet._hb_coord
    partner_pos = partner._hb_coord
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
            bp=bonded[0]._hb_coord: dihedral(pp, tp, bp, op) / 30.0
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
        angle_group = int(chk_func(other._hb_coord))
        if angle_group in [1,2,5,6]:
            # in the tetrahedral "dead zones"
            if debug:
                print("tetCheck for %s; dead zone for %s" % (tet, other))
            return False
        if angle_group == 0:
            if debug:
                print("tetCheck for; pointing towards %s returning %s" % (tet, other, result))
            return result
        towards.append(other._hb_coord)
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
            if not _tet2_check(da._hb_coord, prev[0]._hb_coord, prev[1]._hb_coord,
                            other._hb_coord):
                if debug:
                    print("pruned hbond (tet check)", [str(a) for a in rel])
                processed.add(rel)
                pruned_by.setdefault(hbond, []).append(rel)

