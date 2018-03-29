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

from chimerax.atomic.bondGeom import tetrahedral, planar, linear, single, bond_positions
from .cmd import new_hydrogen, find_nearest, roomiest, _tree_dist, vdw_radius, \
				bond_with_H_length, N_H, find_rotamer_nearest, h_rad
from chimerax.core.geometry import distance_squared

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
        # before adding any of the hydorgens (so that neighbor lists
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
	logger.status("Categorizing heavy atoms", blankAfter=0, secondary=True)
	problem_atoms = []
	fixed = []
	flat = []
	aro_N_rings = {}
	global aro_amines
	aro_amines = {}
	unsaturated = []
	saturated = []
	finished = {}
	hbond_info = {}
	for a, crds in coordinations.items():
        hbond_info[a] = hbi = {}
        # _alt_locs() arranges current alt_loc last
        for alt_loc in _alt_locs(a):
            a.alt_loc = alt_loc
            hbi[alt_loc] = [(True, crd) for crd in crds]
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

	logger.status("Building search tree of atom positions", blankAfter=0, secondary=True)
	# since adaptive search tree is static, it will not include hydrogens added after
    # this; they will have to be found by looking off their heavy atoms

	for need_coplanar in [False, True]:
		if need_coplanar:
			atoms = flat
			logger.status("Adding co-planar hydrogens", blankAfter=0, secondary=True)
		else:
			atoms = fixed
			logger.status("Adding simple fixed hydrogens", blankAfter=0, secondary=True)

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
                else
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
            for alt_loc in _alt_locs(altloc_atom):
                altloc_atom.alt_loc = alt_loc
                #TODO: execute code below, accumulating altloc positions.  Add them after
                # loop.  Reminder: hbond_info uses alt locs now (in value)
			neighbors = []
			neighborAtoms = []
			altLocs = set()
			for n in atom.primaryNeighbors():
				if n.altLoc:
					altLocs.add(n.altLoc)
				neighborAtoms.append(n)
				neighbors.append(n.xformCoord())

			coplanar = []
			if need_coplanar:
				for na in neighborAtoms:
					if not atom.altLoc:
						grandNeighbors = na.primaryNeighbors()
					else:
						grandNeighbors = [gnb for gnb in na.neighbors
								if not gnb.altLoc or gnb.altLoc == atom.altLoc]
					for nna in grandNeighbors:
						if nna != atom:
							if nna.altLoc:
								altLocs.add(nna.altLoc)
							coplanar.append(nna.xformCoord())
			altlocHandlingNeeded = not atom.altLoc and altLocs
			if not altlocHandlingNeeded and len(coplanar) > 2:
				problem_atoms.append(atom)
				continue
			if altlocHandlingNeeded:
				Hpositions = addAltlocHyds(atom, geom, bond_length, need_coplanar, problem_atoms)
			else:
				Hpositions = bond_positions(atom.xformCoord(), geom,
					bond_length, neighbors, coPlanar=coplanar)
				_attachHydrogens(atom, Hpositions)
			finished[atom] = True
			hbond_info[atom] = (Hpositions, [])

	from FindHBond import findHBonds, recDistSlop, recAngleSlop

	logger.status("Finding hydrogen bonds", blankAfter=0, secondary=True)
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
			mlist = [atom_list[0].molecule]
		else:
			return
	else:
		mlist = chimera.openModels.list(modelTypes=[chimera.Molecule])
	hbonds = findHBonds(mlist, distSlop=recDistSlop, angleSlop=recAngleSlop)
	logger.info("%d hydrogen bonds" % len(hbonds))
	# want to assign hydrogens to strongest (shortest) hydrogen
	# bonds first, so sort by distance
	logger.status("Sorting hydrogen bonds by distance", blankAfter=0, secondary=True)
	sortableHbonds = []
	for d, a in hbonds:
		# ignore hbonds for atoms completely occupied with
		# coordinating metals...
		if a in coordinations:
			if len(coordinations[a]) + len(a.primaryBonds()) \
					>= _type_info(a).geometry:
				if debug:
					print a, "completely coordinated by metal"
				continue
		if d in donors:
			if in_isolation and a not in acceptors:
				continue
			sortableHbonds.append((d.xformCoord().sqdistance(
					a.xformCoord()), False, (d, a)))
		if a in acceptors:
			if in_isolation and d not in donors:
				continue
			sortableHbonds.append((d.xformCoord().sqdistance(
					a.xformCoord()), True, (d, a)))
	sortableHbonds.sort()
	logger.status("Organizing h-bond info", blankAfter=0, secondary=True)
	hbonds = {}
	ambiguous = {}
	relBond = {}
	for dsq, isAcc, hbond in sortableHbonds:
		hbonds[hbond] = isAcc
		for da in hbond:
			relBond.setdefault(da, []).append(hbond)
	processed = set()
	breakAmbiguous = False
	breakNring = False
	candidates = {}
	candidates.update(donors)
	candidates.update(acceptors)
	pruned = set()
	prunedBy = {}
	reexamine = {}
	logger.status("Adding hydrogens by h-bond strength", blankAfter=0, secondary=True)
	while len(processed) < len(hbonds):
		logger.status("Adding hydrogens by h-bond strength (%d/%d)" % (len(processed), len(hbonds)), blankAfter=0, secondary=True)
		processedOne = False
		seenAmbiguous = {}
		for dsq, isAcc, hbond in sortableHbonds:
			if hbond in processed:
				continue
			d, a = hbond
			if hbond in pruned:
				if hbond in reexamine:
					del reexamine[hbond]
					if debug:
						print "re-examining", hbond[0].oslIdent(), "->", hbond[1].oslIdent()
				elif not breakAmbiguous:
					seenAmbiguous[d] = True
					seenAmbiguous[a] = True
					continue
			if (d not in candidates or d in finished) \
			and (a not in candidates or a in finished):
				# not relevant
				processed.add(hbond)
				continue

			if (a, d) in hbonds \
			and a not in finished \
			and d not in finished \
			and not _resolveAnilene(d, a, aro_amines, hbond_info) \
			and _angleCheck(d, a, hbond_info) == (True, True):
				# possibly still ambiguous

				# if one or both ends are aromatic nitrogen,
				# then ambiguity depends on whether other
				# nitrogens in ring are finished and are
				# acceptors...
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
						protons, lps = hbond_info[n]
						if protons:
							break
					else:
						# unambiguous, but in the
						# reverse direction!
						processed.add(hbond)
						if debug:
							print "reverse ambiguity resolved"
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
						protons, lps = hbond_info[n]
						if protons:
							break
					else:
						if debug:
							print "ambiguity resolved due to ring acceptors"
						unambiguous = True

				if not unambiguous:
					if not breakAmbiguous \
					or nring and not breakNring:
						if debug:
							print "postponing", [a.oslIdent() for a in hbond]
						seenAmbiguous[d] = True
						seenAmbiguous[a] = True
						if hbond not in pruned:
							_doPrune(hbond, pruned, relBond, processed, prunedBy)
						continue
					if nring:
						if debug:
							print "breaking ambiguous N-ring hbond"
					else:
						if debug:
							print "breaking ambiguous hbond"
			if (a, d) in hbonds:
				if a in finished:
					if debug:
						print "breaking ambiguity because acceptor finished"
				elif d in finished:
					if debug:
						print "breaking ambiguity because donor finished"
				elif _resolveAnilene(d, a, aro_amines, hbond_info):
					if debug:
						print "breaking ambiguity by resolving anilene"
				elif _angleCheck(d, a, hbond_info) != (True, True):
					if debug:
						print "breaking ambiguity due to angle check"
			processed.add(hbond)
			from math import sqrt
			if debug:
				print "processed", d.oslIdent(), "->", a.oslIdent(), sqrt(dsq)
			# if donor or acceptor is tet,
			# see if geometry can work out
			if d not in finished and _type_info(d).geometry == 4:
				if not _tetCheck(d, a, hbond_info, False):
					continue
			if a not in finished and _type_info(a).geometry == 4:
				if d in finished:
					checks = []
					for b in d.primaryNeighbors():
						if b.element.number == 1:
							checks.append(b)
				else:
					checks = [d]
				tetOkay = True
				for c in checks:
					if not _tetCheck(a, c, hbond_info, True):
						tetOkay = False
						break
				if not tetOkay:
					continue
			if a in finished:
				# can d still donate to a?
				if not _canAccept(d, a, *hbond_info[a]):
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
				addAtoD, addDtoA = _angleCheck(d, a,
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
			processedOne = True
			breakAmbigous = False
			breakNring = False

			if hbond not in pruned:
				_doPrune(hbond, pruned, relBond, processed, prunedBy)
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
					for isAcc, da in hbond_info[end]:
						if isAcc:
							print " accepting from", da.oslIdent(),
						else:
							print " donating to", da.oslIdent(),
					print
				didFinish = _tryFinish(end, hbond_info, finished,
						aro_amines, prunedBy, processed)
				if not didFinish:
					continue
				if debug:
					print "finished", end.oslIdent()

				# if this atom is in candidates 
				# then actually add any hydrogens
				if end in candidates and hbond_info[end][0]:
					_attachHydrogens(end, hbond_info[end][0])
					if debug:
						print "protonated", end.oslIdent()
					# if ring nitrogen, eliminate
					# any hbonds where it is an acceptor
					if isinstance(donors[end],chimera.Ring):
						for rb in relBond[end]:
							if rb[1] != end:
								continue
							processed.add(rb)
								
			if a in seenAmbiguous or d in seenAmbiguous:
				# revisit previously ambiguous
				if debug:
					print "revisiting previous ambiguous"
				for da in hbond:
					for rel in relBond[da]:
						if rel in processed:
							continue
						if rel not in pruned:
							continue
						reexamine[rel] = True
				break
		if breakAmbiguous and not processedOne:
			breakNring = True
		breakAmbiguous = not processedOne
	numFinished = 0
	for a in candidates.keys():
		if a in finished:
			numFinished += 1
	if debug:
		print "finished", numFinished, "of", len(candidates), "atoms"

	logger.status("Adding hydrogens to primary aromatic amines",
								blankAfter=0, secondary=True)
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
		atPos = a.xformCoord()
		for isAcc, other in hbond_info.get(a, []):
			if isAcc:
				if not acceptFrom:
					if debug:
						print "accept from", other.oslIdent()
					acceptFrom = other.xformCoord()
				continue
			if debug:
				print "possible donate to", other.oslIdent()
			# find nearest lone pair position on acceptor
			target = _findTarget(a, atPos, other, not isAcc,
							hbond_info, finished)
			# make sure this proton doesn't form
			# an angle < 90 degrees with any other bonds
			# and, if trigonal planar, no angle > 150
			badAngle = False
			checkPlanar = type_info_for_atom[a].geometry == planar
			for bonded in a.primaryNeighbors():
				ang = chimera.angle(bonded.xformCoord(), atPos, target)
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
						coPlanar.append(b2.xformCoord())
			sumVec = chimera.Vector()
			for x in a.primaryNeighbors():
				vec = x.xformCoord() - atPos
				vec.normalize()
				sumVec += vec
			for k in knowns:
				vec = k - atPos
				vec.normalize()
				sumVec += vec
			sumVec.negate()
			sumVec.normalize()

			newPos = bond_positions(atPos, geom, bond_with_H_length(a, geom),
				[nb.xformCoord() for nb in a.primaryNeighbors()] + knowns,
				coPlanar=coPlanar)
			positions.extend(newPos)
		_attachHydrogens(a, positions)
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
								blankAfter=0, secondary=True)
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
		towardPos = towardAtom.xformCoord()
		toward2 = away2 = None
		atPos = a.xformCoord()
		if len(hbInfo) > 1:
			toward2 = hbInfo[1][1].xformCoord()
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
			bondedPos.append(bonded.xformCoord())
		positions = bond_positions(atPos, geom,
			bond_with_H_length(a, geom), bondedPos,
			toward=towardPos, toward2=toward2, away2=away2)
		if len(positions) == hydsToPosition:
			# easy, do them all...
			_attachHydrogens(a, positions)
			continue

		used = {}
		for isAcc, other in hbInfo:
			nearest = None
			otherPos = other.xformCoord()
			for pos in positions:
				dsq = (pos - otherPos).sqlength()
				if not nearest or dsq < nsq:
					nearest = pos
					nsq = dsq
			if nearest in used:
				continue
			used[nearest] = isAcc
			
		remaining = []
		for pos in positions:
			if pos in used:
				continue
			remaining.append(pos)
		# definitely protonate the positions where we donate...
		protonate = []
		for pos, isAcc in used.items():
			if not isAcc:
				protonate.append(pos)
		# ... and the "roomiest" remaining positions.
		rooms = roomiest(remaining, a, _room_dist, _type_info(a))
		needed = hydsToPosition - len(protonate)
		protonate.extend(rooms[:needed])
		# then the least sterically challenged...
		_attachHydrogens(a, protonate)
		hbond_info[a] = (protonate, rooms[needed:])

	if debug:
		print "finished", numFinished, "of", len(candidates), "atoms"

	logger.status("Adding hydrogens to non-h-bonding atoms", blankAfter=0, secondary=True)
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
		atPos = a.xformCoord()
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
			bondedPos.append(bonded.xformCoord())
		positions = bond_positions(atPos, geom,
			bond_with_H_length(a, geom),
			bondedPos, toward=toward, away=away, away2=away2)
		if len(positions) == hydsToPosition:
			# easy, do them all...
			_attachHydrogens(a, positions)
			continue

		# protonate "roomiest" positions
		_attachHydrogens(a, roomiest(positions, a, _room_dist, _type_info(a))[
					:hydsToPosition])

	if debug:
		print "finished", numFinished, "of", len(candidates), "atoms"

	logger.status("Deciding aromatic nitrogen protonation",
								blankAfter=0, secondary=True)
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
				bondedPos.append(bonded.xformCoord())
			positions.append(bond_positions(n.xformCoord(), planar,
				bond_with_H_length(n, planar), bondedPos)[0])
		if len(positions) == 1:
			if debug:
				print "protonate precise ring", Npls[0].oslIdent()
			_attachHydrogens(Npls[0], positions)
		else:
			if debug:
				print "protonate roomiest ring", roomiest(positions, Npls, _room_dist, _type_info)[0][0].oslIdent()
			_attachHydrogens(*roomiest(positions, Npls, _room_dist, _type_info)[0])
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

def _alt_locs(atom):
    # get an alt_loc list with the current one last (so after looping
    # you are at the right alt loc)
    cur_loc = atom.alt_loc
    locs = atom.alt_locs
    locs.remove(cur_loc)
    locs.append(cur_loc)
    return locs

def _findTarget(fromAtom, atPos, toAtom, asDonor, hbond_info, finished):
	if toAtom in coordinations.get(fromAtom, []):
		return toAtom.xformCoord()
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
		toBonded.append(tb.xformCoord())
		if tb.element.number == 1:
			toHPos.append(tb.xformCoord())
		if toGeom == planar:
			toAtomLocs = toAtom.allLocations()
			for btb in tb.primaryNeighbors():
				if btb in toAtomLocs:
					continue
				toCoplanar.append(btb.xformCoord())
	if asDonor:
		# point towards nearest lone pair
		targets = bond_positions(toAtom.xformCoord(), toGeom,
					vdw_radius(toAtom), toBonded,
					coPlanar=toCoplanar, toward=atPos)
	else:
		# point to nearest hydrogen
		if _type_info(toAtom).substituents <= toAtom.num_bonds:
			targets = toHPos
		else:
			targets = toHPos + bond_positions(
				toAtom.xformCoord(), toGeom,
				bond_with_H_length(toAtom, toGeom),
				toBonded, coPlanar=toCoplanar, toward=atPos)
	nearest = None
	for target in targets:
		dsq = (target - atPos).sqlength()
		if not nearest or dsq < nsq:
			nearest = target
			nsq = dsq
	return nearest

def _attachHydrogens(atom, positions, altLoc=None, occupancy=None):
	count = 1
	totalHydrogens = hydrogen_totals[atom]
	naming_schema = (naming_schemas[atom.residue],
						naming_schemas[atom.molecule])
	try:
		mtx = inversion_cache[atom.molecule]
	except KeyError:
		try:
			mtx = atom.molecule.openState.xform
		except ValueError:
			mtx = chimera.Xform.identity()
		mtx.invert()
		inversion_cache[atom.molecule] = mtx
	for Hpos in positions:
		h = new_hydrogen(atom, count, totalHydrogens, naming_schema,
					mtx.apply(Hpos), _type_info(atom), altLoc=altLoc)
		if not h:
			# metal clash
			continue
		if occupancy is not None:
			h.occupancy = occupancy
		count += 1

def _canAccept(donor, acceptor, protons, lonePairs):
	# acceptor is fixed; can it accept from donor?
	if not protons:
		return True
	if not lonePairs:
		raise ValueError("No lone pairs on %s for %s to donate to" % (
							acceptor, donor))
	donPos = donor.xformCoord()
	hDist = min(map(lambda xyz: (xyz - donPos).sqlength(), protons))
	lpDist, lp = min(map(lambda xyz: ((xyz - donPos).sqlength(), xyz),
								lonePairs))
	# besides a lone pair being closest, it must be sufficiently pointed
	# towards the donor
	if lpDist >= hDist:
		from math import sqrt
		if debug:
			print "can't still accept; lp dist (%g) >= h dist (%g)" % (sqrt(lpDist), sqrt(hDist))
	elif chimera.angle(lp, acceptor.xformCoord(), donPos) >= _test_angles[
			type_info_for_atom[acceptor].geometry]:
		if debug:
			print "can't still accept; angle (%g) >= test angle (%g)" % ( chimera.angle(lp, acceptor.xformCoord(), donPos), _test_angles[ type_info_for_atom[acceptor].geometry])
	return lpDist < hDist and chimera.angle(lp,
			acceptor.xformCoord(), donPos) < _test_angles[
			type_info_for_atom[acceptor].geometry]

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
	accPos = acceptor.xformCoord()
	hDist, h = min(map(lambda xyz: ((xyz - accPos).sqlength(), xyz),
								protons))
	lpDist = min(map(lambda xyz: (xyz - accPos).sqlength(), lonePairs))
	# besides a proton being closest, it must be sufficiently pointed
	# towards the acceptor
	if hDist >= lpDist:
		from math import sqrt
		if debug:
			print "can't still donate; h dist (%g) >= lp dist (%g)" % ( sqrt(hDist), sqrt(lpDist))
	elif chimera.angle(h, donor.xformCoord(), accPos) >= _test_angles[
			type_info_for_atom[donor].geometry]:
		if debug:
			print "can't still donate; angle (%g) >= test angle (%g)" % ( chimera.angle(h, donor.xformCoord(), accPos), _test_angles[ type_info_for_atom[donor].geometry])
	return hDist < lpDist and chimera.angle(h,
			donor.xformCoord(), accPos) < _test_angles[
			type_info_for_atom[donor].geometry]

def _tryFinish(atom, hbond_info, finished, aro_amines, prunedBy, processed):
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
	for isAcc, other in hbond_info[atom]:
		all.append(other)
		if isAcc:
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
	atPos = atom.xformCoord()
	targets = []
	for isAcc, other in hbond_info[atom][:2]:
		targets.append(_findTarget(atom, atPos, other, not isAcc,
							hbond_info, finished))

	# for purposes of this intermediate measurement, use hydrogen
	# distances instead of lone pair distances; determine true
	# lone pair positions once hydrogens are found
	bondedPos = []
	testPositions = []
	coplanar = []
	for bonded in atom.primaryNeighbors():
		bondedPos.append(bonded.xformCoord())
		if bonded.element.number == 1:
			testPositions.append(bonded.xformCoord())
		if geom == planar:
			for btb in bonded.primaryNeighbors():
				if btb == atom:
					continue
				coplanar.append(btb.xformCoord())
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
	for isAcc, other in hbond_info[atom]:
		if debug:
			print "other:", other.oslIdent()
		if isAcc:
			key = (other, atom)
		else:
			key = (atom, other)
		conflictAllowable = key not in processed
		nearest = None
		if other in finished:
			oprotons, olps = hbond_info[other]
			if isAcc:
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
			otherPos = other.xformCoord()
			if isAcc:
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
			if isAcc and conflictAllowable:
				# other is trying to donate and is nearest
				# to one of our hydrogens
				conflicting.append((isAcc, other))
			continue
		if nearest in all:
			if isAcc:
				if nearest in protons and conflictAllowable:
					conflicting.append((isAcc, other))
			elif nearest in lonePairs and conflictAllowable:
				conflicting.append((isAcc, other))
			continue
		# check for steric conflict (frequent with metal coordination)
		if isAcc:
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
			if nb.molecule != atom.molecule \
			and nb.molecule.id == atom.molecule.id:
				# ignore clashes with sibling submodels
				continue
			dChk = vdw_radius(nb) + atBump - 0.4
			if dChk*dChk >= distance_squared(nb.xformCoord(), pos):
				stericClash = True
				if debug:
					print "steric clash with", nb.oslIdent(), "(%.3f < %.3f)" % (pos.distance(nb.xformCoord()), dChk)
				break
		if stericClash and conflictAllowable:
			conflicting.append((isAcc, other))
			continue

		all[nearest] = 1
		if isAcc:
			if debug:
				print "determined lone pair"
			lonePairs[nearest] = 1
		else:
			if debug:
				print "determined proton"
			protons[nearest] = 1

	for isAcc, other in conflicting:
		if debug:
			print "Removing hbond to %s due to conflict" % other.oslIdent()
		hbond_info[atom].remove((isAcc, other))
		if not hbond_info[atom]:
			del hbond_info[atom]
		if other in finished:
			continue
		try:
			hbond_info[other].remove((not isAcc, atom))
			if not hbond_info[other]:
				del hbond_info[other]
		except ValueError:
			pass
	# since any conflicting hbonds may have been used to determine
	# positions, determine the positions again with the remaining
	# hbonds
	if conflicting:
		# restore hbonds pruned by the conflicting hbonds
		for isAcc, other in conflicting:
			if isAcc:
				key = (other, atom)
			else:
				key = (atom, other)
			if key in prunedBy:
				for phb in prunedBy[key]:
					if phb[0] in finished or phb[1] in finished:
						continue
					if debug:
						print "restoring %s/%s hbond pruned by hbond to %s" % (phb[0].oslIdent(), phb[1].oslIdent(), other.oslIdent())
					processed.remove(phb)
					hbond_info.setdefault(phb[0], []).append((False, phb[1]))
					hbond_info.setdefault(phb[1], []).append((True, phb[0]))
				del prunedBy[key]
		if atom not in hbond_info:
			if debug:
				print "No non-conflicting hbonds left!"
			return False
		if debug:
			print "calling _tryFinish with non-conflicting hbonds"
		return _tryFinish(atom, hbond_info, finished, aro_amines,
							prunedBy, processed)
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
			neighbors.append(n.xformCoord())

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
						coplanar.append(nna.xformCoord())
		if len(coplanar) > 2:
			problem_atoms.append(atom)
			continue
		Hpositions = bond_positions(atom.xformCoord(), geom,
				bond_length, neighbors, coPlanar=coplanar)
		if numOcc:
			occupancy = totOcc / float(numOcc)
		else:
			occupancy = 0.5
		_attachHydrogens(atom, Hpositions, altLoc=altLoc, occupancy=occupancy)
	return Hpositions

def _angleCheck(d, a, hbond_info):
	addAtoD = addDtoA = True
	# are the protons/lps already added to the
	# donor pointed toward the acceptor?
	if d in hbond_info:
		geom = _type_info(d).geometry
		for isAcc, da in hbond_info[d]:
			angle = chimera.angle(da.xformCoord(),
						d.xformCoord(), a.xformCoord())
			if angle > _test_angles[geom]:
				continue
			if isAcc:
				# lone pair pointing toward acceptor;
				# won't work
				addAtoD = addDtoA = False
				if debug:
					print "can't donate; lone pair (to %s) pointing toward acceptor (angle %g)" % (da.oslIdent(), angle)
				break
			addAtoD = False
			if debug:
				print "donor already pointing (angle %g) towards acceptor (due to %s)" % (angle, da.oslIdent())
	if not addAtoD and not addDtoA:
		return addAtoD, addDtoA
	if a in hbond_info:
		geom = _type_info(a).geometry
		for isAcc, da in hbond_info[a]:
			angle = chimera.angle(da.xformCoord(),
						a.xformCoord(), d.xformCoord())
			if angle > _test_angles[geom]:
				continue
			if not isAcc:
				# proton pointing toward donor; won't work
				if debug:
					print "can't accept; proton (to %s) pointing too much toward donor (angle %g)" % (da.oslIdent(), angle)
				addAtoD = addDtoA = False
				break
			addDtoA = False
			if debug:
				print "acceptor already pointing too much (angle %g) towards donor (due to %s)" % (angle, da.oslIdent())
	return addAtoD, addDtoA

def _pruneCheck(pivotAtom, goldHBond, testHBond):
	geom = _type_info(pivotAtom).geometry
	if geom < 2:
		return False
	if geom < 4 and pivotAtom.num_bonds > 0:
		return False
	if goldHBond[0] == pivotAtom:
		ga = goldHBond[1]
	else:
		ga = goldHBond[0]
	if testHBond[0] == pivotAtom:
		ta = testHBond[1]
	else:
		ta = testHBond[0]
	angle = chimera.angle(ga.xformCoord(), pivotAtom.xformCoord(),
							ta.xformCoord())
	fullAngle = _test_angles[geom] * 3
	while angle > fullAngle / 2.0:
		angle -= fullAngle
	angle = abs(angle)
	return angle > fullAngle / 4.0

def _resolveAnilene(donor, acceptor, aro_amines, hbond_info):
	# donor/acceptor are currently ambiguous;  if donor and/or acceptor are
	# anilenes, see if they can be determined to prefer to donate/accept
	# respectively (if a proton and/or lone pair has been added, see if
	# vector to other atom is more planar or more tetrahedral)

	if donor in aro_amines and donor in hbond_info:
		toward = None
		for isAcc, da in hbond_info[donor]:
			if isAcc:
				return True
			if toward:
				break
			toward = da.xformCoord()
		else:
			# one proton attached
			donorPos = donor.xformCoord()
			acceptorPos = acceptor.xformCoord()
			attached = [
				donor.primaryNeighbors()[0].xformCoord()]
			planars = bond_positions(donorPos, 3, N_H, attached,
							toward=toward)
			planarDist = None
			for planar in planars:
				dist = (acceptorPos - planar).sqlength()
				if planarDist is None or dist < planarDist:
					planarDist = dist

			for tetPos in bond_positions(donorPos, 4, N_H, attached):
				if (tetPos - acceptorPos).sqlength() \
								< planarDist:
					# closer to tet position,
					# prefer acceptor-like behavior
					return False
			if debug:
				print "resolving", donor.oslIdent(), "->", acceptor.oslIdent(), "because of donor"
			return True

	if acceptor in aro_amines and acceptor in hbond_info:
		toward = None
		for isAcc, da in hbond_info[acceptor]:
			if isAcc:
				return False
			if toward:
				break
			toward = da.xformCoord()
		else:
			# one proton attached
			donorPos = donor.xformCoord()
			acceptorPos = acceptor.xformCoord()
			attached = [acceptor.primaryNeighbors()[0].xformCoord()]
			planars = bond_positions(acceptorPos, 3, N_H, attached,
							toward=toward)
			planarDist = None
			for planar in planars:
				dist = (acceptorPos - planar).sqlength()
				if planarDist is None or dist < planarDist:
					planarDist = dist

			for tetPos in bond_positions(acceptorPos, 4, N_H,
								attached):
				if (tetPos - donorPos).sqlength() \
								< planarDist:
					# closer to tet position,
					# prefer acceptor-like behavior
					if debug:
						print "resolving", donor.oslIdent(), "->", acceptor.oslIdent(), "because of acceptor"
					return True
			return False
	return False

def _type_info(atom):
	from chimera.idatm import typeInfo
	if atom in aro_amines:
		return typeInfo['N3']
	return type_info_for_atom[atom]

def _tetCheck(tet, partner, hbond_info, tetAcc):
	"""Check if tet can still work"""
	tetPos = tet.xformCoord()
	partnerPos = partner.xformCoord()
	bonded = tet.primaryNeighbors()
	tetInfo = hbond_info.get(tet, [])
	towards = []
	# if there is a real bond to the tet, we want to check the
	# dihedral to the new position (using a 120 angle) rather than
	# the angle (109.5) since the vector from the tet to the not-
	# yet-added positions is probably not directly along the future bond
	if bonded:
		if debug:
			print "checking dihedral"
		chkFunc = lambda op, pp=partnerPos, tp=tetPos, \
			bp=bonded[0].xformCoord(): chimera.dihedral(
			pp, tp, bp, op) / 30.0
	else:
		if debug:
			print "checking angle"
		chkFunc = lambda op, pp=partnerPos, tp=tetPos: chimera.angle(
			pp, tp, op) / 27.375
	for isAcc, other in tetInfo:
		if isAcc == tetAcc:
			# same "polarity"; pointing towards is good
			result = True
		else:
			# opposite "polarity"; pointing towards is bad
			result = False
		angleGroup = int(chkFunc(other.xformCoord()))
		if angleGroup in [1,2,5,6]:
			# in the tetrahedral "dead zones"
			if debug:
				print "tetCheck for", tet.oslIdent(),
				print "; dead zone for", other.oslIdent()
			return False
		if angleGroup == 0:
			if debug:
				print "tetCheck for", tet.oslIdent(),
				print "; pointing towards", other.oslIdent(),
				print "returning", result
			return result
		towards.append(other.xformCoord())
	# further tests only for 2 of 4 filled...
	if bonded or len(towards) != 2:
		return True

	return _tet2check(tetPos, towards[0], towards[1], partnerPos)

def _tet2check(tetPos, toward, toward2, partnerPos):
	if debug:
		print "2 position tetCheck",
	for pos in bond_positions(tetPos, 4, 1.0, [], toward=toward,
							toward2=toward2):
		if debug:
			print chimera.angle(partnerPos, tetPos, pos),
		if chimera.angle(partnerPos, tetPos, pos) < _test_angles[4]:
			if debug:
				print "true"
			return True
	if debug:
		print "false"
	return False

def _doPrune(hbond, pruned, relBond, processed, prunedBy):
	# prune later hbonds that conflict
	pruned.add(hbond)
	for da in hbond:
		skipping = True
		prev = []
		for rel in relBond[da]:
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
			if _pruneCheck(da, hbond, rel):
				if debug:
					print "pruned hbond ", [a.oslIdent() for a in rel]
				processed.add(rel)
				prunedBy.setdefault(hbond, []).append(rel)
				continue
			# prune hbonds to other altlocs in same residue
			relOther = da == rel[0] and rel[1] or rel[0]
			daOther = da == hbond[0] and hbond[1] or hbond[0]
			if relOther.residue == daOther.residue:
				altLoc1 = relOther.altLoc
				altLoc2 = daOther.altLoc
				if altLoc1.isalnum() and altLoc2.isalnum() and altLoc1 != altLoc2:
					if debug:
						print "pruned altloc hbond ", [a.oslIdent() for a in rel]
					processed.add(rel)
					prunedBy.setdefault(hbond, []).append(rel)
		if len(prev) != 2 or _type_info(da).geometry != 4:
			continue
		# prune based on 2-position tet check
		skipping = True
		for rel in relBond[da]:
			if skipping:
				if rel == hbond:
					skipping = False
				continue
			if rel in processed:
				continue
			other = da == rel[0] and rel[1] or rel[0]
			if not _tet2check(da.xformCoord(), prev[0].xformCoord(),
							prev[1].xformCoord(),
							other.xformCoord()):
				if debug:
					print "pruned hbond (tet check)", [a.oslIdent() for a in rel]
				processed.add(rel)
				prunedBy.setdefault(hbond, []).append(rel)


