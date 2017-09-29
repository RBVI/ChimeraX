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

#from chimera import Coord, Point
#from chimera.bondGeom import tetrahedral, planar, linear, single, bondPositions
#from chimera.idatm import *
#from AddH import newHydrogen, findNearest, roomiest, bondWithHLength, \
#							findRotamerNearest
from math import sin, cos, pi, sqrt
#from chimera import Element
#Element_H = Element(1)

sin5475 = sin(pi * 54.75 / 180.0)
cos5475 = cos(pi * 54.75 / 180.0)

def add_hydrogens(atom, *args, **kw):
    # determine what alt_loc(s) to add hydrogens to...
    alt_loc_atom = None
    if len(atoms.alt_locs) > 1:
        alt_loc_atom = atom
    else:
        for nb in atom.neighbors:
            if len(nb.alt_locs) > 1:
                alt_loc_atom = nb
                break
    _alt_loc_add_hydrogens(atom, alt_loc_atom, *args, **kw)

#TODO: tricky in that later alt locs may not be adding new hydrogen atoms, just new positions
def _alt_loc_add_hydrogens(atom, alt_loc, bonding_info, naming_schema, total_hydrogens, idatm_type,
							invert, coordinations):
	away = away2 = planar = None
	geom = bonding_info.geometry
	substs = bonding_info.substituents
	needed = substs - atom.num_bonds
	if needed <= 0:
		return
	at_pos = atom.scene_coord
	exclude = coordinations + list(atom.neighbors)
	occupancy = None
	if geom == 3:
		if atom.num_bonds == 1:
            bonded = atom.neighbors[0]
            #TODO: need to rethink how to add hydrogens in the face of altloc positions;
            # may need to restore old code and go at it again
            if atom.altLoc:
                grandBonded = [nb for nb in bonded.neighbors
                        if not nb.altLoc or nb.altLoc == atom.altLoc]
            else:
                grandBonded = bonded.primaryNeighbors()
            grandBonded.remove(atom)
            if len(grandBonded) < 3:
                planar = [a.xformCoord() for a in grandBonded]
		elif atom.num_bonds == 2:
			if len(atom.neighbors) > 2:
				for altLoc in set([nb.altLoc for nb in atom.neighbors if nb.altLoc]):
					add_hydrogens(atom, bonding_info, naming_schema, total_hydrogens, idatm_type,
							invert, coordinations, altLoc=altLoc)
				return
	if geom == 4 and atom.num_bonds == 0:
		away, d, natom = findNearest(at_pos, atom, exclude, 3.5)
		if away:
			away2, d2, natom2 = findRotamerNearest(at_pos,
				idatm_type[atom], atom, natom, 3.5)
	elif geom == 4 and len(coordinations) + atom.num_bonds == 1:
		away, d, natom = findRotamerNearest(at_pos,
				idatm_type[atom], atom, (list(atom.neighbors)+coordinations)[0], 3.5)
	else:
		away, d, natom = findNearest(at_pos, atom, exclude, 3.5)

	bondedPos = []
	for bonded in atom.neighbors:
		bondedPos.append(bonded.xformCoord())

	if coordinations:
		toward = coordinations[0].xformCoord()
		away2 = away
		away = None
	else:
		toward = None
	positions = bondPositions(at_pos, geom, bondWithHLength(atom, geom),
		bondedPos, toward=toward, coPlanar=planar, away=away, away2=away2)
	if coordinations:
		coordPos = None
		for pos in positions:
			d = pos.sqdistance(toward)
			if coordPos is None or d < lowest:
				coordPos = pos
				lowest = d
		positions.remove(coordPos)
	if len(positions) > needed:
		positions = roomiest(positions, atom, 3.5, bonding_info)[:needed]
	for i, pos in enumerate(positions):
		h = newHydrogen(atom, i+1, total_hydrogens, naming_schema,
							invert.apply(pos), bonding_info)
		if occupancy is not None:
			h.occupancy = occupancy
