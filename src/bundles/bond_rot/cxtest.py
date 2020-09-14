from chimerax.core.commands import run
run(session, "open 1gcn")
dist_info = ((24, 'OE1'), (17, 'NH1'))
torsion_info = ((17, 'NH1'), (17, 'CD'), (17, 'CG'), (17, 'CB'))
dist_atoms = []
torsion_atoms = []
for r in session.models[0].residues:
	if r.number == 17:
		r17 = r
		nh1 = r.find_atom('NH1')
		dist_atoms.append(nh1)
	elif r.number == 24:
		oe1 = r.find_atom('OE1')
from chimerax.geometry import distance
d = distance(nh1.coord, oe1.coord)
if d < 9.87 or d > 9.88:
	raise SystemExit("Distance between %s and %s not initially 9.871!" % (nh1, oe1))
run(session, "torsion :17@nh1,cd,cg,cb 50")
d = distance(nh1.coord, oe1.coord)
if d < 9.21 or d > 9.23:
	raise SystemExit("Distance between %s and %s after torsion change not 9.220!" % (nh1, oe1))

run(session, "setattr :17 r phi -120")
if r17.phi < -120.1 or r17.phi > -119.9:
	raise SystemExit("Setting phi to -120 didn't work! (%g instead)" % r17.phi)
