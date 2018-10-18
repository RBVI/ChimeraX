from chimerax.core.commands import run
run(session, "open 1gcn")
dist_info = ((24, 'OE1'), (17, 'NH1'))
torsion_info = ((17, 'NH1'), (17, 'CG'), (17, 'CD'), (17, 'CB'))
dist_atoms = []
torsion_atoms = []
for r in session.models[0].residues:
	if r.number == 17:
		nh1 = r.find_atom('NH1')
		dist_atoms.append(nh1)
	elif r.number == 24:
		oe1 = r.find_atom('OE1')
from chimerax.core.geometry import distance
d = distance(nh1.coord, oe1.coord)
if d < 9.87 or d > 9.88:
	raise SystemExit("Distance between %s and %s not initially 9.871!" % (nh1, oe1))
run(session, "torsion :17@nh1,cg,cd,cb 50")
d = distance(nh1.coord, oe1.coord)
if d < 9.96 or d > 9.97:
	raise SystemExit("Distance between %s and %s after torsion change not 9.963!" % (nh1, oe1))
