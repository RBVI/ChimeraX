from chimerax.core.commands import run
run(session, "open 3fx2 ; sel ligand & aromatic")
from chimerax.atomic import selected_atoms
num_selected = len(selected_atoms(session))
if num_selected != 10:
	raise SystemExit("Selecting 3fx2 :FMN aromatic rings selected %d atoms instead of 10!" % num_selected)
run(session, "sel phosphate")
num_selected = len(selected_atoms(session))
if num_selected != 5:
	raise SystemExit("Selecting 3fx2 phoshates selected %d atoms instead of 5!" % num_selected)
