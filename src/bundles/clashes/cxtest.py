from chimerax.core.commands import run
run(session, "open 1www ; clashes #1 test self make false sel true")
from chimerax.atomic import selected_atoms
num_selected = len(selected_atoms(session))
if num_selected != 43:
	raise SystemExit("Finding clashes in  1www selected %d atoms instead of 43!" % num_selected)
run(session, "contacts #1 test self make false sel true")
num_selected = len(selected_atoms(session))
if num_selected != 2581:
	raise SystemExit("Finding contacts in 1www selected %d atoms instead of 2581!" % num_selected)
