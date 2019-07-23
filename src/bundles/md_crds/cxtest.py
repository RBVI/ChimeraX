from chimerax.core.commands import run
run(session, "open test-data/chimera_test.pdb; open test-data/chimera_test.xtc structureModels #1")
if session.models[0].num_coordsets != 21:
	raise SystemExit("Expected chimera_test.xtc to produce 21 coordinate sets; actually produced %s"
		% session.models[0].num_coordsets)
run(session, "close; open test-data/start.pdb; open test-data/test.dcd structureModels #1")
if session.models[0].num_coordsets != 2:
	raise SystemExit("Expected chimera_test.xtc to produce 2 coordinate sets; actually produced %s"
		% session.models[0].num_coordsets)
