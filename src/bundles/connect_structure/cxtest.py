from chimerax.core.commands import run
run(session, "open 3fx2")
struct = session.models[0]
struct.bonds.delete()
struct.connect_structure()
if struct.num_bonds != 1152:
	raise SystemExit("Expected 3fx2.connect_structure() to add 1152 bonds; actually added %s" % struct.num_bonds)
