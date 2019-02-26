from chimerax.core.commands import run
run(session, "close; open 3fx2")
import io
buffer = io.StringIO()
from chimerax.atomic.mol2 import write_mol2
write_mol2(session, buffer)
saved_mol2 = open("test-data/3fx2.mol2")
if saved_mol2.read() != buffer.getvalue():
	raise ValueError("Generated mol2 file differs from saved")
