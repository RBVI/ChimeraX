from chimerax.core.commands import run
run(session, "open 2gbp")
from chimerax.atomic.hbonds import find_hbonds, rec_dist_slop, rec_angle_slop
hbonds = find_hbonds(session, session.models[:1])
if len(hbonds) != 792:
	raise SystemExit("Exepcted to find 792 hbonds in 2gbp; actually found %d" % len(hbonds))
