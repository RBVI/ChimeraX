def test_hbonds(test_production_session):
    from chimerax.core.commands import run
    from chimerax.hbonds import find_hbonds, rec_dist_slop, rec_angle_slop
    session = test_production_session
    run(session, "open 2gbp")
    hbonds = find_hbonds(session, session.models[:1], dist_slop=rec_dist_slop, angle_slop=rec_angle_slop)
    assert(len(hbonds) == 793), "Expected to find 793 hbonds in 2gbp; actually found %d" % len(hbonds)
