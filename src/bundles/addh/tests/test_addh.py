def test_addh(test_production_session):
    session = test_production_session
    from chimerax.core.commands import run
    run(session, "open 4hhb")
    pre_num = session.models[0].num_atoms
    run(session, "addh hb t")
    post_num = session.models[0].num_atoms
    added = post_num - pre_num
    assert(added == 4946), "Expected to add 4946 hydrogens to 4hhb; actually added %d" % (added)
