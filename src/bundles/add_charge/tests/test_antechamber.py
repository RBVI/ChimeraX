def test_match_maker(test_production_session):
    from chimerax.core.commands import run
    session = test_production_session
    run(session, "open 2gbp")
    run(session, "addh")
    run(session, "addcharge")
