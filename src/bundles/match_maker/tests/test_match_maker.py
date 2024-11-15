def test_match_maker(test_production_session):
    from chimerax.core.commands import run
    session = test_production_session
    run(session, "open 1mtx")
    run(session, "mm #1.2-23 to #1.1")
