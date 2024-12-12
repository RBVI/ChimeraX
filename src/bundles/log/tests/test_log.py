def test_log(test_production_session):
    from chimerax.core.commands import run
    session = test_production_session
    run(session, "open 3fx2")
    run(session, "log thumbnail")
    run(session, "log text test")
