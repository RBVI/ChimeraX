def test_maestro(test_production_session):
    session = test_production_session
    from chimerax.core.commands import run
    import os
    test_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "test-data", "glide-test2.mae")
    run(session, "close; open %s" % test_file)
