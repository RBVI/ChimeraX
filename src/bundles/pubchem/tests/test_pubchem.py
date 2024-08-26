def test_pubchem(test_production_session):
    from chimerax.core.commands import run
    run(test_production_session, "open pubchem:2519")
