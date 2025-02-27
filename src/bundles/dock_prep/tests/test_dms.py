import os

test_data_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), "test-data")
test_water_dms_file = os.path.join(test_data_folder, "water.dms")
test_gly_dms_file = os.path.join(test_data_folder, "gly.dms")

def test_water_dms(test_production_session):
    from chimerax.core.commands import run
    session = test_production_session
    run(session, "open smiles:O; surface gridSpacing 2.0 probeRadius 1.4")
    import io
    buffer = io.StringIO()
    from chimerax.dock_prep import save_dms
    save_dms(session, buffer)
    with open(test_water_dms_file, 'r') as f:
        assert(f.read() == buffer.getvalue()), "Generated water.dms file differs from saved"

def test_gly_dms(test_production_session):
    from chimerax.core.commands import run
    session = test_production_session
    run(session, "open smiles:C(C(=O)O)N; surface #1 gridSpacing 2.0 probeRadius 1.4")
    import io
    buffer = io.StringIO()
    from chimerax.dock_prep import save_dms
    save_dms(session, buffer)
    with open(test_gly_dms_file, 'r') as f:
        assert(f.read() == buffer.getvalue()), "Generated gly.dms file differs from saved"
