import os

test_data_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), "test-data")
test_mol2_file = os.path.join(test_data_folder, "3fx2.mol2")

def test_mol2(test_production_session):
    from chimerax.core.commands import run
    session = test_production_session
    run(session, "open 3fx2")
    import io
    buffer = io.StringIO()
    from chimerax.mol2 import write_mol2
    write_mol2(session, buffer)
    with open(test_mol2_file, 'r') as f:
        assert(f.read() == buffer.getvalue()), "Generated mol2 file differs from saved"
