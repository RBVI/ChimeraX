import os

test_data_folder = os.path.join(os.path.dirname(__file__), "data")
test_file = os.path.join(test_data_folder, "8zzi.cif")

def test_mol2(test_production_session):
    from chimerax.core.commands import run
    session = test_production_session
    run(session, f"open {test_file}")
