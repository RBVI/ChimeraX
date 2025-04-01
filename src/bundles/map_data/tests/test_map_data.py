import os

test_data_folder = os.path.join(os.path.dirname(__file__), "data")
test_ome_tiff = os.path.join(test_data_folder, "tubhiswt_C0.ome.tif")

def test_imagestack_ome_tiff(test_production_session):
    from chimerax.core.commands import run
    session = test_production_session
    run(session, f"open {test_ome_tiff}")
