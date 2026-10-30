import os

import pytest

test_data_folder = os.path.join(os.path.dirname(__file__), "data")
test_tiff_1 = os.path.join(test_data_folder, "tubhiswt_C0.ome.tif")
test_tiff_2 = os.path.join(test_data_folder, "nnInteractiveLabelLayerImg.tiff")
test_nitrite_grid3d = os.path.join(test_data_folder, "nitrite.grid3d")

@pytest.mark.parametrize("tiff", [test_tiff_1, test_tiff_2])
def test_imagestack_ome_tiff(test_production_session, tiff):
    from chimerax.core.commands import run
    session = test_production_session
    run(session, f"open {tiff}")

@pytest.mark.parametrize("grid3d", [test_nitrite_grid3d])
def test_lammps_grid3d(test_production_session, grid3d):
    from chimerax.core.commands import run
    session = test_production_session
    run(session, f"open {grid3d}")
