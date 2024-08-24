import glob
import os

import pytest

test_data_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), "test-data")
test_sdfs = glob.glob(os.path.join(test_data_folder, "*.sdf"))

@pytest.mark.parametrize("test_sdf", test_sdfs)
def test_sdf(test_production_session, test_sdf):
    session = test_production_session
    from chimerax.core.commands import run
    run(session, "open %s" % test_sdf)
