import os
import sys

import pytest

from chimerax.core.session import Session
from chimerax.pdb import open_pdb
from chimerax.hbonds.cmd import cmd_hbonds
from chimerax.atomic import initialize_atomic
from chimerax.dist_monitor import _DistMonitorBundleAPI
from chimerax.image_formats import save_image

@pytest.mark.dependency(
    depends=["tests/hbonds/test_hydrogen_bonds.py::test_hydrogen_bonds"]
    , scope="session"
)
@pytest.mark.skipif(
    sys.platform in ['darwin', 'win32']
    , reason="Offscreen rendering is not supported on macOS or Windows"
)
def test_save_image():
    session = Session('cx standalone', offscreen_rendering=True)
    initialize_atomic(session)
    _DistMonitorBundleAPI.initialize(session)
    pdb_loc = os.path.join(os.path.dirname(__file__), "..", "data", "pdb", "1ie9.pdb")
    pdb = open_pdb(session, pdb_loc)
    pdb_model = pdb[0][0]
    session.models.add([pdb_model])
    save_image(session, "test.png")
