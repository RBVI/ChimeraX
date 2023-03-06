import os

import pytest

from chimerax.core.session import Session
from chimerax.pdb import open_pdb
from chimerax.dist_monitor import _DistMonitorBundleAPI
from chimerax.atomic import initialize_atomic

@pytest.mark.dependency()
def test_open_pdb():
    session = Session('cx standalone')
    _DistMonitorBundleAPI.initialize(session)
    initialize_atomic(session)
    pdb_loc = os.path.join(os.path.dirname(__file__), "..", "data", "pdb", "1ie9.pdb")
    pdb = open_pdb(session, pdb_loc)
    assert pdb is not None
