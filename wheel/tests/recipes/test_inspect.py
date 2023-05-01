import os

import pytest

from chimerax.core.session import Session
from chimerax.pdb import open_pdb
from chimerax.dist_monitor import _DistMonitorBundleAPI
from chimerax.atomic import initialize_atomic

@pytest.mark.dependency(
    depends=["tests/pdb/test_open_pdb.py::test_open_pdb"]
    , scope="session"
)
def test_open_pdb():
    session = Session('cx standalone')
    _DistMonitorBundleAPI.initialize(session)
    initialize_atomic(session)
    pdb_loc = os.path.join(os.path.dirname(__file__), "..", "data", "pdb", "1ie9.pdb")
    pdb, status_message = open_pdb(session, pdb_loc)
    session.models.add(pdb)
    pdb = session.models[0]
    assert pdb.name == "1ie9.pdb"
    assert pdb.num_atoms == 2276
    assert pdb.num_residues == 481
    assert pdb.num_chains == 1
