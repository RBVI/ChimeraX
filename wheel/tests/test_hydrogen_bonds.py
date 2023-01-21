import os

import pytest

from chimerax.core.session import Session
from chimerax.pdb import open_pdb
from chimerax.hbonds.cmd import cmd_hbonds
from chimerax.atomic import initialize_atomic
from chimerax.dist_monitor import _DistMonitorBundleAPI

@pytest.mark.dependency(
    depends=["tests/test_open_pdb.py::test_open_pdb"]
    , scope="session"
)
def test_hydrogen_bonds():
    session = Session('cx standalone')
    initialize_atomic(session)
    _DistMonitorBundleAPI.initialize(session)
    pdb_loc = os.path.join(os.path.dirname(__file__), "data", "pdb", "1ie9.pdb")
    pdb = open_pdb(session, pdb_loc)
    pdb_model = pdb[0][0]
    session.models.add([pdb_model])
    ligand = pdb_model.atoms.filter(pdb_model.atoms.structure_categories == 'ligand')
    assert len(cmd_hbonds(session, ligand)) == 10
