import os

import pytest

from chimerax.core.session import Session
from chimerax.pdb import open_pdb
from chimerax.hbonds.cmd import cmd_hbonds
from chimerax.hbonds import find_hbonds, rec_dist_slop, rec_angle_slop
from chimerax.atomic import initialize_atomic
from chimerax.dist_monitor import _DistMonitorBundleAPI

@pytest.mark.dependency(
    depends=["tests/pdb/test_open_pdb.py::test_open_pdb"]
    , scope="session"
)
def test_hydrogen_bonds():
    session = Session('cx standalone')
    _DistMonitorBundleAPI.initialize(session)
    initialize_atomic(session)
    pdb_loc = os.path.join(os.path.dirname(__file__), "..", "data", "pdb", "2gbp.pdb")
    models, status_message = open_pdb(session, pdb_loc)
    session.models.add(models)
    hbonds = find_hbonds(session, session.models[:1],
	    dist_slop=rec_dist_slop, angle_slop=rec_angle_slop)
    assert len(hbonds) == 793

@pytest.mark.dependency(
    depends=["tests/pdb/test_open_pdb.py::test_open_pdb"]
    , scope="session"
)
def test_ligand_hydrogen_bonds():
    session = Session('cx standalone')
    initialize_atomic(session)
    _DistMonitorBundleAPI.initialize(session)
    pdb_loc = os.path.join(os.path.dirname(__file__), "..", "data", "pdb", "1ie9.pdb")
    models, status_message = open_pdb(session, pdb_loc)
    pdb_model = models[0]
    session.models.add(models)
    ligand = pdb_model.atoms.filter(pdb_model.atoms.structure_categories == 'ligand')
    assert len(cmd_hbonds(session, ligand)) == 11
