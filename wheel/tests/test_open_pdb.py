import os

import pytest

from chimerax.core.session import Session
from chimerax.pdb import open_pdb
from chimerax.atomic import AtomicStructure

@pytest.mark.dependency()
def test_open_pdb():
    session = Session('cx standalone')
    pdb_loc = os.path.join(os.path.dirname(__file__), "data", "pdb", "1ie9.pdb")
    pdb = open_pdb(session, pdb_loc)
    assert pdb is not None
