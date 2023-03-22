from chimerax.core.commands import CmdDesc, register
from chimerax.core.session import Session
from chimerax.atomic import AtomsArg
from chimerax.atomic import initialize_atomic
from chimerax.dist_monitor import _DistMonitorBundleAPI

import pytest

#@pytest.mark.dependency(
#    depends=["tests/core/test_session.py::test_create_session"], scope="session"
#)
def test_register_command():
    def flip(session, atoms):
        xyz = atoms.coords
        xyz[:, 2] *= -1
        atoms.coords = xyz

    desc = CmdDesc(required=[('atoms', AtomsArg)], synopsis='flip atom z coordinates')
    session = Session("cx standalone", minimal=True)
    _DistMonitorBundleAPI.initialize(session)
    initialize_atomic(session)
    register('flip', desc, flip, logger=session.logger)
