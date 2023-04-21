import pytest
import chimerax.core.session

def test_create_session():
    session = chimerax.core.session.Session("cx standalone", minimal=True)
    assert session is not None
