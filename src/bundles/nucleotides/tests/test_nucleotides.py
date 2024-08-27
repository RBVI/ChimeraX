import pytest

commands = [
    "nucleotides atoms",
    "nucleotides slab",
    "nucleotides slab shape ellipsoid show_orientation true",
    "nucleotides tube/slab",
    "nucleotides tube/slab shape muffler glycosidic true",
    "nucleotides ladder",
    "nucleotides ladder show_stubs false",
    "nucleotides stubs base_only true"
]

@pytest.mark.parametrize("command", commands)
def test_nucleotides_atoms(test_production_session, command):
    session = test_production_session
    from chimerax.core.commands import run
    run(session, "open 2tpk")
    run(session, command)
    run(session, "~nucleotides")

def test_selecting_nucleotides_when_hidden(test_production_session):
    session = test_production_session
    from chimerax.core.commands import run
    run(session, "open 2tpk")
    run(session, "hide")
    run(session, "select :12")
    run(session, "show")
