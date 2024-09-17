#!/usr/bin/env python3
from chimerax.core.commands import run
from chimerax.atomic import selected_atoms


def test_sel_command(test_production_session):
    session = test_production_session

    run(session, "open 3fx2")
    run(session, "sel ligand & aromatic")

    assert (num_selected := len(selected_atoms(session))) == 10, (
        "Selecting 3fx2 :FMN aromatic rings selected %d atoms instead of 10!"
        % num_selected
    )

    run(session, "sel phosphate")

    assert (phosphate_selected := len(selected_atoms(session))) == 5, (
        "Selecting 3fx2 phoshates selected %d atoms instead of 5!" % phosphate_selected
    )
