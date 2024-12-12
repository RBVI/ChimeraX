def test_clashes(test_production_session):
    from chimerax.atomic import selected_atoms
    from chimerax.core.commands import run

    session = test_production_session
    run(session, "open 1www ; clashes #1 restrict both make false sel true")
    assert (num_selected := len(selected_atoms(session)) == 43), "Finding clashes in  1www selected %d atoms instead of 43!" % num_selected
    run(session, "contacts #1 restrict both make false sel true")
    assert (num_selected := len(selected_atoms(session)) == 2581), "Finding contacts in 1www selected %d atoms instead of 2581!" % num_selected
