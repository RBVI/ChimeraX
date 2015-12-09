def check_for_changes(session):
    """Check for, and propagate ChimeraX atomic data changes.

    This is called once per frame, and whenever otherwise needed.
    """
    ct = session.change_tracker
    if not ct.changed:
        return
    ul = session.update_loop
    with ul.block_redraw():
        changes = ct.changes
        ct.clear()
        session.triggers.activate_trigger("atomic changes", changes)
