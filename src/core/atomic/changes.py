# vim: set expandtab shiftwidth=4 softtabstop=4:
def check_for_changes(session):
    """Check for, and propagate ChimeraX atomic data changes.

    This is called once per frame, and whenever otherwise needed.
    """
    ct = session.change_tracker
    if not ct.changed:
        return
    ul = session.update_loop
    ul.block_redraw()
    try:
        changes = ct.changes
        ct.clear()
        session.triggers.activate_trigger("atomic changes", changes)
    finally:
        ul.unblock_redraw()
