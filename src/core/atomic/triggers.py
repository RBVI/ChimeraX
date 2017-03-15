# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2016 Regents of the University of California.
# All rights reserved.  This software provided pursuant to a
# license agreement containing restrictions on its disclosure,
# duplication and use.  For details see:
# http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
# This notice must be embedded in or attached to all copies,
# including partial copies, of the software or any revisions
# or derivations thereof.
# === UCSF ChimeraX Copyright ===

def get_triggers(session):
    """Get the atomic triggers for this session"""
    if not hasattr(session, "_atomic_triggers"):
        from chimerax.core.triggerset import TriggerSet
        session._atomic_triggers = TriggerSet()
        session._atomic_triggers.add_trigger("changes")
    return session._atomic_triggers
