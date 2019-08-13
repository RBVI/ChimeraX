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

_triggers = None
def get_triggers(session=None):
    """Get the atomic triggers (prior implementation used 'session' arg)"""
    global _triggers
    if _triggers is None:
        from chimerax.core.triggerset import TriggerSet
        _triggers = TriggerSet()
        _triggers.add_trigger("changes")
    return _triggers
