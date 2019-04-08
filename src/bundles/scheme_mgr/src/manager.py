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

from chimerax.core.state import StateManager
class SchemesManager(StateManager):
    """Manager for http schemes used by all bundles"""

    def __init__(self, session):
        self.schemes = set()
        from chimerax.core.triggerset import TriggerSet
        self.triggers = TriggerSet()
        self.triggers.add_trigger("http schemes changed")

    def reset_state(self, session):
        pass

    def add_provider(self, bundle_info, name, **kw):
        self.schemes.add(name)

    def end_providers(self):
        self.triggers.activate_trigger("http schemes changed", self)

    @staticmethod
    def restore_snapshot(session, data):
        return session.http_schemes

    def take_snapshot(self, session, flags):
        # Presets are "session enduring"
        return {}
