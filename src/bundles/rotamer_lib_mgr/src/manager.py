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

class RotamerLibManager:
    """Manager for rotmer libraries"""

    def __init__(self, session):
        self.session = session
        self.rot_libs = None
        from chimerax.core.triggerset import TriggerSet
        self.triggers = TriggerSet()
        self.triggers.add_trigger("rotamer libs changed")
        self.provider_info = []

    @property
    def libraries(self):
        if self.rot_libs is None:
            self.rot_libs = [bi.run_provider(self.session, name, self) for bi, name in self.provider_info]
        return self.rot_libs[:]

    def add_provider(self, bundle_info, name, **kw):
        if self.rot_libs is None:
            self.provider_info.append((bundle_info, name))
        else:
            self.rot_libs.append(bundle_info.run_provider(self.session, name, self))

    def end_providers(self):
        if self.rot_libs:
            self.triggers.activate_trigger("rotamer libs changed", self)
