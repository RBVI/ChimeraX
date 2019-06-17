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

from chimerax.core.toolshed import ProviderManager
class RotamerLibManager(ProviderManager):
    """Manager for rotmer libraries"""

    def __init__(self, session):
        self.session = session
        self.rot_libs = None
        from chimerax.core.triggerset import TriggerSet
        self.triggers = TriggerSet()
        self.triggers.add_trigger("rotamer libs changed")
        self._library_info = {}

    def library(self, name):
        lib_info = self._library_info[name]
        from . import RotamerLibrary
        if not isinstance(lib_info, RotamerLibrary):
            self._library_info[name] = lib_info = lib_info.run_provider(self.session, name, self)
        return lib_info

    def library_names(self, *, installed_only=False):
        if not installed_only:
            return self._library_info.keys()
        from . import RotamerLibrary
        lib_names = []
        for name, info in self.library_info.items():
            if isinstance(info, RotamerLibrary) or info.installed:
                lib_names.append(name)
        return lib_names

    def add_provider(self, bundle_info, name, **kw):
        self._library_info[name] = bundle_info

    def end_providers(self):
        self.triggers.activate_trigger("rotamer libs changed", self)
