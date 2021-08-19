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

class FragmentNotInstalledError(ValueError):
    pass

class FragmentManager(ProviderManager):

    def __init__(self, session):
        self.session = session
        self._provider_bundles = {}
        self._category = {}
        super().__init__("start structure fragments")

    def add_provider(self, bundle_info, name, *, category=None):
        # 'name' is the name used as an arg in the command
        if category is None:
            raise ValueError("Bundle %s did not specify category for structure fragment '%s'"
                % (bundle_info.short_name, name))
        self._provider_bundles[name] = bundle_info
        self._category[name] = category

    def fragment(self, name):
        bi = self._provider_bundles[name]
        if not bi.installed:
            raise FragmentNotInstalledError("Bundle for fragment '%s' is not installed" % name)
        return bi.run_provider(self.session, name, self)

    def fragment_category(self, name):
        return self._category[name]

    def provider_bundle(self, name):
        return self._provider_bundles[name]

    @property
    def provider_names(self):
        return list(self._provider_bundles.keys())
    fragment_names = provider_names

_manager = None
def get_manager(session):
    global _manager
    if _manager is None:
        _manager = FragmentManager(session)
    return _manager
