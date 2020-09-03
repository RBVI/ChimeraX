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

from chimerax.core import ProviderManager

class SeqFeatureManager(ProviderManager):

    def __init__(self):
        self._data_source_info = {}
        super().__init__("sequence features")

    def add_provider(self, bundle_info, name):
        self._data_source_info[name] = bundle_info

    @property
    def data_sources(self):
        return self._data_source_info.keys()

_manager = None
def get_manager(self):
    global _manager
    if _manager is None:
        _manager = SeqFeatureManager()
    return manager
