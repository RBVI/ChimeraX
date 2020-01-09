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

class StartStructureManager(ProviderManager):

    def __init__(self, session):
        # Just for good form.  Base class does not currently define __init__.
        super().__init__()
        self.session = session
        self.providers = {}
        self._new_providers = []

    def add_provider(self, bundle_info, name):
        self.providers[name] = bundle_info
        self._new_providers.append(name)

    def apply(self, name, param_widget, structure):
        # if 'structure' is a string, create a new AtomicStructure with that name
        self.providers[name].run_provider(self.session, name, self, widget=param_widget, structure=structure)

    def end_providers(self):
        # Below code needs to be uncommented once this manager is 'lazy'; doesn't work at startup
        #from .tool import BuildStructureTool
        #for tool in self.session.tools.find_by_class(BuildStructureTool):
        #    tool._new_start_providers(self._new_providers)
        self._new_providers = []

    def fill_parameters_widget(self, name, widget):
        self.providers[name].run_provider(self.session, name, self, widget=widget)

    @property
    def provider_names(self):
        return list(self.providers.keys())

manager = None
