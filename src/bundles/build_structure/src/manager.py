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
        self._ui_names = {}
        self._indirect = {}
        self._new_model_only = {}
        self._new_providers = []

    def add_provider(self, bundle_info, name, *, ui_name=None, indirect=False, new_model_only=False):
        # 'name' is the name used as an arg in the command
        # 'ui_name' is the name used in the tool interface
        # if 'indirect' is True, then the bundle does not directly add atoms but instead provides 
        #     information or links to other tools or web pages (the Apply button will be disabled)
        # if 'new_model_only' is True then the provider can only construct new models and can't
        #     add atoms to existing models.  In that case, no model spec will be expected in the
        #     command and the 'structure' part of run_provider's command_info will be None.
        if isinstance(indirect, str):
            indirect = eval(indirect.capitalize())
        if isinstance(new_model_only, str):
            new_model_only = eval(new_model_only.capitalize())
        self.providers[name] = bundle_info
        self._ui_names[name] = name if ui_name is None else ui_name
        self._indirect[name] = indirect
        self._new_model_only[name] = new_model_only
        self._new_providers.append(name)

    def get_command_substring(self, name, param_widget):
        # given the settings in the parameter widget, get the corresponding command args
        # (can return None if the widget doesn't directly add atoms [e.g. links to another tool])
        return self.providers[name].run_provider(self.session, name, self, widget_info=(param_widget, False))

    def end_providers(self):
        # Below code needs to be uncommented once this manager is 'lazy'; doesn't work at startup
        #from .tool import BuildStructureTool
        #for tool in self.session.tools.find_by_class(BuildStructureTool):
        #    tool._new_start_providers(self._new_providers)
        self._new_providers = []

    def execute_command(self, name, structure, args):
        return self.providers[name].run_provider(self.session, name, self, command_info=(structure, args))

    def fill_parameters_widget(self, name, widget):
        self.providers[name].run_provider(self.session, name, self, widget_info=(widget, True))

    def is_indirect(self, name):
        return self._indirect[name]

    def new_model_only(self, name):
        return self._new_model_only[name]

    @property
    def provider_names(self):
        return list(self.providers.keys())

    def ui_name(self, provider_name):
        return self._ui_names[provider_name]

manager = None
