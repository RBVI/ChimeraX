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
        self.session = session
        self.providers = {}
        self._provider_bundles = {}
        self._ui_names = {}
        self._indirect = {}
        self._new_model_only = {}
        self._auto_style = {}
        self._new_providers = []
        super().__init__("start structure")

    def add_provider(self, bundle_info, name, *, ui_name=None, indirect=False, new_model_only=False,
            auto_style=True):
        # 'name' is the name used as an arg in the command
        # 'ui_name' is the name used in the tool interface
        # if 'indirect' is True, then the bundle does not directly add atoms but instead provides 
        #     information or links to other tools or web pages (the Apply button will be disabled)
        # if 'new_model_only' is True then the provider can only construct new models and can't
        #     add atoms to existing models.  In that case, no model spec will be expected in the
        #     command and the 'structure' part of run_provider's command_info will be None.
        # if 'auto_style' controls whether new structures have autostyling turned on ('auto_style'
        #     keyword in their constructor).
        if isinstance(indirect, str):
            indirect = eval(indirect.capitalize())
        if isinstance(new_model_only, str):
            new_model_only = eval(new_model_only.capitalize())
        if isinstance(auto_style, str):
            auto_style = eval(auto_style.capitalize())
        self._provider_bundles[name] = bundle_info
        self._ui_names[name] = name if ui_name is None else ui_name
        self._indirect[name] = indirect
        self._new_model_only[name] = new_model_only
        self._auto_style[name] = auto_style
        self._new_providers.append(name)

    def auto_style(self, name):
        return self._auto_style[name]

    def end_providers(self):
        from .tool import BuildStructureTool
        for tool in self.session.tools.find_by_class(BuildStructureTool):
            tool._new_start_providers(self._new_providers)
        self._new_providers = []

    def execute_command(self, name, structure, args):
        return self._get_provider(name).execute_command(structure, args)

    def fill_parameters_widget(self, name, widget):
        if self._provider_bundles[name].installed:
            self._get_provider(name).fill_parameters_widget(widget)
        else:
            from Qt.QtWidgets import QLabel, QVBoxLayout
            from Qt.QtCore import Qt
            layout = QVBoxLayout()
            widget.setLayout(layout)
            info = QLabel('This feature is not installed.  To enable it,'
                ' <a href="internal toolshed">install the %s bundle</a>'
                " from the Toolshed.  Then restart ChimeraX." % self._provider_bundles[name].short_name)
            from chimerax.core.commands import run
            info.linkActivated.connect(lambda *args, bundle_name=self._provider_bundles[name].name:
                run(self.session, "toolshed show %s" % bundle_name))
            info.setWordWrap(True)
            # specify alignment within the label itself (instead of the layout) so that the label
            # is given the full width of the layout to work with, otherwise you get unneeded line
            # wrapping
            info.setAlignment(Qt.AlignCenter)
            layout.addWidget(info)

    def get_command_substring(self, name, param_widget):
        # given the settings in the parameter widget, get the corresponding command args
        # (can return None if the widget doesn't directly add atoms [e.g. links to another tool])
        return self._get_provider(name).command_string(param_widget)

    def is_indirect(self, name):
        return self._indirect[name]

    def new_model_only(self, name):
        return self._new_model_only[name]

    @property
    def provider_names(self):
        return list(self._provider_bundles.keys())

    def ui_name(self, provider_name):
        return self._ui_names[provider_name]

    def _get_provider(self, name):
        if name not in self.providers:
            self.providers[name] = self._provider_bundles[name].run_provider(self.session, name, self)
        return self.providers[name]

_manager = None
def get_manager(session):
    global _manager
    if _manager is None:
        _manager = StartStructureManager(session)
    return _manager
