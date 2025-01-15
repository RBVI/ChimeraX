# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2016 Regents of the University of California.
# All rights reserved.  This software provided pursuant to a
# license agreement containing restrictions on its disclosure,
# duplication and use.  For details see:
# https://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
# This notice must be embedded in or attached to all copies,
# including partial copies, of the software or any revisions
# or derivations thereof.
# === UCSF ChimeraX Copyright ===

from chimerax.core.toolshed import ProviderManager

class MDPlottingManager(ProviderManager):

    def __init__(self, session):
        self.session = session
        self.providers = {}
        self._provider_bundles = {}
        self._ui_names = {}
        self._new_providers = []
        super().__init__("start structure")

    def add_provider(self, bundle_info, name, *, ui_name=None, num_atoms=None):
        # 'name' is the name used as an arg in the command
        # 'ui_name' is the name used in the tool interface (defaults to 'name')
        # 'num_atoms' indicates how many atoms are needed to compute the quantity (and therefore are
        #   needed in the command form).  If num_atoms is zero, the quantity is a scalar (e.g. energy).
        if num_atoms is None:
            raise ValueError(f"MD plotting provider {name} did not supply 'num_atoms' in its description")
        try:
            num_atoms = int(num_atoms)
            assert num_atoms >= 0
        except (ValueError, AssertionError):
            raise ValueError(f"'num_atoms' for provider {name} is not a non-negative integer ({num_atoms})")
        self._provider_bundles[name] = bundle_info
        self._ui_names[name] = name if ui_name is None else ui_name
        self._num_atoms[name] = num_atoms
        self._new_providers.append(name)

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

    def num_atoms(self, provider_name):
        return self._num_atoms[provider_name]

    @property
    def provider_names(self):
        return list(self._provider_bundles.keys())

    def ui_name(self, provider_name):
        return self._ui_names[provider_name]

    def _get_provider(self, name):
        if name not in self.providers:
            self.providers[name] = self._provider_bundles[name].run_provider(self.session, name, self)
        return self.providers[name]

_plotting_manager = None
def get_plotting_manager(session):
    global _plotting_manager
    if _plotting_manager is None:
        _plotting_manager = MDPlottingManager(session)
    return _plotting_manager
