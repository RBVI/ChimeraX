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

from .provider_api import StartStructureProvider

from chimerax.core.toolshed import BundleAPI

_lazy_exports = {
    'modify_atom': 'mod',
    'cn_peptide_bond': 'mod',
    'place_fragment': 'start',
    'place_helium': 'start',
    'place_peptide': 'start',
    'PeptideError': 'start',
}


def __getattr__(name):
    try:
        module_name = _lazy_exports[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    from importlib import import_module
    value = getattr(import_module('.' + module_name, __name__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(_lazy_exports))


# Preserve wildcard exports, including the submodules previously imported here.
__all__ = [*_lazy_exports, 'mod', 'start', 'provider_api',
           'StartStructureProvider', 'BundleAPI', 'BuildStructureAPI', 'bundle_api']


class BuildStructureAPI(BundleAPI):

    @staticmethod
    def get_class(class_name):
        if class_name == 'BuildStructureTool':
            from .tool import BuildStructureTool
            return BuildStructureTool
        return BundleAPI.getclass(class_name)

    @staticmethod
    def initialize(session, bundle_info):
        if session.ui.is_gui:
            session.ui.triggers.add_handler('ready', lambda *args, ses=session:
                BuildStructureAPI._add_gui_items(ses))

    @staticmethod
    def register_command(command_name, logger):
        from . import cmd
        cmd.register_command(command_name, logger)

    @staticmethod
    def run_provider(session, name, mgr):
        from .providers import get_provider
        return get_provider(session, name, mgr)

    @staticmethod
    def start_tool(session, tool_name):
        from .tool import BuildStructureTool
        return BuildStructureTool(session, tool_name)

    @staticmethod
    def _add_gui_items(session):
        from .contextmenu import add_selection_context_menu_items
        add_selection_context_menu_items(session)

bundle_api = BuildStructureAPI()
