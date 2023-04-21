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

from .mod import modify_atom, cn_peptide_bond
from .start import place_fragment, place_helium, place_peptide, PeptideError
from .providers import StartStructureProvider

from chimerax.core.toolshed import BundleAPI

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
