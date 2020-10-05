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

from .mod import modify_atom
from .start import place_helium, place_peptide, PeptideError

from chimerax.core.toolshed import BundleAPI

class BuildStructureAPI(BundleAPI):

    @staticmethod
    def init_manager(session, bundle_info, name, **kw):
        from .import manager
        if manager.manager is None:
            manager.manager = manager.StartStructureManager(session)
        return manager.manager

    @staticmethod
    def register_command(command_name, logger):
        from . import cmd
        cmd.register_command(command_name, logger)

    @staticmethod
    def run_provider(session, name, mgr, *, widget_info=None, command_info=None, **kw):
        if widget_info:
            widget, fill = widget_info
            if fill:
                # fill parameters widget
                from .providers import fill_widget
                fill_widget(name, widget)
            else:
                # process parameters widget to generate provider command (sub)string;
                # will not be called if 'indirect' was specified as 'true' in the provider info
                # (e.g. it links to another tool/interface);
                # if 'new_model_only' in the provider info was 'true' then the returned string
                # should be the _entire_ command for opening the model, not a substring of the
                # 'build' command
                from .providers import process_widget
                return process_widget(session, name, widget)
        else:
            # add atoms to structure (process provider command string)
            structure, substring = command_info
            from .providers import process_command
            return process_command(session, name, structure, substring)

    @staticmethod
    def start_tool(session, tool_name):
        from .tool import BuildStructureTool
        return BuildStructureTool(session, tool_name)

bundle_api = BuildStructureAPI()
