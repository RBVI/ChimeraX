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

from .util import complete_terminal_carboxylate, determine_termini, bond_with_H_length, check_no_hyds
from .dock_prep import dock_prep_arg_info, run_for_dock_prep

from chimerax.core.toolshed import BundleAPI

class AddH_API(BundleAPI):

    @staticmethod
    def get_class(class_name):
        if class_name == "AddHTool":
            from .tool import AddHTool
            return AddHTool
        else:
            raise ValueError(f"Don't know how to get class {class_name}")

    @staticmethod
    def register_command(command_name, logger):
        from . import cmd
        cmd.register_command(command_name, logger)

    @staticmethod
    def start_tool(session, tool_name):
        from .tool import AddHTool
        return AddHTool(session, tool_name)

bundle_api = AddH_API()
