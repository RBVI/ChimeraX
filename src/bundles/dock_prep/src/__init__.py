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

#--- public API ---
#TODO

# all modules involved in DockPrep provide these variables/functions
from chimerax.core.commands import BoolArg
dock_prep_arg_info = {
    'del_solvent': BoolArg,
}

def run_for_dock_prep(*args, **kw):
    from .prep import prep
    prep(*args, **kw)

#--- toolshed/session-init funcs ---

from chimerax.core.toolshed import BundleAPI

class DockPrepAPI(BundleAPI):

    @staticmethod
    def register_command(command_name, logger):
        from .cmd import register_command
        register_command(logger)

    #@staticmethod
    #def start_tool(session, tool_name):
    #    from .tool import DockPrepTool
    #    return DockPrepTool(session, tool_name)

bundle_api = DockPrepAPI()
