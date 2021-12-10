# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2021 Regents of the University of California.
# All rights reserved.  This software provided pursuant to a
# license agreement containing restrictions on its disclosure,
# duplication and use.  For details see:
# http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
# This notice must be embedded in or attached to all copies,
# including partial copies, of the software or any revisions
# or derivations thereof.
# === UCSF ChimeraX Copyright ===

from chimerax.core.toolshed import BundleAPI

# Expose all of our important modules so they can be imported
# from chimerax.blastprotein instead of chimerax.blastprotein.X
from .cmd import *
from .data_model import *
from .job import *
from .ui import *
from .utils import *

class _MyAPI(BundleAPI):

    api_version = 1

    @staticmethod
    def get_class(class_name):
        class_names = {
            'BlastProteinTool': BlastProteinTool
            , 'BlastProteinResults': BlastProteinResults
            , 'ToolUI': BlastProteinResults
        }
        return class_names.get(class_name, None)

    @staticmethod
    def start_tool(session, bi, ti):
        return BlastProteinTool(session, ti.name)

    @staticmethod
    def register_command(bi, ci, logger):
        command_name = ci.name
        if command_name == "blastpdb":
            from chimerax.core.commands import create_alias
            create_alias(
                command_name, "blastprotein $*", logger=logger,
                url="help:user/commands/blastprotein.html"
            )
            return
        from . import cmd
        function_name = command_name.replace(' ', '_')
        func = getattr(cmd, function_name)
        desc = getattr(cmd, function_name + "_desc")
        if desc.synopsis is None:
            desc.synopsis = ci.synopsis
        from chimerax.core.commands import register
        register(command_name, desc, func, logger=logger)


bundle_api = _MyAPI()
