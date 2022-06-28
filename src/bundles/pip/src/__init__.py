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

from chimerax.core.commands import create_alias, register
from chimerax.core.toolshed import BundleAPI

# Expose all of our important modules so they can be imported
# from chimerax.blastprotein instead of chimerax.blastprotein.X
from .cmd import *

class _MyAPI(BundleAPI):

    api_version = 1

    @staticmethod
    def register_command(bi, ci, logger):
        command_name = ci.name
        if command_name == "devel pip":
            create_alias(command_name, "devel pip $*", logger=logger)
            return
        function_name = command_name.replace(' ', '_')
        func = getattr(cmd, function_name)
        desc = getattr(cmd, function_name + "_desc")
        register(command_name, desc, func, logger=logger)

bundle_api = _MyAPI()
