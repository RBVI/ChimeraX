# vim: set expandtab ts=4 sw=4:

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
import os

def get_lib() -> str:
    return os.path.join(__file__, "lib")

from chimerax.core.toolshed import BundleAPI

class _LookingGlassAPI(BundleAPI):

    @staticmethod
    def register_command(command_name, logger):
        # 'register_command' is lazily called when the command is referenced
        from .lookingglass import register_lookingglass_command
        register_lookingglass_command(logger)

bundle_api = _LookingGlassAPI()
