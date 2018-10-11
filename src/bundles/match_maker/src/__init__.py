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
from .match import CP_SPECIFIC_SPECIFIC, CP_SPECIFIC_BEST, CP_BEST_BEST
from .match import AA_NEEDLEMAN_WUNSCH, AA_SMITH_WATERMAN
from .match import match

#--- toolshed/session-init funcs ---

from chimerax.core.toolshed import BundleAPI

class _MyAPI(BundleAPI):

    @staticmethod
    def register_command(command_name, logger):
        from .match import register_command
        register_command(logger)

bundle_api = _MyAPI()
