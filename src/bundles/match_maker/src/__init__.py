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
CP_SPECIFIC_SPECIFIC = "ss"
CP_SPECIFIC_BEST = "bs"
CP_BEST_BEST = "bb"

AA_NEEDLEMAN_WUNSCH = "Needleman-Wunsch"
AA_SMITH_WATERMAN = "Smith-Waterman"


#--- toolshed/session-init funcs ---

from chimerax.core.toolshed import BundleAPI

class _MyAPI(BundleAPI):

    @staticmethod
    def initialize(session, bundle_info):
        from . import settings
        settings.init(session)

    @staticmethod
    def finish(session, bi):
        # deinitialize bundle in session (causes import)
        pass

    @staticmethod
    def register_command(command_name, logger):
        from . import match
        match.register_command(logger)

bundle_api = _MyAPI()
