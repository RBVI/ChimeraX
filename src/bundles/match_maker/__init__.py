# vim: set expandtab shiftwidth=4 softtabstop=4:

#--- public API ---
CP_SPECIFIC_SPECIFIC = "ss"
CP_SPECIFIC_BEST = "bs"
CP_BEST = "bb"

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
    def register_command(command_name):
        from . import match
        match.register_command()

bundle_api = _MyAPI()
