# vim: set expandtab shiftwidth=4 softtabstop=4:

CP_SPECIFIC_SPECIFIC = "ss"
CP_SPECIFIC_BEST = "sb"
CP_BEST = "bb"

AA_NEEDLEMAN_WUNSCH = "Needleman-Wunsch"
AA_SMITH_WATERMAN = "Smith-Waterman"

#--- toolshed/session-init funcs ---
def initialize(bundle_info, session):
    from . import settings
    settings.init(session)

def register_command(command_name, bundle_info):
    from . import match
    match.register_command()
