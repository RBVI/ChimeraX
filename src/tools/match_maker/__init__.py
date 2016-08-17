# vim: set expandtab shiftwidth=4 softtabstop=4:

from .match import CP_SPECIFIC_SPECIFIC, CP_SPECIFIC_BEST, CP_BEST
from .match import AA_NEEDLEMAN_WUNSCH, AA_SMITH_WATERMAN

#--- toolshed/session-init funcs ---
def initialize(bundle_info, session):
    from . import settings
    settings.init(session)
