# vim: set expandtab shiftwidth=4 softtabstop=4:

from chimerax.core.settings import Settings

class _AlignmentsSettings(Settings):

    EXPLICIT_SAVE = {
        'viewer': 'mav'
    }

settings = None
def init(session):
    global settings
    # don't initialize a zillion times, which would also overwrite any changed but not
    # saved settings
    if settings is None:
        settings = _AlignmentsSettings(session, "alignments")
