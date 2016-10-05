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

from chimerax.core.settings import Settings

class _AlignmentsSettings(Settings):

    EXPLICIT_SAVE = {
        'viewer': 'Multalign Viewer',
        'assoc_error_rate': 10
    }

settings = None
def init(session):
    global settings
    # don't initialize a zillion times, which would also overwrite any changed but not
    # saved settings
    if settings is None:
        settings = _AlignmentsSettings(session, "alignments")
