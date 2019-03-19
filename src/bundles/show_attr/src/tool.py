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

from chimerax.core.tools import ToolInstance

class ShowAttr(ToolInstance):

    SESSION_ENDURING = True
    # if SESSION_ENDURING is True, tool instance not deleted at session closure
    #help = "help:user/tools/modelpanel.html"

    def __init__(self, session):
        ToolInstance.__init__(self, session, "Render/Select By Attribute")
        self.display_name = "Models"
        self.settings = ShowAttrSettings(session, "ShowAttr")

from chimerax.core.settings import Settings
class ShowAttrSettings(Settings):
    AUTO_SAVE = {
        #'last_use': None
    }

_sa = None
def show_attr(session):
    global _sa
    if _sa is None:
        _sa = ShowAttr(session)
    return _sa
