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
from chimerax.core.errors import UserError
from chimerax.core.settings import Settings
from Qt.QtCore import Qt

class DouseSettings(Settings):
    AUTO_SAVE = {
        "show_hbonds": True,
    }

from chimerax.check_waters.tool import CheckWaterViewer
class DouseResultsViewer(CheckWaterViewer):
    def __init__(self, session, tool_name, orig_model=None, douse_model=None, compared_waters=None,
            map=None):
        # if 'model' is None, we are being restored from a session and _finalize_init() will be called later
        super().__init__(session, tool_name, douse_model, compare_info=(orig_model, compared_waters),
            model_labels=("input", "douse"), compare_map=map)

