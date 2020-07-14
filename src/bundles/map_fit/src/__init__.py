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

from .fitmap import map_overlap_and_correlation, overlap_and_correlation
from .fitmap import move_selected_atoms_to_maximum, move_atoms_to_maxima
from .fitmap import locate_maximum
from .fitcmd import register_fitmap_command

# -----------------------------------------------------------------------------
#
from chimerax.core.toolshed import BundleAPI

class _MapFitBundle(BundleAPI):

    @staticmethod
    def start_tool(session, tool_name):
        from .fitgui import show_fit_map_dialog
        d = show_fit_map_dialog(session)
        return d

    @staticmethod
    def register_command(command_name, logger):
        # 'register_command' is lazily called when the command is referenced
        from chimerax import map_fit
        map_fit.register_fitmap_command(logger)

bundle_api = _MapFitBundle()
