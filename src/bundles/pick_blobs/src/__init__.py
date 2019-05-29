# vim: set expandtab ts=4 sw=4:

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

from chimerax.core.toolshed import BundleAPI

class _PickBlobsAPI(BundleAPI):

    @staticmethod
    def start_tool(session, tool_name):
        from .pickblobs import pick_blobs_panel, PickBlobs
        p = pick_blobs_panel(session)
        # Bind mouse button when panel shown.
        mm = session.ui.mouse_modes
        mm.bind_mouse_mode('right', [], PickBlobs(session))
        return p

    @staticmethod
    def initialize(session, bundle_info):
        """Register pick blobs mouse mode."""
        if session.ui.is_gui:
            from . import pickblobs
            pickblobs.register_mousemode(session)

    @staticmethod
    def finish(session, bundle_info):
        # TODO: remove mouse mode
        pass

    @staticmethod
    def register_command(command_name, logger):
        # 'register_command' is lazily called when the command is referenced
        from . import pickblobs
        pickblobs.register_measure_blob_command(logger)

bundle_api = _PickBlobsAPI()
