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

class _MarkersAPI(BundleAPI):

    @staticmethod
    def initialize(session, bundle_info):
        """Register marker mouse modes."""
        if session.ui.is_gui:
            from . import mouse
            mouse.register_mousemode(session)

    @staticmethod
    def start_tool(session, tool_name):
        from .markergui import marker_panel
        p = marker_panel(session, tool_name)
        return p

    @staticmethod
    def open_file(session, path):
        # 'open_file' is called by session code to open a file
        # returns (list of models, status message)
        from . import cmmfiles
        return cmmfiles.read_cmm(session, path)

    @staticmethod
    def save_file(session, path, models=None):
        # 'save_file' is called by session code to save a file
        from . import cmmfiles
        return cmmfiles.write_cmm(session, path, models)

    @staticmethod
    def get_class(class_name):
        # 'get_class' is called by session code to get class saved in a session
        if class_name == 'MarkerSet':
            from .markers import MarkerSet
            return MarkerSet
        return None

bundle_api = _MarkersAPI()

from .markers import MarkerSet, create_link, selected_markers, selected_links
from .mouse import mark_map_center
