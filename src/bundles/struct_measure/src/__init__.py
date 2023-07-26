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

from chimerax.core.toolshed import BundleAPI

class _StructMeasureBundleAPI(BundleAPI):

    @staticmethod
    def initialize(session, bundle_info):
        """Install distance mouse mode"""
        if session.ui.is_gui:
            mm = session.ui.mouse_modes
            from .mouse_dist import DistMouseMode
            mm.add_mode(DistMouseMode(session))

            from chimerax.mouse_modes import SelectContextMenuAction, SelectMouseMode
            class DistanceContextMenuEntry(SelectContextMenuAction):
                def label(self, session):
                    return 'Distance'
                def criteria(self, session):
                    from chimerax.atomic import selected_atoms
                    return len(selected_atoms(session)) == 2
                def callback(self, session):
                    from chimerax.atomic import selected_atoms
                    a1, a2 = selected_atoms(session)
                    command = "dist %s %s" % (a1.string(style="command line"),
                        a2.string(style="command line"))
                    from chimerax.core.commands import run
                    run(session, command)
            SelectMouseMode.register_menu_entry(DistanceContextMenuEntry())

    @staticmethod
    def start_tool(session, tool_name):
        from .tool import get_tool
        return get_tool(session, tool_name)

bundle_api = _StructMeasureBundleAPI()
