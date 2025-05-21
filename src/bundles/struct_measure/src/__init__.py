# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2022 Regents of the University of California. All rights reserved.
# The ChimeraX application is provided pursuant to the ChimeraX license
# agreement, which covers academic and commercial uses. For more details, see
# <https://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
#
# This particular file is part of the ChimeraX library. You can also
# redistribute and/or modify it under the terms of the GNU Lesser General
# Public License version 2.1 as published by the Free Software Foundation.
# For more details, see
# <https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html>
#
# THIS SOFTWARE IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND, EITHER
# EXPRESSED OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
# OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. ADDITIONAL LIABILITY
# LIMITATIONS ARE DESCRIBED IN THE GNU LESSER GENERAL PUBLIC LICENSE
# VERSION 2.1
#
# This notice must be embedded in or attached to all copies, including partial
# copies, of the software or any revisions or derivations thereof.
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
