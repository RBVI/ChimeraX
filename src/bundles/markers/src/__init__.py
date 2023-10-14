# vim: set expandtab ts=4 sw=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2022 Regents of the University of California. All rights reserved.
# This software is provided pursuant to the ChimeraX license agreement, which
# covers academic and commercial uses. For more information, see
# <http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
#
# This file is part of the ChimeraX library. You can also redistribute and/or
# modify it under the GNU Lesser General Public License version 2.1 as
# published by the Free Software Foundation. For more details, see
# <https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html>
#
# This file is distributed WITHOUT ANY WARRANTY; without even the implied
# warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. This notice
# must be embedded in or attached to all copies, including partial copies, of
# the software or any revisions or derivations thereof.
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
    def register_command(command_name, logger):
        from . import cmd
        cmd.register_marker_command(logger)

    @staticmethod
    def get_class(class_name):
        # 'get_class' is called by session code to get class saved in a session
        if class_name == 'MarkerSet':
            from .markers import MarkerSet
            return MarkerSet
        return None

    @staticmethod
    def run_provider(session, name, mgr):
        if mgr == session.open_command:
            from chimerax.open_command import OpenerInfo
            class MarkerInfo(OpenerInfo):
                def open(self, session, data, file_name, **kw):
                    from . import cmmfiles
                    return cmmfiles.read_cmm(session, data)
        else:
            from chimerax.save_command import SaverInfo
            class MarkerInfo(SaverInfo):
                def save(self, session, path, *, models=None, **kw):
                    from . import cmmfiles
                    return cmmfiles.write_cmm(session, path, models)

                @property
                def save_args(self):
                    from chimerax.core.commands import ModelsArg
                    return { 'models': ModelsArg }
        return MarkerInfo()

bundle_api = _MarkersAPI()

from .markers import MarkerSet, create_link, selected_markers, selected_links
from .mouse import mark_map_center
