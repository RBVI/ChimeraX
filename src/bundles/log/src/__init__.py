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

class _MyAPI(BundleAPI):

    @staticmethod
    def initialize(session, bundle_info):
        from . import settings
        settings.settings = settings._LogSettings(session, "Log")

        if session.ui.is_gui:
            session.ui.triggers.add_handler('ready',
                lambda *args, ses=session: settings.register_settings_options(ses))

    @staticmethod
    def get_class(class_name):
        # 'get_class' is called by session code to get class saved in a session
        if class_name == 'Log':
            from . import tool
            return tool.Log
        return None

    @staticmethod
    def start_tool(session, tool_name):
        # 'start_tool' is called to start an instance of the tool
        from . import cmd
        return cmd.get_singleton(session, create=True)

    @staticmethod
    def register_command(command_name, logger):
        # 'register_command' is lazily called when command is referenced
        if command_name == "echo":
            from chimerax.core.commands import create_alias
            create_alias("echo", "log text $*", logger=logger)
        elif command_name.startswith("log"):
            from . import cmd
            cmd.register_log_command(logger)

bundle_api = _MyAPI()
