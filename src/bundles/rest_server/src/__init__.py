# vim: set expandtab shiftwidth=4 softtabstop=4:

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
    def register_command(command_name, logger):
        # 'register_command' is lazily called when the command is referenced
        from . import cmd
        from chimerax.core.commands import register
        register(command_name + " start",
                 cmd.start_desc, cmd.start_server, logger=logger)
        register(command_name + " port",
                 cmd.port_desc, cmd.report_port, logger=logger)
        register(command_name + " stop",
                 cmd.stop_desc, cmd.stop_server, logger=logger)

    @staticmethod
    def initialize(session, bi):
        # bundle-specific initialization (causes import)
        pass

    @staticmethod
    def finish(session, bi):
        # deinitialize bundle in session
        from . import cmd
        cmd.stop_server(session, quiet=True)

    @staticmethod
    def get_class(class_name):
        if class_name == "RESTServer":
            from . import server
            return server.RESTServer
        return None

bundle_api = _MyAPI()
