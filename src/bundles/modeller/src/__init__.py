# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2022 Regents of the University of California. All rights reserved.
# The ChimeraX application is provided pursuant to the ChimeraX license
# agreement, which covers academic and commercial uses. For more details, see
# <http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
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

class ModellerAPI(BundleAPI):

    @staticmethod
    def get_class(class_name):
        if class_name == "ModellerResultsViewer":
            from .tool import ModellerResultsViewer
            return ModellerResultsViewer
        elif class_name == "ModellerLauncher":
            from .tool import ModellerLauncher
            return ModellerLauncher
        elif class_name == "ModellerWebJob":
            from .common import ModellerWebJob
            return ModellerWebJob
        from . import comparative
        return getattr(comparative, class_name)

    @staticmethod
    def register_command(command_name, logger):
        from . import cmd
        cmd.register_command(logger)

    @staticmethod
    def start_tool(session, tool_name):
        from .tool import ModellerLauncher
        return ModellerLauncher(session, tool_name)


bundle_api = ModellerAPI()
