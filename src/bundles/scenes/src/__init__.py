# vim: set expandtab ts=4 sw=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2025 Regents of the University of California. All rights reserved.
# The ChimeraX application is provided pursuant to the ChimeraX license
# agreement, which covers academic and commercial uses. For more details, see
# <https://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
#
# You can also
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
__version__ = "0.2.4"

from chimerax.core.toolshed import BundleAPI


class _ScenesBundleAPI(BundleAPI):
    api_version = 1

    @staticmethod
    def get_class(class_name):
        if class_name == "SceneManager":
            from . import manager

            return manager.SceneManager
        elif class_name == "ScenesTool":
            from . import tool

            return tool.ScenesTool

    @staticmethod
    def initialize(session, bundle_info):
        """Install scene manager into existing session"""
        from .manager import SceneManager

        session.scenes = SceneManager(session)

    @staticmethod
    def register_command(bi, ci, logger):
        # 'register_command' is lazily called when the command is referenced
        from . import cmd

        cmd.register_commands(logger)

    @staticmethod
    def start_tool(session, bi, ti):
        if ti.name == "Scenes":
            from .tool import ScenesTool
            return ScenesTool(session, ti.name)
        raise ValueError("unknown tool %s" % ti.name)


bundle_api = _ScenesBundleAPI()
