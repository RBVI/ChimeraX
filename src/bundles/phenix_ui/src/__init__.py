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

class _PhenixBundle(BundleAPI):

    @staticmethod
    def get_class(class_name):
        from . import tool
        return getattr(tool, class_name)

    @staticmethod
    def register_command(command_name, logger):
        from . import cmd
        cmd.register_command(logger)

    @staticmethod
    def start_tool(session, tool_name):
        if tool_name == 'Water Placement':
            from .tool import LaunchDouseTool
            return LaunchDouseTool(session, tool_name)
        if tool_name == 'Local EM Fitting':
            from .tool import LaunchEmplaceLocalTool
            return LaunchEmplaceLocalTool(session, tool_name)

bundle_api = _PhenixBundle()
