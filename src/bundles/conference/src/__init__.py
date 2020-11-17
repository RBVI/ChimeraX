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

class _ConferenceAPI(BundleAPI):

    api_version = 1

    @staticmethod
    def start_tool(session, bi, ti):
        from .tool import ConferenceUI
        return ConferenceUI(session, ti.name)

    @staticmethod
    def register_command(bi, ci, logger):
        # 'register_command' is lazily called when the command is referenced
        cmd_name = ci.name
        if cmd_name.startswith('conference'):
            from . import reg_cmd
            reg_cmd.register_command(ci, logger)

bundle_api = _ConferenceAPI()
