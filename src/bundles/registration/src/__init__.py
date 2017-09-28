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


class _MyAPI(BundleAPI):

    api_version = 1

    @staticmethod
    def register_command(bundle_info, command_info, logger):
        from . import cmd
        from chimerax.core.commands import register
        desc = cmd.register_desc
        if desc.synopsis is None:
            desc.synopsis = command_info.synopsis
        register(command_info.name, desc, cmd.register)

    @staticmethod
    def start_tool(session, bundle_info, tool_info, **kw):
        from .gui import RegistrationUI
        return RegistrationUI(session, tool_info.name, **kw)

    @staticmethod
    def initialize(session, bundle_info):
        session.logger.info("Initializing registration")

    @staticmethod
    def finish(session, bundle_info):
        pass

bundle_api = _MyAPI()
