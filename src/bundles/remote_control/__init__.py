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

    @staticmethod
    def start_tool(session, bundle_info):
        from .remotecmd import remote_control
        remote_control(session, enable = True)	# Start XMLRPC server

    @staticmethod
    def register_command(command_name, bundle_info):
        from . import remotecmd
        remotecmd.register_remote_control_command()

bundle_api = _MyAPI()
