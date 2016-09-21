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

    @staticmethod
    def start_tool(session, bundle_info):
        # 'start_tool' is called to start an instance of the tool
        # If providing more than one tool in package,
        # look at the name in 'bundle_info.name' to see which is being started.
        from .gui import ShellUI
        return ShellUI(session, bundle_info)     # UI should register itself with tool state manager

    @staticmethod
    def get_class(class_name):
        # 'get_class' is called by session code to get class saved in a session
        if class_name == 'ShellUI':
            from . import gui
            return gui.ShellUI
        return None

bundle_api = _MyAPI()
