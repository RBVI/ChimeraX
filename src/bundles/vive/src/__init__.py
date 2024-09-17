# vim: set expandtab ts=4 sw=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2016 Regents of the University of California.
# All rights reserved.  This software provided pursuant to a
# license agreement containing restrictions on its disclosure,
# duplication and use.  For details see:
# https://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
# This notice must be embedded in or attached to all copies,
# including partial copies, of the software or any revisions
# or derivations thereof.
# === UCSF ChimeraX Copyright ===

from chimerax.core.toolshed import BundleAPI

class _VRAPI(BundleAPI):

    @staticmethod
    def register_command(command_name, logger):
        # 'register_command' is lazily called when the command is referenced
        if command_name == 'vr':
            from . import vr
            vr.register_vr_command(logger)
        elif command_name == 'xr':
            from . import xr
            xr.register_vr_command(logger)

    @staticmethod
    def initialize(session, bundle_info):
        # 'initialize' is called by the toolshed on start up
        # Allow tools to register for vr updates before vr is started, e.g. meeting.
        session.triggers.add_trigger('vr started')
        session.triggers.add_trigger('vr update')
        session.triggers.add_trigger('vr stopped')

    @staticmethod
    def finish(session, bundle_info):
        # 'finish' is called by the toolshed when updated/reloaded
        pass

    @staticmethod
    def get_class(class_name):
        # 'get_class' is called by session code to get class saved in a session
        if class_name == 'SteamVRCamera':
            from . import vr
            return vr.SteamVRCamera
        if class_name == 'OpenXRCamera':
            from . import xr
            return xr.OpenXRCamera
        return None

bundle_api = _VRAPI()
