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

class _PresetsBundleAPI(BundleAPI):

    @staticmethod
    def get_class(class_name):
        if class_name == "PresetsManager":
            # PresetsManager used to be a state manager, so for session compatibility...
            from chimerax.core.state import State
            class FakePresetsManager(State):
                @staticmethod
                def reset_state(self, session):
                    pass
                def restore_snapshot(session, data):
                    return session.presets
                def take_snapshot(self, session, flags):
                    return None
            return FakePresetsManager

    @staticmethod
    def init_manager(session, bundle_info, name, **kw):
        """Initialize presets manager"""
        if name == "presets":
            from .manager import PresetsManager
            session.presets = PresetsManager(session)
            return session.presets

    @staticmethod
    def run_provider(session, name, mgr, **kw):
        """Invoke presets provider"""
        from .builtin import run_preset
        run_preset(session, name, mgr, **kw)

    @staticmethod
    def finish(session, bundle_info):
        """De-install presets manager from existing session"""
        del session.presets

    @staticmethod
    def register_command(command_name, logger):
        # 'register_command' is lazily called when the command is referenced
        from . import cmd
        cmd.register_preset_command(logger)

bundle_api = _PresetsBundleAPI()
