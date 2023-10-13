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
            session.presets = PresetsManager(session, name)

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
