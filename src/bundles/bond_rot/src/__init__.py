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

class _BondRotBundleAPI(BundleAPI):

    @staticmethod
    def get_class(class_name):
        if class_name == "BondRotationManager":
            from . import manager
            return manager.BondRotationManager
        elif class_name == "BondRotation":
            from . import bond_rot
            return bond_rot.BondRotation
        elif class_name == "BondRotater":
            from . import bond_rot
            return bond_rot.BondRotater

    @staticmethod
    def initialize(session, bundle_info):
        """Install bond-rotation manager into existing session"""
        from .manager import BondRotationManager
        session.bond_rotations = BondRotationManager(session, bundle_info)
        if session.ui.is_gui:
            mm = session.ui.mouse_modes
            from .mouse_rot import BondRotationMouseMode
            mm.add_mode(BondRotationMouseMode(session))

    @staticmethod
    def finish(session, bundle_info):
        """De-install bond-rotation manager from existing session"""
        del session.bond_rotations

    @staticmethod
    def register_command(command_name, logger):
        # 'register_command' is lazily called when the command is referenced
        from . import cmd
        cmd.register_command(command_name, logger)

bundle_api = _BondRotBundleAPI()
