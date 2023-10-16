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

class _FunctionKeyBundleAPI(BundleAPI):

    @staticmethod
    def initialize(session, bundle_info):
        """Register to get function key presses"""
        if not session.ui.is_gui:
            return
        def fkey_press(session, key):
            from Qt.QtCore import Qt
            key_num = key - Qt.Key.Key_F1 + 1
            from . import fkey
            fkey.function_key_pressed(session, key_num)

        from Qt.QtCore import Qt
        for k in range(Qt.Key.Key_F1, Qt.Key.Key_F15):
            session.ui.intercept_key(k, fkey_press)

    @staticmethod
    def register_command(command_name, logger):
        from . import fkey
        fkey.register_functionkey_command(logger)

bundle_api = _FunctionKeyBundleAPI()
