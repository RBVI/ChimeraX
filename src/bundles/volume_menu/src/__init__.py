# vim: set expandtab ts=4 sw=4:

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

class _VolumeMenuAPI(BundleAPI):

    @staticmethod
    def start_tool(session, tool_name):
        from .volmenu import VolumeMenu
        m = VolumeMenu.get_singleton(session)
        return m

    @staticmethod
    def initialize(session, bundle_info):
        """Register map eraser mouse mode."""
        from . import volmenu
        volmenu.settings = volmenu._VolumeMenuSettings(session, "volume menu")
        if session.ui.is_gui:
            if volmenu.settings.show_volume_menu:
                # Delay creating volume menu until all tools are initialized.
                def show_volume_menu(*args, ses=session):
                    from .volmenu import VolumeMenu
                    m = VolumeMenu.get_singleton(session)
                    m.show()
                session.ui.triggers.add_handler('ready', show_volume_menu)

    @staticmethod
    def finish(session, bundle_info):
        pass

bundle_api = _VolumeMenuAPI()
