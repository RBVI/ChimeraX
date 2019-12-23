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
