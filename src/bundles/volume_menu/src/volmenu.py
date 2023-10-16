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

# -----------------------------------------------------------------------------
# Panel for erasing parts of map in sphere with map eraser mouse mode.
#
from chimerax.core.tools import ToolInstance
class VolumeMenu(ToolInstance):

    def __init__(self, session, tool_name):
        
        self._shown = False

        ToolInstance.__init__(self, session, tool_name)
        
    @classmethod
    def get_singleton(self, session, create=True):
        from chimerax.core import tools
        return tools.get_singleton(session, VolumeMenu, 'Volume Menu', create=create)

    @property
    def shown(self):
        return self._shown

    def display(self, show):
        if show:
            self.show()
        else:
            self.hide()

    def displayed(self):
        return self._shown
    
    def show(self):
        if self._shown:
            return
        self._shown = True
        global settings
        settings.show_volume_menu = True
        for tool in self._volume_tools():
            if tool.name == 'Show Volume Menu':
                continue	# Add a hide menu entry at end.
            def callback(*, ses = self.session, tool_name=tool.name, vmenu = self):
                from chimerax.core.commands import run, StringArg
                run(ses, "ui tool show %s" % StringArg.unparse(tool_name))
            self.session.ui.main_window.add_menu_entry(['Volume'], tool.name, callback)
        self.session.ui.main_window.add_menu_entry(['Volume'], 'Hide Volume Menu', self.hide)

    def hide(self):
        if not self._shown:
            return
        self._shown = False
        global settings
        settings.show_volume_menu = False
        self.session.ui.main_window.remove_menu(['Volume'])

    def toggle(self):
        if self.shown:
            self.hide()
        else:
            self.show()

    def _volume_tools(self):
        tools = []
        ses = self.session
        for bi in ses.toolshed.bundle_info(ses.logger):
            for tool in bi.tools:
                if 'Volume Data' in tool.categories:
                    tools.append(tool)
        tools.sort(key = lambda t: t.name)
        return tools


from chimerax.core.settings import Settings
class _VolumeMenuSettings(Settings):
    AUTO_SAVE = {
        'show_volume_menu': False,
    }

settings = None	# Set by bundle initialization code _VolumeMenuAPI.initialize().
