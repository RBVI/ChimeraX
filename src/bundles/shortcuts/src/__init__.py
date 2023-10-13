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

class _ShortcutsAPI(BundleAPI):

    @staticmethod
    def start_tool(session, tool_name):
        from .tool import panel_classes
        cls = panel_classes[tool_name]
        spanel = cls.get_singleton(session)
        if spanel is not None:
            spanel.display(True)
        return spanel

    @staticmethod
    def register_command(command_name, logger):
        # 'register_command' is lazily called when command is referenced
        from . import shortcuts
        if command_name == 'ks':
            shortcuts.register_shortcut_command(logger)
        elif command_name == 'snapshot':
            shortcuts.register_snapshot_command(logger)

    @staticmethod
    def register_selector(selector_name, logger):
        # 'register_selector' is lazily called when command is referenced
        from . import shortcuts
        shortcuts.register_selectors(logger)

    @staticmethod
    def get_class(class_name):
        # 'get_class' is called by session code to get class saved in a session
        from .tool import panel_classes
        for c in panel_classes.values():
            if class_name == c.__name__:
                return c
        return None

    @staticmethod
    def run_provider(session, name, mgr, **kw):
        """Run toolbar provider"""
        from . import shortcuts
        shortcuts.run_provider(session, name)


bundle_api = _ShortcutsAPI()
