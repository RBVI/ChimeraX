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

class _MyAPI(BundleAPI):

    @staticmethod
    def start_tool(session, tool_name):
        from .tool import panel_classes
        cls = panel_classes[tool_name]
        spanel = cls.get_singleton(session)
        if spanel is not None:
            spanel.display(True)

        # TODO: Is there a better place to register selectors?
        from . import shortcuts
        shortcuts.register_selectors(session)

        return spanel

    @staticmethod
    def register_command(command_name):
        # 'register_command' is lazily called when command is referenced
        from . import shortcuts
        shortcuts.register_shortcut_command()

    @staticmethod
    def get_class(class_name):
        # 'get_class' is called by session code to get class saved in a session
        from .tool import panel_classes
        for c in panel_classes.values():
            if class_name == c.__name__:
                return c
        return None

bundle_api = _MyAPI()
