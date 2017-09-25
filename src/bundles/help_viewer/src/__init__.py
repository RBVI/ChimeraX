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


class _MyAPI(BundleAPI):

    @staticmethod
    def register_command(command_name, logger):
        # 'register_command' is lazily called when command is referenced
        from . import cmd
        from chimerax.core.commands import register
        register(command_name, cmd.help_desc, cmd.help, logger=logger)

    @staticmethod
    def open_file(session, path, file_name, new=False):
        # 'open_file' is called by session code to open a file
        import os
        path = os.path.abspath(path)
        from urllib.parse import urlunparse
        from urllib.request import pathname2url
        url = urlunparse(('file', '', pathname2url(path), '', '', ''))
        show_url(session, url, new=new)
        return [], "Opened %s" % file_name

    @staticmethod
    def get_class(class_name):
        # 'get_class' is called by session code to get class saved in a session
        if class_name == 'HelpUI':
            from . import tool
            return tool.HelpUI
        return None


def show_url(session, url, *, new=False):
    if session.ui.is_gui:
        from .tool import HelpUI
        help_viewer = HelpUI.get_viewer(session)
        help_viewer.show(url, new=new)
    else:
        import webbrowser
        webbrowser.open(url, new=new)


bundle_api = _MyAPI()
