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
    def register_command(command_name):
        # 'register_command' is lazily called when command is referenced
        from . import cmd
        from chimerax.core.commands import register
        register(command_name, cmd.help_desc, cmd.help)

    @staticmethod
    def open_file(session, f, name, filespec=None, **kw):
        # 'open_file' is called by session code to open a file
        from . import cmd
        import os.path
        cmd.help(session, "file:" + os.path.realpath(filespec))
        return [], "Opened %s" % name

    @staticmethod
    def get_class(class_name):
        # 'get_class' is called by session code to get class saved in a session
        if class_name == 'HelpUI':
            from . import tool
            return tool.HelpUI
        return None

bundle_api = _MyAPI()
