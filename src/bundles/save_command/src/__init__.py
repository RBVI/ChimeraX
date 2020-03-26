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

class SaverInfo:
    def save(self, session, path, **kw):
        raise NotImplementedError("Saver did not implement mandatory 'save' method")

    @property
    def save_args(self):
        return {}

    @property
    def hidden_args(self):
        return []

    def save_args_widget(self, session):
        return None

    def save_args_string_from_widget(self, widget):
        raise NotImplementedError("Saver did not implement 'save_args_string_from_widget' method")

from .manager import NoSaverError
from .dialog import show_save_dialog

from chimerax.core.toolshed import BundleAPI
class _OpenBundleAPI(BundleAPI):

    @staticmethod
    def init_manager(session, bundle_info, name, **kw):
        """Initialize save-command manager"""
        if session.ui.is_gui:
            from . import dialog
            session.ui.triggers.add_handler('ready',
                lambda *args, ses=session: dialog.create_menu_entry(ses))
        from . import manager
        session.save_command = manager.SaveManager(session)
        return session.save_command

    @staticmethod
    def register_command(command_name, logger):
        from . import cmd
        cmd.register_command(command_name, logger)

bundle_api = _OpenBundleAPI()
