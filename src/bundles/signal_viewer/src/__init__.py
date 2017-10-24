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

class _SignalViewerBundle(BundleAPI):

    @staticmethod
    def get_class(class_name):
        # 'get_class' is called by session code to get class saved in a session
        if class_name == 'Signals':
            from . import signals
            return signals.Signals
        return None

    @staticmethod
    def open_file(session, path):
        # 'open_file' is called by session code to open a file
        # returns (list of models, status message)
        from . import signals
        return signals.read_signals(session, path)

    @staticmethod
    def register_command(command_name, logger):
        # 'register_command' is lazily called when the command is referenced
        from . import signals
        signals.register_signal_command(logger)

bundle_api = _SignalViewerBundle()
