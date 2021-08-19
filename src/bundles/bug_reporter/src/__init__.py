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

class _BugReporterAPI(BundleAPI):

    @staticmethod
    def initialize(session, bundle_info):
        '''Check Mac crash logs after startup.'''
        from . import crash_report
        crash_report.check_for_crash(session)
        crash_report.register_signal_handler(session)

        # Add Report a Bug to Help menu
        from . import bug_reporter_gui
        bug_reporter_gui.add_help_menu_entry(session)

bundle_api = _BugReporterAPI()

from .bug_reporter_gui import show_bug_reporter
