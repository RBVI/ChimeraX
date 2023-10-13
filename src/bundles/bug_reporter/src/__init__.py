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

class _BugReporterAPI(BundleAPI):

    @staticmethod
    def initialize(session, bundle_info):
        '''
        Check crash logs at startup and register faulthandler
        and the log recording temporary file to record if the
        current session crashes.
        '''
        from sys import platform
        if platform != 'win32':
            from . import crash_report
            crash_report.check_for_crash(session)
            crash_report.register_signal_handler(session)
            crash_report.register_log_recorder(session)

        # Add Report a Bug to Help menu
        from . import bug_reporter_gui
        bug_reporter_gui.add_help_menu_entry(session)

bundle_api = _BugReporterAPI()

from .bug_reporter_gui import show_bug_reporter, system_summary, opengl_info
