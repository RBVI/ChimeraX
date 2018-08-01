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
    def start_tool(session, tool_name):
        from .bug_reporter_gui import BugReporter
        tool = BugReporter(session, tool_name)
        return tool

    @staticmethod
    def initialize(session, bundle_info):
        '''Check if for Mac crash logs after startup.'''
        from . import mac_crash_report
        mac_crash_report.register_mac_crash_checker(session)

bundle_api = _BugReporterAPI()

def show_bug_reporter(session):
    ts = session.toolshed
    bi, tool_name = ts.find_bundle_for_tool('Bug Reporter')
    tool = bi.start_tool(session, tool_name)
    return tool
