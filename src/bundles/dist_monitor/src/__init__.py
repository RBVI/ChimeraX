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

class _DistMonitorBundleAPI(BundleAPI):

    @staticmethod
    def get_class(class_name):
        if class_name == "DistancesMonitor":
            from . import monitor
            return monitor.DistancesMonitor

    @staticmethod
    def initialize(session, bundle_info):
        """Install distance monitor into existing session"""
        from . import settings
        settings.init(session)

        from .monitor import DistancesMonitor
        session.pb_dist_monitor = DistancesMonitor(session, bundle_info)

    @staticmethod
    def finish(session, bundle_info):
        """De-install distance monitor from existing session"""
        del session.pb_dist_monitor

    '''
    @staticmethod
    def register_command(command_name, logger):
        # 'register_command' is lazily called when the command is referenced
        from . import cmd
        cmd.register_seqalign_command(logger)
    '''

bundle_api = _DistMonitorBundleAPI()
