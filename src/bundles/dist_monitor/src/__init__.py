# vim: set expandtab shiftwidth=4 softtabstop=4:

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
from .cmd import SimpleMeasurable, ComplexMeasurable

class _DistMonitorBundleAPI(BundleAPI):

    @staticmethod
    def get_class(class_name):
        if class_name == "DistancesMonitor":
            from . import monitor
            return monitor.DistancesMonitor

    @staticmethod
    def initialize(session, bundle_info = None):
        """Install distance monitor into existing session"""
        from . import settings
        settings.settings = settings._DistanceSettings(session, "distances")

        from .monitor import DistancesMonitor
        session.pb_dist_monitor = DistancesMonitor(session)

        if session.ui.is_gui:
            session.ui.triggers.add_handler('ready',
                lambda *args, ses=session: settings.register_settings_options(ses))

    @staticmethod
    def finish(session, bundle_info):
        """De-install distance monitor from existing session"""
        del session.pb_dist_monitor

    @staticmethod
    def register_command(command_name, logger):
        # 'register_command' is lazily called when the command is referenced
        from . import cmd
        cmd.register_command(logger)

bundle_api = _DistMonitorBundleAPI()
