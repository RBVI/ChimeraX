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

from .series import MapSeries
from .vseries_command import register_vseries_command
from .play import PlaySeriesMouseMode

# -----------------------------------------------------------------------------
#
from chimerax.core.toolshed import BundleAPI

class _MapSeriesBundle(BundleAPI):

    @staticmethod
    def register_command(command_name, logger):
        # 'register_command' is lazily called when the command is referenced
        if command_name == 'vseries':
            from . import vseries_command
            vseries_command.register_vseries_command(logger)
        elif command_name == 'measure motion':
            from . import measure_motion
            measure_motion.register_command(logger)

    @staticmethod
    def initialize(session, bundle_info):
        # 'initialize' is called by the toolshed on start up
        if session.ui.is_gui:
            from . import play
            play.register_mousemode(session)
            from . import slider
            slider.show_slider_on_open(session)

    @staticmethod
    def finish(session, bundle_info):
        # 'finish' is called by the toolshed when updated/reloaded
        if session.ui.is_gui:
            from . import slider
            slider.remove_slider_on_open(session)

    @staticmethod
    def get_class(class_name):
        # 'get_class' is called by session code to get class saved in a session
        if class_name == 'MapSeries':
            from . import series
            return series.MapSeries
        return None
        
bundle_api = _MapSeriesBundle()
