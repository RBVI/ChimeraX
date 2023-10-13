# vim: set et sw=4 sts=4:
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
# from chimerax.core
__version__ = "1.0"

from chimerax.core.commands import register
from chimerax.core.toolshed import BundleAPI
from chimerax.core.tools import get_singleton
from .cmd import *


class _MyAPI(BundleAPI):

    api_version = 1

    @staticmethod
    def get_class(class_name):
        pass

    @staticmethod
    def start_tool(session, bi, ti):
        if ti.name == "Task Manager":
            from .tool import TaskManager
            return get_singleton(session, TaskManager, "Task Manager")

    @staticmethod
    def register_command(bi, ci, logger):
        register(ci.name, taskman_desc, taskman, logger=logger)


bundle_api = _MyAPI()
