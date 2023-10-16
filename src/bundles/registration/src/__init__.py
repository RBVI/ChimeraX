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


class _MyAPI(BundleAPI):

    api_version = 1

    @staticmethod
    def register_command(bi, ci, logger):
        from . import cmd
        from chimerax.core.commands import register
        func_attr = ci.name.replace(' ', '_')
        func = getattr(cmd, func_attr)
        desc_attr = func_attr + "_desc"
        desc = getattr(cmd, desc_attr)
        if desc.synopsis is None:
            desc.synopsis = ci.synopsis
        register(ci.name, desc, func)

    @staticmethod
    def start_tool(session, bi, ti):
        from .gui import RegistrationUI
        return RegistrationUI(session, ti)

    @staticmethod
    def initialize(session, bi):
        from .nag import nag
        nag(session)

    @staticmethod
    def finish(session, bi):
        pass


bundle_api = _MyAPI()
