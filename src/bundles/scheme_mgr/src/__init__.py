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


class _SchemesBundleAPI(BundleAPI):

    @staticmethod
    def get_class(class_name):
        if class_name == "SchemesManager":
            from . import manager
            return manager.SchemesManager

    @staticmethod
    def init_manager(session, bundle_info, name, **kw):
        """Initialize schemes manager"""
        if name == "url_schemes":
            from .manager import SchemesManager
            session.url_schemes = SchemesManager(session, name)

    @staticmethod
    def run_provider(session, name, mgr, **kw):
        """Run scheme provider (which does nothing)"""
        return


bundle_api = _SchemesBundleAPI()
