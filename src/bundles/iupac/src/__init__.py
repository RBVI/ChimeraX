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

class _IupacAPI(BundleAPI):

    @staticmethod
    def run_provider(session, name, mgr, **kw):
        if mgr == session.open_command:
            from chimerax.open_command import FetcherInfo
            class IupacFetcherInfo(FetcherInfo):
                def fetch(self, session, ident, format_name, ignore_cache, *, res_name=None, **kw):
                    from .import iupac
                    return iupac.fetch_iupac(session, ident, ignore_cache=ignore_cache, res_name=res_name)

                @property
                def fetch_args(self):
                    from chimerax.core.commands import StringArg
                    return { 'res_name': StringArg }
            return IupacFetcherInfo()

        from .build_ui import IupacProvider
        return IupacProvider(session)

bundle_api = _IupacAPI()
